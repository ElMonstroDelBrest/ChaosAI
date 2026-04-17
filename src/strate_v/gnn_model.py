"""OnChainGNN: GraphSAGE + GATv2 model for on-chain transaction analysis.

Architecture (per the plan):
  Input: (N, 8) node_feat + (2, E) edge_index + (E, 2) edge_attr
    -> SAGEConv(8->128) + GATv2Conv(128->128, heads=4) + LN + ELU
    -> SAGEConv(128->256) + GATv2Conv(256->256, heads=4) + LN + ELU
    -> SAGEConv(256->256) + GATv2Conv(256->256, heads=4) + LN + ELU
    -> Global pool: mean || max || attention -> (768,)
    -> MLP(768->512->256) + LN
    -> Output: (256,)  <- gnn_dim (MXU-aligned: 2x128)

~450K params. Three self-supervised loss heads:
  - Link prediction (BCE)
  - Temporal contrastive (InfoNCE)
  - Exchange flow prediction (MSE)
"""

import torch
from torch import Tensor, nn
from torch_geometric.nn import SAGEConv, GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.data import Data


class AttentionPool(nn.Module):
    """Attention-based global graph pooling."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.gate = nn.Linear(in_dim, 1)

    def forward(self, x: Tensor, batch: Tensor | None = None) -> Tensor:
        """Pool node features to graph-level via attention.

        Args:
            x: (N, D) node features.
            batch: (N,) batch assignment vector (None for single graph).

        Returns:
            (B, D) pooled graph features.
        """
        gate_logits = self.gate(x)  # (N, 1)

        if batch is None:
            # Single graph: simple softmax
            weights = torch.softmax(gate_logits, dim=0)  # (N, 1)
            return (weights * x).sum(dim=0, keepdim=True)  # (1, D)

        # Batched: scatter softmax
        from torch_geometric.utils import softmax as pyg_softmax
        weights = pyg_softmax(gate_logits.squeeze(-1), batch).unsqueeze(-1)  # (N, 1)

        # Scatter sum
        num_graphs = batch.max().item() + 1
        out = torch.zeros(num_graphs, x.size(1), device=x.device, dtype=x.dtype)
        out.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), weights * x)
        return out  # (B, D)


class MessagePassingBlock(nn.Module):
    """One SAGE + GATv2 + LayerNorm + ELU block."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, edge_dim: int | None = None):
        super().__init__()
        self.sage = SAGEConv(in_dim, out_dim)
        # GATv2: heads * out_dim -> out_dim via concat=False (average heads)
        self.gat = GATv2Conv(out_dim, out_dim, heads=heads, concat=False, edge_dim=edge_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ELU()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        x = self.sage(x, edge_index)
        x = self.gat(x, edge_index, edge_attr=edge_attr)
        x = self.norm(x)
        x = self.act(x)
        return x


class OnChainGNN(nn.Module):
    """GNN for on-chain transaction graph analysis.

    Args:
        node_features: Input node feature dimension (default 8).
        edge_features: Input edge feature dimension (default 2).
        hidden_dim: Hidden dimension (default 128).
        gnn_dim: Output embedding dimension (default 256, MXU-aligned).
        n_layers: Number of message-passing blocks (default 3).
        gat_heads: Number of GAT attention heads (default 4).
        dropout: Dropout rate (default 0.1).
    """

    def __init__(
        self,
        node_features: int = 8,
        edge_features: int = 2,
        hidden_dim: int = 128,
        gnn_dim: int = 256,
        n_layers: int = 3,
        gat_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert gnn_dim % 128 == 0, f"gnn_dim must be MXU-aligned (multiple of 128), got {gnn_dim}"

        self.gnn_dim = gnn_dim
        self.dropout = dropout

        # Message-passing layers
        dims = [node_features] + [hidden_dim] * (n_layers - 1) + [gnn_dim]
        self.mp_blocks = nn.ModuleList()
        for i in range(n_layers):
            edge_dim = edge_features if i == 0 else None  # Only first layer uses edge features
            self.mp_blocks.append(
                MessagePassingBlock(dims[i], dims[i + 1], heads=gat_heads, edge_dim=edge_dim)
            )

        # Triple pooling: mean || max || attention -> 3 * gnn_dim
        self.attn_pool = AttentionPool(gnn_dim)
        pool_dim = 3 * gnn_dim  # 768

        # Readout MLP: 768 -> 512 -> 256
        self.readout = nn.Sequential(
            nn.Linear(pool_dim, 512),      # 768 -> 512
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(512, gnn_dim),       # 512 -> 256
            nn.LayerNorm(gnn_dim),
        )

        # Loss heads
        self.link_head = nn.Bilinear(gnn_dim, gnn_dim, 1)  # Link prediction
        self.flow_head = nn.Linear(gnn_dim, 1)  # Exchange net flow prediction

    def encode(self, data: Data) -> Tensor:
        """Encode graph to node embeddings.

        Args:
            data: PyG Data with x, edge_index, edge_attr.

        Returns:
            (N, gnn_dim) node embeddings.
        """
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None

        for i, block in enumerate(self.mp_blocks):
            ea = edge_attr if i == 0 else None
            x = block(x, edge_index, edge_attr=ea)

        return x

    def forward(self, data: Data) -> Tensor:
        """Forward pass: encode graph to a single embedding vector.

        Args:
            data: PyG Data with x, edge_index, edge_attr, batch (optional).

        Returns:
            (B, gnn_dim) or (gnn_dim,) graph-level embedding.
        """
        node_emb = self.encode(data)  # (N, gnn_dim)
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else None

        # Triple pooling
        mean_pool = global_mean_pool(node_emb, batch)  # (B, gnn_dim)
        max_pool = global_max_pool(node_emb, batch)    # (B, gnn_dim)
        attn_pool = self.attn_pool(node_emb, batch)    # (B, gnn_dim)

        pooled = torch.cat([mean_pool, max_pool, attn_pool], dim=-1)  # (B, 3*gnn_dim)
        out = self.readout(pooled)  # (B, gnn_dim)

        if out.shape[0] == 1:
            return out.squeeze(0)  # (gnn_dim,)
        return out

    def link_prediction_loss(self, node_emb: Tensor, pos_edges: Tensor, neg_edges: Tensor) -> Tensor:
        """Binary cross-entropy link prediction loss.

        Args:
            node_emb: (N, gnn_dim) node embeddings.
            pos_edges: (E_pos, 2) positive edge indices.
            neg_edges: (E_neg, 2) negative edge indices.

        Returns:
            Scalar BCE loss.
        """
        pos_src = node_emb[pos_edges[:, 0]]
        pos_dst = node_emb[pos_edges[:, 1]]
        neg_src = node_emb[neg_edges[:, 0]]
        neg_dst = node_emb[neg_edges[:, 1]]

        pos_score = self.link_head(pos_src, pos_dst).squeeze(-1)
        neg_score = self.link_head(neg_src, neg_dst).squeeze(-1)

        pos_loss = nn.functional.binary_cross_entropy_with_logits(
            pos_score, torch.ones_like(pos_score)
        )
        neg_loss = nn.functional.binary_cross_entropy_with_logits(
            neg_score, torch.zeros_like(neg_score)
        )
        return (pos_loss + neg_loss) / 2

    def contrastive_loss(
        self, emb_t: Tensor, emb_tp1: Tensor, temperature: float = 0.07
    ) -> Tensor:
        """InfoNCE temporal contrastive loss between consecutive snapshots.

        Args:
            emb_t: (B, gnn_dim) graph embeddings at time t.
            emb_tp1: (B, gnn_dim) graph embeddings at time t+1.
            temperature: Softmax temperature.

        Returns:
            Scalar InfoNCE loss.
        """
        emb_t = nn.functional.normalize(emb_t, dim=-1)
        emb_tp1 = nn.functional.normalize(emb_tp1, dim=-1)

        logits = emb_t @ emb_tp1.T / temperature  # (B, B)
        labels = torch.arange(logits.size(0), device=logits.device)
        return nn.functional.cross_entropy(logits, labels)

    def flow_prediction_loss(self, graph_emb: Tensor, target_flow: Tensor) -> Tensor:
        """MSE loss for exchange net inflow/outflow prediction.

        Args:
            graph_emb: (B, gnn_dim) graph embeddings.
            target_flow: (B,) actual net flow values.

        Returns:
            Scalar MSE loss.
        """
        pred = self.flow_head(graph_emb).squeeze(-1)
        return nn.functional.mse_loss(pred, target_flow)
