"""OnChainGNN: JAX/Flax port of the GraphSAGE + GATv2 model (Strate V).

Architecture (mirrors PyTorch version exactly):
  Input: jraph.GraphsTuple with nodes (N, 8), edges (E, 2), senders/receivers
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

Uses jraph.GraphsTuple for graph representation. Batching via jraph.batch
merges graphs with offset indices. Segment operations (segment_sum,
segment_max) replace PyG's scatter operations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jraph
import optax
from flax import linen as nn
from jax import Array


# ---------------------------------------------------------------------------
# SAGEConv layer: mean-aggregate neighbors + Linear([self || agg])
# ---------------------------------------------------------------------------

class SAGEConvLayer(nn.Module):
    """GraphSAGE convolution: mean-neighbor aggregation + linear projection.

    Equivalent to PyG's SAGEConv with default (mean) aggregation.
    out = Linear([x_self || mean_neighbors])
    """
    out_features: int

    @nn.compact
    def __call__(self, nodes: Array, senders: Array, receivers: Array,
                 n_node: int) -> Array:
        # Aggregate neighbor features via mean
        # senders[i] -> receivers[i]: message from sender to receiver
        neighbor_sum = jax.ops.segment_sum(
            nodes[senders], receivers, num_segments=n_node
        )
        # Count neighbors per node for mean (avoid /0)
        ones = jnp.ones(senders.shape[0])
        neighbor_count = jax.ops.segment_sum(ones, receivers, num_segments=n_node)
        neighbor_count = jnp.maximum(neighbor_count, 1.0)[:, None]
        neighbor_mean = neighbor_sum / neighbor_count

        # Concatenate self + aggregated neighbors -> project
        concat = jnp.concatenate([nodes, neighbor_mean], axis=-1)
        return nn.Dense(self.out_features)(concat)


# ---------------------------------------------------------------------------
# GATv2Conv layer: dynamic attention with LeakyReLU
# ---------------------------------------------------------------------------

class GATv2Layer(nn.Module):
    """GATv2 convolution with multi-head attention.

    Equivalent to PyG's GATv2Conv(concat=False) — heads are averaged.
    Attention: a^T * LeakyReLU(W_l[src] + W_r[dst] + W_e[edge]) -> softmax -> weighted sum.
    """
    out_features: int
    heads: int = 4
    edge_dim: int | None = None

    @nn.compact
    def __call__(self, nodes: Array, senders: Array, receivers: Array,
                 n_node: int, edge_attr: Array | None = None) -> Array:
        d = self.out_features

        # Per-head linear projections: (N, in) -> (N, heads, d)
        x_l = nn.Dense(self.heads * d, use_bias=False, name="W_l")(nodes)
        x_l = x_l.reshape(-1, self.heads, d)
        x_r = nn.Dense(self.heads * d, use_bias=False, name="W_r")(nodes)
        x_r = x_r.reshape(-1, self.heads, d)

        # Compute attention logits: LeakyReLU(x_l[src] + x_r[dst] [+ W_e*edge])
        msg = x_l[senders] + x_r[receivers]  # (E, heads, d)

        if edge_attr is not None and self.edge_dim is not None:
            edge_proj = nn.Dense(self.heads * d, use_bias=False, name="W_e")(edge_attr)
            edge_proj = edge_proj.reshape(-1, self.heads, d)
            msg = msg + edge_proj

        msg = nn.leaky_relu(msg, negative_slope=0.2)

        # Attention coefficients: a^T * msg -> (E, heads)
        attn_vec = self.param(
            "attn_vec",
            nn.initializers.glorot_uniform(),
            (self.heads, d),
        )
        attn_logits = (msg * attn_vec[None, :, :]).sum(axis=-1)  # (E, heads)

        # Numerically stable softmax per receiver per head
        attn_max = jax.ops.segment_max(
            attn_logits, receivers, num_segments=n_node
        )  # (N, heads)
        attn_logits = attn_logits - attn_max[receivers]  # subtract max for stability
        attn_exp = jnp.exp(attn_logits)  # (E, heads)
        attn_sum = jax.ops.segment_sum(
            attn_exp, receivers, num_segments=n_node
        )  # (N, heads)
        attn_weights = attn_exp / (attn_sum[receivers] + 1e-8)  # (E, heads)

        # Weighted message aggregation
        # Values: x_l[senders] (reuse the left-projected features)
        values = x_l[senders]  # (E, heads, d)
        weighted = values * attn_weights[:, :, None]  # (E, heads, d)
        agg = jax.ops.segment_sum(
            weighted, receivers, num_segments=n_node
        )  # (N, heads, d)

        # Average heads (concat=False)
        return agg.mean(axis=1)  # (N, d)


# ---------------------------------------------------------------------------
# MessagePassingBlock: SAGE + GAT + LayerNorm + ELU
# ---------------------------------------------------------------------------

class MessagePassingBlock(nn.Module):
    """One SAGE + GATv2 + LayerNorm + ELU block."""
    out_dim: int
    heads: int = 4
    edge_dim: int | None = None

    @nn.compact
    def __call__(self, nodes: Array, senders: Array, receivers: Array,
                 n_node: int, edge_attr: Array | None = None) -> Array:
        x = SAGEConvLayer(self.out_dim)(nodes, senders, receivers, n_node)
        ea = edge_attr if self.edge_dim is not None else None
        x = GATv2Layer(self.out_dim, heads=self.heads, edge_dim=self.edge_dim)(
            x, senders, receivers, n_node, edge_attr=ea
        )
        x = nn.LayerNorm()(x)
        x = nn.elu(x)
        return x


# ---------------------------------------------------------------------------
# AttentionPool: attention-weighted global graph pooling
# ---------------------------------------------------------------------------

class AttentionPool(nn.Module):
    """Attention-based global graph pooling.

    Computes per-node gate logits, applies softmax within each graph,
    then weighted sum to produce graph-level features.
    """

    @nn.compact
    def __call__(self, nodes: Array, graph_idx: Array, n_graphs: int) -> Array:
        """Pool node features to graph-level via attention.

        Args:
            nodes: (N_total, D) node features.
            graph_idx: (N_total,) graph assignment for each node.
            n_graphs: Number of graphs in the batch.

        Returns:
            (n_graphs, D) pooled graph features.
        """
        gate_logits = nn.Dense(1)(nodes).squeeze(-1)  # (N_total,)

        # Numerically stable per-graph softmax
        gate_max = jax.ops.segment_max(
            gate_logits, graph_idx, num_segments=n_graphs
        )  # (n_graphs,)
        gate_logits = gate_logits - gate_max[graph_idx]
        gate_exp = jnp.exp(gate_logits)  # (N_total,)
        gate_sum = jax.ops.segment_sum(
            gate_exp, graph_idx, num_segments=n_graphs
        )  # (n_graphs,)
        weights = gate_exp / (gate_sum[graph_idx] + 1e-8)  # (N_total,)

        # Weighted sum per graph
        weighted_nodes = nodes * weights[:, None]  # (N_total, D)
        return jax.ops.segment_sum(
            weighted_nodes, graph_idx, num_segments=n_graphs
        )  # (n_graphs, D)


# ---------------------------------------------------------------------------
# Bilinear head for link prediction (replaces nn.Bilinear)
# ---------------------------------------------------------------------------

class BilinearHead(nn.Module):
    """Bilinear(gnn_dim, gnn_dim, 1) implemented as two Dense + dot + bias.

    score = sum(Dense_l(src) * Dense_r(dst), axis=-1) + bias
    Equivalent to nn.Bilinear but JAX-friendly.
    """
    features: int

    @nn.compact
    def __call__(self, src: Array, dst: Array) -> Array:
        src_proj = nn.Dense(self.features, use_bias=False, name="bilinear_l")(src)
        dst_proj = nn.Dense(self.features, use_bias=False, name="bilinear_r")(dst)
        score = (src_proj * dst_proj).sum(axis=-1)
        bias = self.param("bias", nn.initializers.zeros, ())
        return score + bias


# ---------------------------------------------------------------------------
# OnChainGNN: full model
# ---------------------------------------------------------------------------

class OnChainGNN(nn.Module):
    """GNN for on-chain transaction graph analysis.

    Port of the PyTorch OnChainGNN to JAX/Flax using jraph.

    Args:
        node_features: Input node feature dimension (default 8).
        edge_features: Input edge feature dimension (default 2).
        hidden_dim: Hidden dimension (default 128).
        gnn_dim: Output embedding dimension (default 256, MXU-aligned).
        n_layers: Number of message-passing blocks (default 3).
        gat_heads: Number of GAT attention heads (default 4).
        dropout: Dropout rate (default 0.1).
    """
    node_features: int = 8
    edge_features: int = 2
    hidden_dim: int = 128
    gnn_dim: int = 256
    n_layers: int = 3
    gat_heads: int = 4
    dropout: float = 0.1

    def setup(self):
        assert self.gnn_dim % 128 == 0, (
            f"gnn_dim must be MXU-aligned (multiple of 128), got {self.gnn_dim}"
        )

        # Build dimension list: [8, 128, 128, 256] for 3 layers
        dims = [self.node_features] + [self.hidden_dim] * (self.n_layers - 1) + [self.gnn_dim]

        blocks = []
        for i in range(self.n_layers):
            edge_dim = self.edge_features if i == 0 else None
            blocks.append(
                MessagePassingBlock(
                    out_dim=dims[i + 1],
                    heads=self.gat_heads,
                    edge_dim=edge_dim,
                )
            )
        self.mp_blocks = blocks

        self.attn_pool = AttentionPool()

        # Readout MLP: 3*gnn_dim -> 512 -> gnn_dim
        self.readout_dense1 = nn.Dense(512)
        self.readout_ln1 = nn.LayerNorm()
        self.readout_dense2 = nn.Dense(self.gnn_dim)
        self.readout_ln2 = nn.LayerNorm()

        # Loss heads
        self.link_head = BilinearHead(features=self.gnn_dim)
        self.flow_head = nn.Dense(1)

    def _build_graph_idx(self, n_node: Array) -> Array:
        """Build per-node graph assignment vector from n_node array.

        Args:
            n_node: (G,) number of nodes per graph.

        Returns:
            (N_total,) graph index per node.
        """
        n_graphs = n_node.shape[0]
        total_nodes = jnp.sum(n_node)
        # Create graph index: [0,0,...,0, 1,1,...,1, ...]
        graph_idx = jnp.repeat(
            jnp.arange(n_graphs),
            n_node,
            total_repeat_length=total_nodes,
        )
        return graph_idx

    def encode(self, graph: jraph.GraphsTuple, deterministic: bool = True) -> Array:
        """Encode graph to node embeddings.

        Args:
            graph: jraph.GraphsTuple with nodes, edges, senders, receivers.
            deterministic: If True, disable dropout.

        Returns:
            (N_total, gnn_dim) node embeddings.
        """
        x = graph.nodes
        senders = graph.senders
        receivers = graph.receivers
        edge_attr = graph.edges
        total_nodes = jnp.sum(graph.n_node)

        for i, block in enumerate(self.mp_blocks):
            ea = edge_attr if i == 0 else None
            x = block(x, senders, receivers, total_nodes, edge_attr=ea)

        return x

    def __call__(self, graph: jraph.GraphsTuple, deterministic: bool = True) -> Array:
        """Forward pass: encode graph to graph-level embedding.

        Args:
            graph: jraph.GraphsTuple with nodes, edges, senders, receivers.
            deterministic: If True, disable dropout.

        Returns:
            (gnn_dim,) for single graph, (B, gnn_dim) for batched graphs.
        """
        node_emb = self.encode(graph, deterministic=deterministic)

        # Build graph assignment vector
        n_node_arr = graph.n_node
        n_graphs = n_node_arr.shape[0]
        graph_idx = self._build_graph_idx(n_node_arr)

        # Triple pooling: mean || max || attention
        mean_pool = jax.ops.segment_sum(
            node_emb, graph_idx, num_segments=n_graphs
        )
        node_count = jax.ops.segment_sum(
            jnp.ones(node_emb.shape[0]), graph_idx, num_segments=n_graphs
        )
        mean_pool = mean_pool / jnp.maximum(node_count[:, None], 1.0)

        max_pool = jax.ops.segment_max(
            node_emb, graph_idx, num_segments=n_graphs
        )
        # Guard against -inf from segment_max on padding graphs (0 nodes)
        max_pool = jnp.where(
            node_count[:, None] > 0, max_pool, jnp.zeros_like(max_pool)
        )

        attn_pool = self.attn_pool(node_emb, graph_idx, n_graphs)

        pooled = jnp.concatenate([mean_pool, max_pool, attn_pool], axis=-1)

        # Readout MLP
        out = self.readout_dense1(pooled)
        out = self.readout_ln1(out)
        out = nn.elu(out)
        if not deterministic:
            out = nn.Dropout(rate=self.dropout, deterministic=False)(out)
        out = self.readout_dense2(out)
        out = self.readout_ln2(out)

        # Initialize loss head params (needed so they exist in the param tree
        # even when only __call__ is used for init). The results are discarded.
        _dummy = jnp.zeros((1, self.gnn_dim))
        self.link_head(_dummy, _dummy)
        self.flow_head(_dummy)

        # Squeeze for single graph
        if n_graphs == 1:
            return out.squeeze(0)
        return out

    def link_prediction_loss(self, node_emb: Array, pos_edges: Array,
                             neg_edges: Array) -> Array:
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

        pos_score = self.link_head(pos_src, pos_dst)
        neg_score = self.link_head(neg_src, neg_dst)

        # BCE with logits
        pos_loss = optax.sigmoid_binary_cross_entropy(
            pos_score, jnp.ones_like(pos_score)
        ).mean()
        neg_loss = optax.sigmoid_binary_cross_entropy(
            neg_score, jnp.zeros_like(neg_score)
        ).mean()

        return (pos_loss + neg_loss) / 2

    def contrastive_loss(self, emb_t: Array, emb_tp1: Array,
                         temperature: float = 0.07) -> Array:
        """InfoNCE temporal contrastive loss.

        Args:
            emb_t: (B, gnn_dim) graph embeddings at time t.
            emb_tp1: (B, gnn_dim) graph embeddings at time t+1.
            temperature: Softmax temperature.

        Returns:
            Scalar InfoNCE loss.
        """
        emb_t = emb_t / (jnp.linalg.norm(emb_t, axis=-1, keepdims=True) + 1e-8)
        emb_tp1 = emb_tp1 / (jnp.linalg.norm(emb_tp1, axis=-1, keepdims=True) + 1e-8)

        logits = emb_t @ emb_tp1.T / temperature  # (B, B)
        labels = jnp.arange(logits.shape[0])
        return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    def flow_prediction_loss(self, graph_emb: Array, target_flow: Array) -> Array:
        """MSE loss for exchange net inflow/outflow prediction.

        Args:
            graph_emb: (B, gnn_dim) graph embeddings.
            target_flow: (B,) actual net flow values.

        Returns:
            Scalar MSE loss.
        """
        pred = self.flow_head(graph_emb).squeeze(-1)
        return jnp.mean((pred - target_flow) ** 2)
