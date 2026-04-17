"""Lightning module for Strate V (GNN On-Chain) self-supervised pre-training.

Three self-supervised objectives:
  1. Link prediction (BCE) — intra-snapshot structure
  2. Temporal contrastive (InfoNCE) — inter-snapshot dynamics
  3. Exchange flow prediction (MSE) — next-hour net inflow/outflow

Loss: L_link + 0.5 * L_contrastive + 0.1 * L_flow
"""

import math

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.data import Data, Batch
from torch_geometric.utils import negative_sampling

from .config import StrateVConfig
from .gnn_model import OnChainGNN


class StrateVLightningModule(pl.LightningModule):
    """Lightning wrapper for GNN On-Chain self-supervised training."""

    def __init__(self, config: StrateVConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        self.gnn = OnChainGNN(
            node_features=config.graph.node_features,
            edge_features=config.graph.edge_features,
            hidden_dim=config.model.hidden_dim,
            gnn_dim=config.model.gnn_dim,
            n_layers=config.model.n_layers,
            gat_heads=config.model.gat_heads,
            dropout=config.model.dropout,
        )

    def _compute_losses(self, batch: Batch) -> dict[str, torch.Tensor]:
        """Compute all three self-supervised losses.

        Args:
            batch: PyG Batch of graph pairs (t, t+1) stored as consecutive graphs.

        Returns:
            Dict with individual losses and total.
        """
        # Encode all graphs in batch
        node_emb = self.gnn.encode(batch)  # (N_total, gnn_dim)
        graph_emb = self.gnn(batch)  # (B, gnn_dim)

        # 1. Link prediction loss (intra-snapshot)
        pos_edges = batch.edge_index.T  # (E, 2)
        neg_edges = negative_sampling(
            batch.edge_index,
            num_nodes=batch.num_nodes,
            num_neg_samples=pos_edges.shape[0],
        ).T  # (E_neg, 2)
        link_loss = self.gnn.link_prediction_loss(node_emb, pos_edges, neg_edges)

        # 2. Temporal contrastive loss (inter-snapshot)
        # Batch contains pairs: even indices = t, odd = t+1
        B = graph_emb.shape[0]
        contrastive_loss = torch.tensor(0.0, device=graph_emb.device)
        if B >= 2:
            emb_t = graph_emb[0::2]    # even: time t
            emb_tp1 = graph_emb[1::2]  # odd: time t+1
            min_len = min(emb_t.shape[0], emb_tp1.shape[0])
            if min_len > 0:
                contrastive_loss = self.gnn.contrastive_loss(
                    emb_t[:min_len], emb_tp1[:min_len],
                    temperature=self.config.training.temperature,
                )

        # 3. Exchange flow prediction loss
        if hasattr(batch, "exchange_net_flow"):
            target_flow = batch.exchange_net_flow  # (B,)
            flow_loss = self.gnn.flow_prediction_loss(graph_emb, target_flow)
        else:
            flow_loss = torch.tensor(0.0, device=graph_emb.device)

        total = (
            self.config.training.link_weight * link_loss
            + self.config.training.contrastive_weight * contrastive_loss
            + self.config.training.flow_weight * flow_loss
        )

        return {
            "loss": total,
            "link_loss": link_loss,
            "contrastive_loss": contrastive_loss,
            "flow_loss": flow_loss,
        }

    def training_step(self, batch, batch_idx):
        losses = self._compute_losses(batch)
        self.log_dict(
            {f"train/{k}": v for k, v in losses.items()},
            on_step=True, on_epoch=True, prog_bar=True, batch_size=1,
        )
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        losses = self._compute_losses(batch)
        self.log_dict(
            {f"val/{k}": v for k, v in losses.items()},
            on_epoch=True, prog_bar=True, batch_size=1,
        )
        return losses["loss"]

    def configure_optimizers(self):
        optimizer = AdamW(
            self.gnn.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )

        warmup_epochs = self.config.training.warmup_epochs
        max_epochs = self.config.training.max_epochs

        warmup = LinearLR(
            optimizer, start_factor=1e-5, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs)
        scheduler = SequentialLR(
            optimizer, [warmup, cosine], milestones=[warmup_epochs]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }
