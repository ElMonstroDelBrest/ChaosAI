"""Lightning module for Strate II (Fin-JEPA) pre-training."""

import math

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .config import StrateIIConfig
from .jepa import FinJEPA


class StrateIILightningModule(pl.LightningModule):
    """Lightning wrapper for Fin-JEPA training.

    Handles:
    - Forward pass and loss computation
    - EMA target encoder update after each step
    - EMA tau cosine annealing over epochs
    - Logging of all loss components
    """

    def __init__(self, config: StrateIIConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        self.jepa = FinJEPA(
            num_codes=config.embedding.num_codes,
            codebook_dim=config.embedding.codebook_dim,
            d_model=config.mamba2.d_model,
            d_state=config.mamba2.d_state,
            n_layers=config.mamba2.n_layers,
            n_heads=config.mamba2.n_heads,
            expand_factor=config.mamba2.expand_factor,
            conv_kernel=config.mamba2.conv_kernel,
            seq_len=config.embedding.seq_len,
            pred_hidden_dim=config.predictor.hidden_dim,
            pred_n_layers=config.predictor.n_layers,
            pred_dropout=config.predictor.dropout,
            pred_z_dim=config.predictor.z_dim,
            cfm_weight=config.predictor.cfm_weight,
            cfm_n_steps=config.predictor.cfm_n_steps,
            cfm_ot=config.predictor.cfm_ot,
            cfm_ot_batch_size=config.predictor.cfm_ot_batch_size,
            encoder_type=config.mamba2.encoder_type,
            mask_ratio=config.masking.mask_ratio,
            block_size_min=config.masking.block_size_min,
            block_size_max=config.masking.block_size_max,
            inv_weight=config.vicreg.inv_weight,
            var_weight=config.vicreg.var_weight,
            cov_weight=config.vicreg.cov_weight,
            var_gamma=config.vicreg.var_gamma,
            tau=config.ema.tau_start,
        )

    def _compute_tau(self) -> float:
        """Cosine anneal EMA tau from tau_start to tau_end."""
        progress = min(self.current_epoch / max(self.config.ema.anneal_epochs, 1), 1.0)
        tau_start = self.config.ema.tau_start
        tau_end = self.config.ema.tau_end
        return tau_end - (tau_end - tau_start) * (1.0 + math.cos(math.pi * progress)) / 2.0

    def on_train_epoch_start(self):
        tau = self._compute_tau()
        self.jepa.set_tau(tau)
        self.log("ema/tau", tau, prog_bar=True)

    def _shared_step(self, batch: dict[str, torch.Tensor], prefix: str):
        out = self.jepa(
            token_indices=batch["token_indices"],
            weekend_mask=batch["weekend_mask"],
            exo_clock=batch.get("exo_clock"),
        )

        self.log_dict(
            {
                f"{prefix}loss/total": out["loss"],
                f"{prefix}loss/invariance": out["invariance"],
                f"{prefix}loss/variance": out["variance"],
                f"{prefix}loss/covariance": out["covariance"],
                f"{prefix}loss/cfm": out["cfm_loss"],
                f"{prefix}mask_ratio": out["mask_ratio"],
            },
            on_step=(prefix == ""),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return out["loss"]

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, prefix="")
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.jepa.update_target_encoder()

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, prefix="val/")

    def configure_optimizers(self):
        # Only optimize context encoder, predictor, and output_proj (not target encoder)
        params = [
            {"params": self.jepa.context_encoder.parameters()},
            {"params": self.jepa.predictor.parameters()},
            {"params": self.jepa.output_proj.parameters()},
        ]
        if self.jepa.flow_predictor is not None:
            params.append({"params": self.jepa.flow_predictor.parameters()})
        optimizer = AdamW(
            params,
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
