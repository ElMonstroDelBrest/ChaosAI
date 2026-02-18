"""PyTorch Lightning training module for Strate I."""

import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .config import StrateIConfig
from .losses import VQVAELoss
from .tokenizer import TopologicalTokenizer


class StrateILightningModule(pl.LightningModule):
    def __init__(self, config: StrateIConfig):
        super().__init__()
        self.config = config
        self.tokenizer = TopologicalTokenizer(config)
        self.loss_fn = VQVAELoss(
            huber_delta=config.loss.huber_delta,
            sdtw_alpha=config.loss.sdtw_alpha,
            sdtw_gamma=config.loss.sdtw_gamma,
        )

    def _shared_step(self, batch, prefix: str):
        out = self.tokenizer(batch)
        loss_dict = self.loss_fn(batch, out["x_hat"], out["commitment_loss"])

        self.log_dict(
            {
                f"{prefix}loss/total": loss_dict["total"],
                f"{prefix}loss/huber": loss_dict["huber"],
                f"{prefix}loss/sdtw": loss_dict["sdtw"],
                f"{prefix}loss/commitment": loss_dict["commitment"],
                f"{prefix}codebook/perplexity": out["perplexity"],
                f"{prefix}codebook/utilization": out["utilization"],
            },
            on_step=(prefix == ""),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss_dict["total"]

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, prefix="")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, prefix="val/")

    def on_train_epoch_end(self):
        self.tokenizer.vqvae.codebook.reset_usage()
        # FSQ: refresh the `embeddings` buffer (proj_out(grid)) after each epoch
        # so downstream code (FinJEPA.load_codebook, decode_from_indices) always
        # has up-to-date 64-dim representations of each discrete code.
        if self.tokenizer.vqvae._use_fsq:
            self.tokenizer.vqvae.codebook.sync_embeddings()

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )
        warmup_epochs = self.config.training.warmup_epochs
        max_epochs = self.config.training.max_epochs

        warmup = LinearLR(optimizer, start_factor=1e-5, end_factor=1.0, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs)
        scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }
