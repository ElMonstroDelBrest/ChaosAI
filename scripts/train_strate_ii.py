"""Train Strate II (Fin-JEPA) model.

Usage:
    # Standard training:
    python scripts/train_strate_ii.py --config configs/strate_ii.yaml

    # Synthetic data for development:
    python scripts/train_strate_ii.py --config configs/strate_ii.yaml --synthetic

    # H100 Production-Scale with torch.compile:
    python scripts/train_strate_ii.py --config configs/strate_ii.yaml --compile

    # FP8 precision (H100 only, requires nvidia-transformer-engine):
    python scripts/train_strate_ii.py --config configs/strate_ii.yaml --compile --fp8

    # Override token_dir (e.g. local NVMe SSD):
    python scripts/train_strate_ii.py --config configs/strate_ii.yaml --token_dir /mnt/disks/local-ssd/data/tokens_v5/

    # Auto-resume from last checkpoint:
    python scripts/train_strate_ii.py --config configs/strate_ii.yaml
    # (automatically detects and resumes from checkpoints/strate_ii/last.ckpt)

    # With Strate I codebook:
    python scripts/train_strate_ii.py --config configs/strate_ii.yaml \
        --codebook_checkpoint checkpoints/strate_i_best.ckpt
"""

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.strate_ii.config import load_config
from src.strate_ii.data.datamodule import StrateIIDataModule
from src.strate_ii.lightning_module import StrateIILightningModule


CHECKPOINT_DIR = "checkpoints/strate_ii/"


def main():
    parser = argparse.ArgumentParser(description="Train Strate II (Fin-JEPA)")
    parser.add_argument("--config", type=str, default="configs/strate_ii.yaml")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--num_synthetic", type=int, default=512)
    parser.add_argument(
        "--codebook_checkpoint", type=str, default=None,
        help="Path to Strate I checkpoint (to load codebook weights)",
    )
    parser.add_argument(
        "--strate_i_config", type=str, default="configs/strate_i_binance.yaml",
        help="Path to Strate I config (for codebook loading)",
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="Use torch.compile for faster training (requires PyTorch 2.1+)",
    )
    parser.add_argument(
        "--accumulate_grad_batches", type=int, default=1,
        help="Gradient accumulation steps (effective batch = batch_size × this)",
    )
    parser.add_argument(
        "--no_resume", action="store_true",
        help="Force fresh start, ignore existing checkpoints",
    )
    parser.add_argument(
        "--fp8", action="store_true",
        help="Use FP8 precision via TransformerEngine (H100 only, ~1.5× faster matmuls)",
    )
    parser.add_argument(
        "--token_dir", type=str, default=None,
        help="Override token_dir from config (e.g. local NVMe SSD path)",
    )
    parser.add_argument(
        "--spot_watcher", action="store_true",
        help="Enable GCP Spot preemption watcher (polls metadata at 1Hz)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Enable TF32 for H100/A100 (free 3× speedup on matmuls)
    torch.set_float32_matmul_precision("high")

    # Override token_dir if specified (e.g. local NVMe SSD)
    token_dir = args.token_dir or config.data.token_dir

    # FP8 precision setup
    precision = config.training.precision
    plugins = []
    if args.fp8:
        try:
            from lightning.pytorch.plugins import TransformerEnginePrecision
            plugins.append(TransformerEnginePrecision(weights_dtype=torch.bfloat16))
            precision = None  # TransformerEngine plugin handles precision
            print("FP8: Using TransformerEngine precision plugin")
        except ImportError:
            print("WARNING: --fp8 requested but nvidia-transformer-engine not installed.")
            print("  Install with: pip install nvidia-transformer-engine")
            print("  Falling back to bf16-mixed.")

    # Data
    datamodule = StrateIIDataModule(
        token_dir=token_dir,
        seq_len=config.embedding.seq_len,
        num_codes=config.embedding.num_codes,
        batch_size=config.training.batch_size,
        val_split=config.data.val_split,
        num_workers=config.data.num_workers,
        prefetch_factor=config.data.prefetch_factor,
        synthetic=args.synthetic,
        num_synthetic=args.num_synthetic,
    )

    # Model
    model = StrateIILightningModule(config)

    # Load codebook from Strate I if provided
    if args.codebook_checkpoint:
        from src.strate_i.config import load_config as load_strate_i_config
        from src.strate_i.lightning_module import StrateILightningModule

        strate_i_config = load_strate_i_config(args.strate_i_config)
        strate_i = StrateILightningModule.load_from_checkpoint(
            args.codebook_checkpoint, config=strate_i_config
        )
        codebook_weights = strate_i.tokenizer.vqvae.codebook.embeddings.clone()
        model.jepa.load_codebook(codebook_weights)
        print(f"Loaded codebook from {args.codebook_checkpoint}")

    # torch.compile for fused kernels (massive speedup on H100)
    if args.compile:
        print("Compiling model with torch.compile (mode=max-autotune)...")
        model.jepa = torch.compile(model.jepa, mode="max-autotune")

    # Print model stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params / 1e6:.1f}M")
    print(f"Trainable params: {trainable_params / 1e6:.1f}M")

    # --- Callbacks ---

    # Best model checkpoint (by validation loss)
    best_cb = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="strate_ii-{epoch:03d}-{val/loss/total:.4f}",
        monitor="val/loss/total",
        mode="min",
        save_top_k=3,
    )

    # Last checkpoint (every 20 min for Spot VM resilience)
    last_cb = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="last",
        every_n_train_steps=500,  # ~every 15-20 min depending on throughput
        save_top_k=1,
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [best_cb, last_cb, lr_monitor]

    # Spot preemption watcher (GCP metadata polling)
    if args.spot_watcher:
        from infra.spot_watcher import SpotPreemptionCallback
        callbacks.append(SpotPreemptionCallback())
        print("Spot preemption watcher enabled")

    # Logger
    logger = TensorBoardLogger("tb_logs", name="strate_ii")

    # --- Auto-resume ---
    ckpt_path = None
    if not args.no_resume:
        last_ckpt = Path(CHECKPOINT_DIR) / "last.ckpt"
        if last_ckpt.exists():
            ckpt_path = str(last_ckpt)
            print(f"Auto-resuming from {ckpt_path}")
        else:
            # Also check for last-v1.ckpt pattern
            last_v = Path(CHECKPOINT_DIR) / "last-v1.ckpt"
            if last_v.exists():
                ckpt_path = str(last_v)
                print(f"Auto-resuming from {ckpt_path}")

    # Trainer
    trainer_kwargs = dict(
        max_epochs=config.training.max_epochs,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=10,
    )
    if precision is not None:
        trainer_kwargs["precision"] = precision
    if plugins:
        trainer_kwargs["plugins"] = plugins
    trainer = pl.Trainer(**trainer_kwargs)

    trainer.fit(model, datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
