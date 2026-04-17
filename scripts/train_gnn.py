"""Train Strate V GNN On-Chain model (self-supervised, GPU).

Usage:
    python scripts/train_gnn.py --config configs/strate_v.yaml
    python scripts/train_gnn.py --config configs/strate_v.yaml --gpus 1 --precision 16-mixed

Training time: ~2-4h on a single GPU for full dataset.
"""

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strate_v.config import load_config
from src.strate_v.lightning_module import StrateVLightningModule


class GraphPairDataset(torch.utils.data.Dataset):
    """Dataset of consecutive graph snapshot pairs for temporal contrastive learning.

    Loads .pt files (PyG Data) and returns consecutive pairs (t, t+1).
    """

    def __init__(self, graph_dirs: list[str], include_singles: bool = True):
        """
        Args:
            graph_dirs: List of directories containing .pt graph files.
            include_singles: If True, also yield individual graphs (not just pairs).
        """
        self.pairs = []
        self.singles = []

        for gdir in graph_dirs:
            pt_files = sorted(Path(gdir).glob("*.pt"))
            for i in range(len(pt_files) - 1):
                self.pairs.append((pt_files[i], pt_files[i + 1]))
            if include_singles:
                for f in pt_files:
                    self.singles.append(f)

    def __len__(self):
        return len(self.pairs) + len(self.singles)

    def __getitem__(self, idx):
        if idx < len(self.pairs):
            f_t, f_tp1 = self.pairs[idx]
            data_t = torch.load(f_t, map_location="cpu", weights_only=False)
            data_tp1 = torch.load(f_tp1, map_location="cpu", weights_only=False)
            return [data_t, data_tp1]
        else:
            f = self.singles[idx - len(self.pairs)]
            data = torch.load(f, map_location="cpu", weights_only=False)
            return [data]


def collate_fn(batch_list):
    """Collate variable-size graph pairs into a single PyG Batch."""
    from torch_geometric.data import Batch
    all_graphs = []
    for item in batch_list:
        all_graphs.extend(item)
    return Batch.from_data_list(all_graphs)


def main():
    parser = argparse.ArgumentParser(description="Train Strate V GNN On-Chain")
    parser.add_argument("--config", default="configs/strate_v.yaml")
    parser.add_argument("--graph_dirs", nargs="+", default=["data/onchain/graphs/btc/", "data/onchain/graphs/eth/"])
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--precision", default=None, help="Override precision from config")
    parser.add_argument("--ckpt_dir", default="checkpoints/strate_v/")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    config = load_config(args.config)
    precision = args.precision or config.training.precision

    # Dataset
    dataset = GraphPairDataset(args.graph_dirs)
    if len(dataset) == 0:
        print("ERROR: No graph files found. Run scripts/build_graphs.py first.")
        sys.exit(1)

    # Train/val split (80/20 by index)
    n_val = max(1, int(len(dataset) * 0.2))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # Model
    module = StrateVLightningModule(config)
    print(f"GNN params: {sum(p.numel() for p in module.gnn.parameters()):,}")

    # Callbacks
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="strate_v-{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.gpus if torch.cuda.is_available() else 1,
        precision=precision,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )

    trainer.fit(module, train_loader, val_loader, ckpt_path=args.resume)
    print(f"\nBest checkpoint: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
