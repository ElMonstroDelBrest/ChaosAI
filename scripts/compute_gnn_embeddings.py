"""Compute GNN embeddings from trained Strate V model.

Usage:
    python scripts/compute_gnn_embeddings.py \
        --checkpoint checkpoints/strate_v/best.ckpt \
        --graph_dir data/onchain/graphs/btc/ \
        --output data/onchain/embeddings/btc_embeddings.parquet

Output: parquet file with columns: timestamp (str), emb_0..emb_255 (float32)
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strate_v.config import load_config
from src.strate_v.lightning_module import StrateVLightningModule


def main():
    parser = argparse.ArgumentParser(description="Compute GNN embeddings")
    parser.add_argument("--checkpoint", required=True, help="Strate V checkpoint path")
    parser.add_argument("--config", default="configs/strate_v.yaml")
    parser.add_argument("--graph_dir", required=True, help="Directory with .pt graph files")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    module = StrateVLightningModule.load_from_checkpoint(
        args.checkpoint, config=config, map_location=args.device,
    )
    module.eval()
    module.to(args.device)
    gnn = module.gnn

    graph_dir = Path(args.graph_dir)
    pt_files = sorted(graph_dir.glob("*.pt"))
    if not pt_files:
        print(f"No .pt files in {graph_dir}")
        sys.exit(1)

    print(f"Computing embeddings for {len(pt_files)} graphs...")

    rows = []
    with torch.no_grad():
        for pt_file in tqdm(pt_files):
            data = torch.load(pt_file, map_location=args.device, weights_only=False)
            emb = gnn(data)  # (gnn_dim,)
            if emb.dim() > 1:
                emb = emb.squeeze(0)

            timestamp = pt_file.stem  # e.g. "2023-01-15_14"
            row = {"timestamp": timestamp}
            for i, val in enumerate(emb.cpu().numpy()):
                row[f"emb_{i}"] = float(val)
            rows.append(row)

    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Saved {len(df)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
