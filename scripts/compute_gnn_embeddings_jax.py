#!/usr/bin/env python3
"""Compute GNN embeddings from trained JAX Strate V model.

Usage:
    PYTHONPATH=$PWD python scripts/compute_gnn_embeddings_jax.py \
        --checkpoint checkpoints/strate_v_jax/best \
        --graph_dir data/onchain/graphs/eth/ \
        --output data/onchain/embeddings/eth_embeddings.parquet

Output: parquet file with columns: timestamp (str), emb_0..emb_255 (float32)
"""

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.jax_v6.strate_v.gnn_model import OnChainGNN
from src.jax_v6.strate_v.data_loader import load_pyg_as_jraph


def load_params(ckpt_path: str):
    """Load params from orbax checkpoint or pickle."""
    try:
        import orbax.checkpoint as ocp
        checkpointer = ocp.PyTreeCheckpointer()
        return checkpointer.restore(ckpt_path)
    except (ImportError, Exception):
        import pickle
        with open(ckpt_path, "rb") as f:
            return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="Compute GNN embeddings (JAX)")
    parser.add_argument("--checkpoint", required=True, help="JAX checkpoint path")
    parser.add_argument("--graph_dir", required=True, help="Directory with .pt graph files")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--gnn_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--gat_heads", type=int, default=4)
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}", flush=True)

    # Model
    model = OnChainGNN(
        node_features=8, edge_features=2, hidden_dim=args.hidden_dim,
        gnn_dim=args.gnn_dim, n_layers=args.n_layers, gat_heads=args.gat_heads,
        dropout=0.0,
    )

    # Load params
    print(f"Loading checkpoint from {args.checkpoint}...", flush=True)
    params = load_params(args.checkpoint)

    # Graph files
    graph_dir = Path(args.graph_dir)
    pt_files = sorted(graph_dir.glob("*.pt"))
    if not pt_files:
        print(f"No .pt files in {graph_dir}")
        sys.exit(1)
    print(f"Computing embeddings for {len(pt_files)} graphs...", flush=True)

    # Preload all graphs into memory (avoids 11s/graph torch.load per inference)
    import jraph
    print("Preloading all graphs...", flush=True)
    t_pre = time.time()
    graphs = {}
    for i, pt_file in enumerate(pt_files):
        graphs[pt_file] = load_pyg_as_jraph(pt_file)
        if (i + 1) % 500 == 0:
            print(f"  loaded {i+1}/{len(pt_files)} ({time.time()-t_pre:.0f}s)", flush=True)
    print(f"  Preloaded {len(graphs)} graphs in {time.time()-t_pre:.0f}s", flush=True)

    # Pad all graphs to uniform size for JIT
    max_nodes = max(int(g.n_node[0]) for g in graphs.values()) + 1
    max_edges = max(int(g.n_edge[0]) for g in graphs.values()) + 1
    print(f"  Padding to n_node={max_nodes}, n_edge={max_edges}", flush=True)
    for f in graphs:
        graphs[f] = jraph.pad_with_graphs(graphs[f], n_node=max_nodes, n_edge=max_edges, n_graph=2)

    # JIT-compiled forward pass (single compilation since all shapes are uniform)
    @jax.jit
    def forward(params, graph):
        return model.apply(params, graph, deterministic=True)

    # Compute embeddings
    rows = []
    t0 = time.time()
    for i, pt_file in enumerate(pt_files):
        graph = graphs[pt_file]
        emb = forward(params, graph)

        # Padded graph: emb is (2, gnn_dim) — take real graph [0]
        if emb.ndim > 1:
            emb = emb[0]

        emb_np = np.array(emb)
        timestamp = pt_file.stem  # e.g. "2023-01-15_14"
        row = {"timestamp": timestamp}
        for j in range(emb_np.shape[0]):
            row[f"emb_{j}"] = float(emb_np[j])
        rows.append(row)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  {i+1}/{len(pt_files)} ({rate:.1f} graphs/s)", flush=True)

    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    elapsed = time.time() - t0
    print(f"Saved {len(df)} embeddings ({df.shape[1]-1} dims) to {args.output} "
          f"in {elapsed:.1f}s ({len(df)/elapsed:.1f} graphs/s)", flush=True)


if __name__ == "__main__":
    main()
