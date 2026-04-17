"""Build PyG graph snapshots from on-chain parquet data.

Usage:
    python scripts/build_graphs.py --chain eth
    python scripts/build_graphs.py --chain eth --workers 12
    python scripts/build_graphs.py --chain all  # both chains

Output: data/onchain/graphs/{chain}/{YYYY-MM-DD_HH}.pt (PyG Data per hour)
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strate_v.graph_builder import build_hourly_graph, load_exchange_addresses


def _build_one(args_tuple):
    """Worker function for multiprocessing. Returns (filename, success, error)."""
    parquet_path, out_path, exchange_addresses, max_nodes, k_hop = args_tuple
    out_file = out_path / f"{parquet_path.stem}.pt"
    if out_file.exists():
        return (parquet_path.name, True, "skip")
    try:
        df = pd.read_parquet(parquet_path)
        data = build_hourly_graph(df, exchange_addresses, max_nodes=max_nodes, k_hop=k_hop)
        torch.save(data, out_file)
        return (parquet_path.name, True, None)
    except Exception as e:
        return (parquet_path.name, False, str(e))


def build_chain(
    chain: str,
    input_dir: str,
    output_dir: str,
    exchange_addresses: set[str],
    max_nodes: int = 50_000,
    k_hop: int = 2,
    workers: int = 1,
):
    """Build graphs for a single chain.

    Args:
        chain: Chain name (for logging).
        input_dir: Directory with hourly .parquet files.
        output_dir: Output directory for .pt files.
        exchange_addresses: Known exchange addresses.
        max_nodes: Max nodes per snapshot.
        k_hop: Subgraph depth.
        workers: Number of parallel workers (1 = sequential).
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(in_path.glob("*.parquet"))
    if not parquet_files:
        print(f"  [{chain}] No parquet files found in {input_dir}")
        return

    # Filter out already-built graphs
    todo = [pf for pf in parquet_files if not (out_path / f"{pf.stem}.pt").exists()]
    print(f"  [{chain}] {len(todo)} to build ({len(parquet_files) - len(todo)} already exist)")

    if not todo:
        return

    if workers <= 1:
        # Sequential (original behavior)
        for pf in tqdm(todo, desc=f"[{chain}]"):
            out_file = out_path / f"{pf.stem}.pt"
            try:
                df = pd.read_parquet(pf)
                data = build_hourly_graph(df, exchange_addresses, max_nodes=max_nodes, k_hop=k_hop)
                torch.save(data, out_file)
            except Exception as e:
                print(f"  [{chain}] Error on {pf.name}: {e}")
    else:
        # Parallel with ProcessPoolExecutor
        tasks = [(pf, out_path, exchange_addresses, max_nodes, k_hop) for pf in todo]
        errors = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_build_one, t): t[0].name for t in tasks}
            with tqdm(total=len(futures), desc=f"[{chain}] {workers}w") as pbar:
                for future in as_completed(futures):
                    name, success, err = future.result()
                    if not success:
                        errors += 1
                        tqdm.write(f"  [{chain}] Error on {name}: {err}")
                    pbar.update(1)
        if errors:
            print(f"  [{chain}] {errors} errors out of {len(todo)}")


def main():
    parser = argparse.ArgumentParser(description="Build PyG graph snapshots from parquet data")
    parser.add_argument("--chain", default="all", choices=["btc", "eth", "all"])
    parser.add_argument("--input", default=None, help="Input parquet dir (auto: data/onchain/raw/{chain}/)")
    parser.add_argument("--output", default=None, help="Output graph dir (auto: data/onchain/graphs/{chain}/)")
    parser.add_argument("--exchange_addrs", default="data/onchain/exchange_addresses.json")
    parser.add_argument("--max_nodes", type=int, default=50_000)
    parser.add_argument("--k_hop", type=int, default=2)
    parser.add_argument("--workers", type=int, default=12,
                        help="Parallel workers (default: 12, set 1 for sequential)")
    args = parser.parse_args()

    exchange_addresses = load_exchange_addresses(args.exchange_addrs)
    print(f"Loaded {len(exchange_addresses)} exchange addresses")

    chains = ["btc", "eth"] if args.chain == "all" else [args.chain]

    for chain in chains:
        input_dir = args.input or f"data/onchain/raw/{chain}/"
        output_dir = args.output or f"data/onchain/graphs/{chain}/"
        print(f"\n=== Building {chain.upper()} graphs ===")
        build_chain(chain, input_dir, output_dir, exchange_addresses,
                    args.max_nodes, args.k_hop, args.workers)

    print("\nDone.")


if __name__ == "__main__":
    main()
