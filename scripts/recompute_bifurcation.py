"""Recompute bifurcation_index in an existing RL buffer with arbitrary M.

The original precompute_rl_buffer.py hardcodes M=3 geodesic perturbations.
This script reads each asset's .npz, recomputes bifurcation with custom M,
overwrites bifurcation_index in place. JEPA inference NOT redone.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def bifurcation(h_norm: np.ndarray, M: int, sigma: float, rng) -> float:
    """Geodesic perturbation entropy on unit hypersphere."""
    d = h_norm.shape[0]
    perturbed = np.empty((M, d), dtype=np.float32)
    for k in range(M):
        n = rng.standard_normal(d).astype(np.float32) * sigma
        n = n - np.dot(n, h_norm) * h_norm  # tangent plane projection
        p = h_norm + n
        p /= np.linalg.norm(p) + 1e-8
        perturbed[k] = p
    cov = perturbed @ perturbed.T
    ev = np.abs(np.linalg.eigvalsh(cov))
    ev = ev / (ev.sum() + 1e-10)
    return float(-np.sum(ev * np.log(ev + 1e-10)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer_dir", required=True)
    ap.add_argument("--M", type=int, default=30)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--inplace", action="store_true",
                    help="Overwrite bifurcation_index in original .npz")
    ap.add_argument("--dry_run", action="store_true",
                    help="Just print stats without writing")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    paths = sorted(Path(args.buffer_dir).glob("*.npz"))
    print(f"Found {len(paths)} assets in {args.buffer_dir}")
    print(f"Recomputing bifurcation with M={args.M}, sigma={args.sigma}")

    all_old, all_new = [], []
    for i, p in enumerate(paths):
        d = dict(np.load(p))
        h = d["h_last"]                       # (T, d_model)
        old_bif = d["bifurcation_index"]
        norms = np.linalg.norm(h, axis=1, keepdims=True) + 1e-8
        h_norm = h / norms
        new_bif = np.array([bifurcation(h_norm[t], args.M, args.sigma, rng)
                            for t in range(h.shape[0])], dtype=np.float32)
        all_old.append(old_bif); all_new.append(new_bif)
        if args.inplace and not args.dry_run:
            d["bifurcation_index"] = new_bif
            np.savez_compressed(p, **d)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(paths)}  last: old={old_bif.mean():.4f} -> new={new_bif.mean():.4f}")

    O = np.concatenate(all_old); N = np.concatenate(all_new)
    print(f"\n=== Bifurcation stats ===")
    print(f"  Old (M=3):  mean={O.mean():.4f}  std={O.std():.4f}  "
          f"p10={np.percentile(O,10):.4f}  p50={np.percentile(O,50):.4f}  "
          f"p90={np.percentile(O,90):.4f}  p99={np.percentile(O,99):.4f}")
    print(f"  New (M={args.M}): mean={N.mean():.4f}  std={N.std():.4f}  "
          f"p10={np.percentile(N,10):.4f}  p50={np.percentile(N,50):.4f}  "
          f"p90={np.percentile(N,90):.4f}  p99={np.percentile(N,99):.4f}")
    print(f"  Range Old: [{O.min():.4f}, {O.max():.4f}]")
    print(f"  Range New: [{N.min():.4f}, {N.max():.4f}]")

    if args.inplace and not args.dry_run:
        print(f"\n  Overwrote bifurcation_index in {len(paths)} files")


if __name__ == "__main__":
    main()
