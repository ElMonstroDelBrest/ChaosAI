"""Linear probe on fresh JEPA latents — test A from leakage analysis.

Goal: Determine whether the v6.4 JEPA encoder transfers to truly held-out
data, independently of the DQN policy.

Method:
  1. Load all fresh assets from data/rl_buffer_fresh/ (post-pretrain dates)
  2. Concat (h_last, returns) → (N, d_model), (N,)
  3. Ridge regression with random + temporal train/val split
  4. Weighted R² + directional accuracy per split
  5. Compare against:
       (a) lag-1 return baseline (returns[t-1] → returns[t])
       (b) zero baseline

If JEPA R² > 0  → encoder generalises, DQN broke it
If JEPA R² ≤ 0 → encoder doesn't generalise either; leakage in encoder weights
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge


def weighted_r2(pred: np.ndarray, true: np.ndarray, w: np.ndarray | None = None) -> float:
    """Zero-mean weighted R^2 (Jane Street convention). If w is None, uniform."""
    if w is None:
        w = np.ones_like(true)
    mask = np.isfinite(true) & np.isfinite(pred)
    if mask.sum() == 0:
        return float("nan")
    w, true, pred = w[mask], true[mask], pred[mask]
    num = (w * (true - pred) ** 2).sum()
    den = (w * true ** 2).sum() + 1e-8
    return float(1.0 - num / den)


def dir_acc(pred: np.ndarray, true: np.ndarray) -> float:
    mask = np.isfinite(true) & np.isfinite(pred) & (np.abs(true) > 1e-10)
    if mask.sum() == 0:
        return float("nan")
    return float((np.sign(pred[mask]) == np.sign(true[mask])).mean())


def load_buffer(buffer_dir: str):
    """Load all .npz from a buffer dir → concat (h_last, returns, asset_id, t_idx)."""
    paths = sorted(Path(buffer_dir).glob("*.npz"))
    H, R, A, T = [], [], [], []
    for ai, p in enumerate(paths):
        d = np.load(p)
        h, r = d["h_last"], d["returns"]
        if h.shape[0] < 5:
            continue
        H.append(h.astype(np.float32))
        R.append(r.astype(np.float32))
        A.append(np.full(h.shape[0], ai, dtype=np.int32))
        T.append(np.arange(h.shape[0], dtype=np.int32))
    H = np.concatenate(H)
    R = np.concatenate(R)
    A = np.concatenate(A)
    T = np.concatenate(T)
    print(f"  Loaded {len(paths)} assets, {H.shape[0]} sequences, d_model={H.shape[1]}")
    return H, R, A, T


def lag1_returns(R: np.ndarray, A: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Build per-asset lag-1 returns vector aligned to current."""
    lag = np.zeros_like(R)
    for asset_id in np.unique(A):
        mask = A == asset_id
        order = np.argsort(T[mask])
        idx = np.where(mask)[0][order]
        if len(idx) > 1:
            lag[idx[1:]] = R[idx[:-1]]
    return lag


def fit_eval(name: str, X_tr, y_tr, X_va, y_va, alphas=(0.1, 1.0, 10.0, 100.0)):
    best = {"alpha": None, "r2": -1e9, "dir": float("nan")}
    for a in alphas:
        reg = Ridge(alpha=a)
        reg.fit(X_tr, y_tr)
        p = reg.predict(X_va)
        r2 = weighted_r2(p, y_va)
        if r2 > best["r2"]:
            best = {"alpha": a, "r2": r2, "dir": dir_acc(p, y_va)}
    print(f"  {name:30s}  R^2={best['r2']:+.5f}  dir-acc={best['dir']:.4f}  "
          f"(best alpha={best['alpha']})")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--output", default="results/probe_fresh.json")
    args = ap.parse_args()

    print(f"\n=== Linear probe on {args.buffer_dir} ===")
    H, R, A, T = load_buffer(args.buffer_dir)

    # Standardise H per-feature (Ridge needs it)
    mu, sd = H.mean(0), H.std(0) + 1e-6
    H = (H - mu) / sd

    # Random split (asset-stratified — keep balance per asset)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(H.shape[0])
    n_va = int(args.val_frac * H.shape[0])
    va_idx, tr_idx = perm[:n_va], perm[n_va:]
    H_tr, H_va = H[tr_idx], H[va_idx]
    R_tr, R_va = R[tr_idx], R[va_idx]
    A_tr, A_va = A[tr_idx], A[va_idx]
    T_tr, T_va = T[tr_idx], T[va_idx]
    print(f"\n  Random split: train={H_tr.shape[0]}  val={H_va.shape[0]}")

    print("\n  --- Random split (any time) ---")
    jepa_rand = fit_eval("JEPA latent (Ridge)", H_tr, R_tr, H_va, R_va)

    # Lag-1 baseline
    L_all = lag1_returns(R, A, T)
    L_tr, L_va = L_all[tr_idx][:, None], L_all[va_idx][:, None]
    lag_rand = fit_eval("Lag-1 return baseline", L_tr, R_tr, L_va, R_va)

    # Zero baseline
    z_r2 = weighted_r2(np.zeros_like(R_va), R_va)
    print(f"  {'Zero baseline':30s}  R^2={z_r2:+.5f}")

    # Temporal split: per asset, last 20% timesteps = val
    is_va = np.zeros(H.shape[0], dtype=bool)
    for asset_id in np.unique(A):
        mask = A == asset_id
        idx = np.where(mask)[0]
        order = np.argsort(T[mask])
        idx = idx[order]
        cutoff = int(len(idx) * (1 - args.val_frac))
        is_va[idx[cutoff:]] = True
    H_tr2, H_va2 = H[~is_va], H[is_va]
    R_tr2, R_va2 = R[~is_va], R[is_va]
    L_tr2, L_va2 = L_all[~is_va, None], L_all[is_va, None]
    print(f"\n  Temporal split (last {int(args.val_frac*100)}% per asset): "
          f"train={H_tr2.shape[0]}  val={H_va2.shape[0]}")
    jepa_temp = fit_eval("JEPA latent (Ridge)", H_tr2, R_tr2, H_va2, R_va2)
    lag_temp = fit_eval("Lag-1 return baseline", L_tr2, R_tr2, L_va2, R_va2)
    z_temp = weighted_r2(np.zeros_like(R_va2), R_va2)
    print(f"  {'Zero baseline':30s}  R^2={z_temp:+.5f}")

    out = {
        "buffer_dir": args.buffer_dir,
        "n_seq": int(H.shape[0]),
        "d_model": int(H.shape[1]),
        "random_split": {
            "jepa_r2": jepa_rand["r2"], "jepa_dir": jepa_rand["dir"],
            "lag1_r2": lag_rand["r2"], "lag1_dir": lag_rand["dir"],
            "zero_r2": z_r2,
        },
        "temporal_split": {
            "jepa_r2": jepa_temp["r2"], "jepa_dir": jepa_temp["dir"],
            "lag1_r2": lag_temp["r2"], "lag1_dir": lag_temp["dir"],
            "zero_r2": z_temp,
        },
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWritten {args.output}")


if __name__ == "__main__":
    main()
