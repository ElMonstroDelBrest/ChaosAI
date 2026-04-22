"""OOS eval with Multiverse Crossing filter — Option 1 from leakage analysis.

For each timestep, the bifurcation index (pre-computed in the RL buffer)
measures how unstable the JEPA latent is under small perturbations.
We mask portfolio weights to 0 when bifurcation > threshold (model "hesitates"),
following the MVX thesis: trust stable decisions, flat on uncertain ones.

Sweep thresholds and compare Sharpe to the unfiltered baseline.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)


def load_buffer(buffer_dir: str, episode_len: int):
    with open(os.path.join(buffer_dir, "manifest.json")) as f:
        manifest = json.load(f)
    d_model = manifest["d_model"]
    min_len = episode_len + 2
    assets = []
    for info in manifest["assets"]:
        path = info["path"]
        if not os.path.exists(path):
            path = os.path.join(buffer_dir, Path(path).name)
        if not os.path.exists(path):
            continue
        d = np.load(path)
        h, v, r = d["h_last"], d["vol"], d["returns"]
        bif = d.get("bifurcation_index", np.zeros(h.shape[0], dtype=np.float32))
        if h.shape[0] < min_len:
            continue
        assets.append((info["pair"], h, v, r, bif))
    log.info("Buffer: %d assets, d_model=%d", len(assets), d_model)
    return assets, d_model


def sample_episodes_with_bif(assets, episode_len, rng, n_portfolios, k_assets):
    win_len = episode_len + 1
    d_model = assets[0][1].shape[1]
    h = np.zeros((n_portfolios, k_assets, win_len, d_model), dtype=np.float32)
    v = np.zeros((n_portfolios, k_assets, win_len), dtype=np.float32)
    r = np.zeros((n_portfolios, k_assets, win_len), dtype=np.float32)
    b = np.zeros((n_portfolios, k_assets, win_len), dtype=np.float32)
    for i in range(n_portfolios):
        idx = rng.choice(len(assets), size=k_assets, replace=k_assets > len(assets))
        for j, ai in enumerate(idx):
            _, h_a, v_a, r_a, bif_a = assets[ai]
            max_start = max(h_a.shape[0] - episode_len - 1, 1)
            si = rng.integers(max_start)
            h[i, j] = h_a[si:si + win_len]
            v[i, j] = v_a[si:si + win_len]
            r[i, j] = r_a[si:si + win_len]
            b[i, j] = bif_a[si:si + win_len]
    return h, v, r, b


def load_dqn_checkpoint(path: str) -> dict:
    data = np.load(path, allow_pickle=False)
    result = {}
    for key in data.files:
        parts = re.findall(r"key='([^']+)'", key)
        if not parts:
            parts = key.replace("\\", "/").split("/")
        d = result
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = data[key]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos_dir", required=True)
    ap.add_argument("--dqn_ckpt", required=True)
    ap.add_argument("--episode_len", type=int, default=64)
    ap.add_argument("--n_eval", type=int, default=2000)
    ap.add_argument("--k_assets", type=int, default=16)
    ap.add_argument("--n_long", type=int, default=3)
    ap.add_argument("--n_short", type=int, default=3)
    ap.add_argument("--fee_rate", type=float, default=0.0008)
    ap.add_argument("--slippage_factor", type=float, default=0.001)
    ap.add_argument("--cvar_alpha", type=float, default=0.25)
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[1.0, 0.18, 0.17, 0.165, 0.16],
                    help="bifurcation thresholds to sweep (1.0 = no filter)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="results/oos_mvx_sweep.json")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    log.info("JAX: %d devices", jax.device_count())

    from src.jax_v6.strate_iv.env_cross_sectional import (
        compute_scores, score_weighted_allocate,
        reset_portfolio, step_portfolio,
    )
    from src.jax_v6.strate_iv.dqn_agent import QNetwork

    K = args.k_assets
    L = args.episode_len

    assets, d_model = load_buffer(args.oos_dir, L)
    if not assets:
        log.error("No assets")
        return

    q_params = jax.tree.map(jnp.array, load_dqn_checkpoint(args.dqn_ckpt))
    q_net = QNetwork(hidden_dim=512, n_actions=3, n_quantiles=32)
    log.info("DQN loaded")

    # Eval single episode with optional bifurcation filter on portfolio weights.
    # threshold = 1.0 disables filter (weights pass through).
    def eval_single(q_params, h_K, vol_K, ret_K, bif_K, threshold, key):
        obs, state = reset_portfolio(h_K, vol_K, K)

        def body(carry, t):
            obs, state, key = carry
            scores = compute_scores(q_net, q_params, obs, args.cvar_alpha)
            weights = score_weighted_allocate(scores, args.n_long, args.n_short)
            # Mask weights where current-step bifurcation > threshold (uncertain → flat)
            bif_t = bif_K[:, t]              # (K,)
            mask = bif_t <= threshold        # (K,) bool
            weights = jnp.where(mask, weights, 0.0)
            # Renormalize so sum(|w|) = 1 if any survive (avoids dilution)
            denom = jnp.maximum(jnp.sum(jnp.abs(weights)), 1e-8)
            weights = weights / denom
            next_obs, next_state, _, rew, _ = step_portfolio(
                state, weights, h_K, vol_K, ret_K,
                L, args.fee_rate, args.slippage_factor,
            )
            return (next_obs, next_state, key), rew

        _, rewards = jax.lax.scan(body, (obs, state, key), jnp.arange(L))
        return rewards.sum()

    # Vectorize over portfolios; threshold passed as scalar
    batched_eval = jax.jit(jax.vmap(
        eval_single, in_axes=(None, 0, 0, 0, 0, None, 0)
    ))

    # Sample episodes once, evaluate at each threshold
    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    t0 = time.time()
    log.info("Sampling %d episodes (L=%d)...", args.n_eval, L)
    h_np, v_np, r_np, b_np = sample_episodes_with_bif(assets, L, rng, args.n_eval, K)
    h_j, v_j, r_j, b_j = jnp.array(h_np), jnp.array(v_np), jnp.array(r_np), jnp.array(b_np)
    key, *ev_keys = jax.random.split(key, args.n_eval + 1)
    ev_keys = jnp.stack(ev_keys)

    # Sweep
    results = {}
    for thr in args.thresholds:
        ep_rets = np.asarray(batched_eval(q_params, h_j, v_j, r_j, b_j, float(thr), ev_keys))
        sharpe = float(ep_rets.mean()) / (float(ep_rets.std()) + 1e-8)
        # Active fraction = % of (port, asset, t) cells where bif <= threshold
        active_frac = float((b_np <= thr).mean())
        log.info("thr=%.4f  active=%.1f%%  sharpe=%+.4f  mean=%+.4f  std=%.4f",
                 thr, 100 * active_frac, sharpe, ep_rets.mean(), ep_rets.std())
        results[f"thr_{thr:.4f}"] = {
            "threshold": float(thr),
            "active_fraction": active_frac,
            "sharpe": sharpe,
            "mean": float(ep_rets.mean()),
            "std": float(ep_rets.std()),
        }

    log.info("Done in %.1fs", time.time() - t0)
    out = {
        "oos_dir": args.oos_dir,
        "dqn_ckpt": args.dqn_ckpt,
        "n_eval": args.n_eval,
        "k_assets": K,
        "episode_len": L,
        "sweep": results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Written %s", args.output)


if __name__ == "__main__":
    main()
