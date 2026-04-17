"""OOS temporal evaluation: load trained DQN, run on OOS buffer (last 20% of sequences).

Uses the same eval loop as train_cross_sectional.py (greedy, no noise).

Usage:
    PYTHONPATH=. python3 scripts/eval_oos_temporal.py \
        --oos_dir data/rl_buffer_oos/ \
        --dqn_ckpt checkpoints/cs_oos_train/best_cs_dqn.npz \
        --episode_len 32 \
        --n_eval 2000 \
        --k_assets 16 \
        --output results/oos_temporal_eval.json
"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Asset class detection ────────────────────────────────────────────────────

def detect_class(pair_name: str) -> str:
    p = pair_name.lower()
    if "futures__" in p or "spot__" in p or p.endswith("usdt") or p.endswith("busd"):
        return "crypto"
    if "ohlcv_forex__" in p:
        return "forex"
    if "ohlcv_commodities__" in p:
        return "commodities"
    if "ohlcv_sp500__" in p or "ohlcv_stocks_daily__" in p or "ohlcv_stocks_1h__" in p:
        return "stocks"
    if "yfinance__" in p:
        return "stocks_etf"
    return "other"


# ── Buffer loading ───────────────────────────────────────────────────────────

def load_buffer(buffer_dir: str, episode_len: int):
    """Load all assets from buffer (no train/val split)."""
    with open(os.path.join(buffer_dir, "manifest.json")) as f:
        manifest = json.load(f)
    d_model = manifest["d_model"]
    min_len = episode_len + 2
    assets = []
    for info in manifest["assets"]:
        pair, path = info["pair"], info["path"]
        if not os.path.exists(path):
            path = os.path.join(buffer_dir, Path(path).name)
        if not os.path.exists(path):
            continue
        data = np.load(path)
        h, v, r = data["h_last"], data["vol"], data["returns"]
        bif = data.get("bifurcation_index", np.zeros(h.shape[0], dtype=np.float32))
        if h.shape[0] < min_len:
            continue
        assets.append((pair, h, v, r, bif, detect_class(pair)))
    log.info("OOS buffer: %d assets, d_model=%d", len(assets), d_model)
    return assets, d_model


# ── Episode sampling ─────────────────────────────────────────────────────────

def sample_episodes(assets, episode_len, rng, n_portfolios, k_assets,
                    class_filter=None):
    pool = [a for a in assets if class_filter is None or a[5] == class_filter]
    if len(pool) < k_assets:
        pool = assets
    win_len = episode_len + 1
    d_model = assets[0][1].shape[1]
    h = np.zeros((n_portfolios, k_assets, win_len, d_model), dtype=np.float32)
    v = np.zeros((n_portfolios, k_assets, win_len), dtype=np.float32)
    r = np.zeros((n_portfolios, k_assets, win_len), dtype=np.float32)
    for i in range(n_portfolios):
        idx = rng.choice(len(pool), size=k_assets, replace=k_assets > len(pool))
        for j, ai in enumerate(idx):
            _, h_a, v_a, r_a, _, _ = pool[ai]
            max_start = max(h_a.shape[0] - episode_len - 1, 1)
            si = rng.integers(max_start)
            h[i, j] = h_a[si:si + win_len]
            v[i, j] = v_a[si:si + win_len]
            r[i, j] = r_a[si:si + win_len]
    return h, v, r


# ── DQN checkpoint loading ───────────────────────────────────────────────────

def load_dqn_checkpoint(path: str) -> dict:
    """Load saved QNetwork params from .npz → nested dict.

    Keys look like "(DictKey(key='Dense_0'), DictKey(key='kernel'))".
    Parses back to {'Dense_0': {'kernel': ..., 'bias': ...}, ...}.
    (Same logic as train_cs_finetune.py:load_dqn_checkpoint.)
    """
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OOS temporal evaluation of trained DQN")
    parser.add_argument("--oos_dir", type=str, required=True)
    parser.add_argument("--dqn_ckpt", type=str, required=True)
    parser.add_argument("--episode_len", type=int, default=32)
    parser.add_argument("--n_eval", type=int, default=2000)
    parser.add_argument("--k_assets", type=int, default=16)
    parser.add_argument("--n_long", type=int, default=3)
    parser.add_argument("--n_short", type=int, default=3)
    parser.add_argument("--fee_rate", type=float, default=0.0008)
    parser.add_argument("--slippage_factor", type=float, default=0.001)
    parser.add_argument("--cvar_alpha", type=float, default=0.25)
    parser.add_argument("--soft_alloc", action="store_true", default=True)
    parser.add_argument("--output", type=str, default="results/oos_temporal_eval.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()

    import jax
    import jax.numpy as jnp
    log.info("JAX: %d devices (%s)", jax.device_count(), jax.devices()[0].device_kind)

    from src.jax_v6.strate_iv.env_cross_sectional import (
        obs_dim as get_obs_dim,
        compute_scores,
        score_weighted_allocate,
        reset_portfolio,
        step_portfolio,
    )
    from src.jax_v6.strate_iv.dqn_agent import QNetwork

    K = args.k_assets
    L = args.episode_len
    win_len = L + 1
    n_long = args.n_long
    n_short = args.n_short

    # Load OOS buffer
    assets, d_model = load_buffer(args.oos_dir, L)
    if not assets:
        log.error("No assets in OOS buffer with min_len=%d. Try smaller --episode_len.", L + 2)
        return

    o_dim = get_obs_dim(d_model)
    log.info("obs_dim=%d, d_model=%d, K=%d, episode_len=%d", o_dim, d_model, K, L)

    # Load DQN params
    q_params_np = load_dqn_checkpoint(args.dqn_ckpt)
    q_params = jax.tree.map(jnp.array, q_params_np)
    log.info("DQN loaded from %s", args.dqn_ckpt)

    # Build QNetwork
    q_net = QNetwork(hidden_dim=512, n_actions=3, n_quantiles=32)

    # Greedy eval episode (same as eval_single in train_cross_sectional.py)
    def eval_single(q_params, h_K, vol_K, ret_K, key):
        obs, state = reset_portfolio(h_K, vol_K, K)

        def body(carry, _):
            obs, state, key = carry
            key, _ = jax.random.split(key)
            scores = compute_scores(q_net, q_params, obs, args.cvar_alpha)
            weights = score_weighted_allocate(scores, n_long, n_short)
            next_obs, next_state, _, portfolio_rew, done = step_portfolio(
                state, weights, h_K, vol_K, ret_K,
                L, args.fee_rate, args.slippage_factor,
            )
            return (next_obs, next_state, key), portfolio_rew

        _, rewards = jax.lax.scan(body, (obs, state, key), None, length=L)
        return rewards.sum()

    batched_eval = jax.jit(jax.vmap(eval_single, in_axes=(None, 0, 0, 0, 0)))

    # JIT warmup
    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    log.info("JIT compiling...")
    dh = jnp.zeros((4, K, win_len, d_model))
    dv = jnp.zeros((4, K, win_len))
    dr = jnp.zeros((4, K, win_len))
    key, *wk = jax.random.split(key, 5)
    _ = batched_eval(q_params, dh, dv, dr, jnp.stack(wk))
    log.info("JIT done")

    # ── Global OOS eval ──────────────────────────────────────────────────────
    log.info("Global OOS eval (%d episodes)...", args.n_eval)
    h_np, v_np, r_np = sample_episodes(assets, L, rng, args.n_eval, K)
    key, *ev_keys = jax.random.split(key, args.n_eval + 1)
    ep_rets = np.asarray(batched_eval(
        q_params, jnp.array(h_np), jnp.array(v_np),
        jnp.array(r_np), jnp.stack(ev_keys),
    ))
    global_sharpe = float(ep_rets.mean()) / (float(ep_rets.std()) + 1e-8)
    log.info("Global OOS  sharpe=%.4f  mean=%.4f  std=%.4f  n=%d",
             global_sharpe, ep_rets.mean(), ep_rets.std(), args.n_eval)

    # ── Per-class OOS eval ───────────────────────────────────────────────────
    classes = sorted(set(a[5] for a in assets))
    class_counts = {c: sum(1 for a in assets if a[5] == c) for c in classes}
    log.info("Classes: %s", class_counts)

    class_results = {}
    for cls in classes:
        pool = [a for a in assets if a[5] == cls]
        n_cls = min(args.n_eval, 500)
        if len(pool) < K:
            log.warning("  %s: only %d assets < K=%d, using full pool", cls, len(pool), K)
        h_c, v_c, r_c = sample_episodes(assets, L, rng, n_cls, K, class_filter=cls)
        key, *ck = jax.random.split(key, n_cls + 1)
        rets_c = np.asarray(batched_eval(
            q_params, jnp.array(h_c), jnp.array(v_c),
            jnp.array(r_c), jnp.stack(ck),
        ))
        s_c = float(rets_c.mean()) / (float(rets_c.std()) + 1e-8)
        class_results[cls] = {
            "sharpe": s_c,
            "mean": float(rets_c.mean()),
            "std": float(rets_c.std()),
            "n_assets": len(pool),
            "n_eval": n_cls,
        }
        log.info("  %-15s sharpe=%.4f  mean=%.4f  std=%.4f  (%d assets)",
                 cls, s_c, rets_c.mean(), rets_c.std(), len(pool))

    # ── Save results ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results = {
        "global_sharpe": global_sharpe,
        "global_mean": float(ep_rets.mean()),
        "global_std": float(ep_rets.std()),
        "n_eval": args.n_eval,
        "episode_len": L,
        "k_assets": K,
        "n_assets_oos": len(assets),
        "asset_classes": class_counts,
        "per_class": class_results,
        "dqn_ckpt": args.dqn_ckpt,
        "oos_dir": args.oos_dir,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results → %s", args.output)
    log.info("=== OOS DONE: sharpe=%.4f | %d assets | classes=%s ===",
             global_sharpe, len(assets), class_counts)


if __name__ == "__main__":
    main()
