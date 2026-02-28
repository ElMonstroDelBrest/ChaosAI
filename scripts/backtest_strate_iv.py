"""Backtest trained TD-MPC2 agent on dual-scale test set.

Loads trained agent, runs MPPI planning on test observations,
computes per-pair Sharpe/CumRet/MaxDD with median+IQR aggregation.

Usage on TPU VM:
    PYTHONPATH=. python3 -u scripts/backtest_strate_iv.py \
      --buffer_dir data/dual_buffer/ \
      --checkpoint checkpoints/strate_iv_jax/best_agent.pkl
"""

import argparse
import json
import os
import pickle
import sys
import time

if __name__ != "__main__":
    sys.exit(0)

from src.common.env_setup import setup_tpu_env
setup_tpu_env()

import numpy as np
import jax
import jax.numpy as jnp

from src.jax_v6.strate_iv.tdmpc2 import TDMPC2Agent
from src.jax_v6.config import StrateIVJAXConfig

from src.common.metrics import compute_sharpe as _compute_sharpe, compute_max_drawdown, compute_cum_return

TX_COST = 0.0008
ANNUALIZE = np.sqrt(252 * 24 * 60 / 32)


def p(msg):
    print(msg, flush=True)


def compute_sharpe(returns):
    return _compute_sharpe(returns, annualize=ANNUALIZE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer_dir", default="data/dual_buffer/")
    parser.add_argument("--checkpoint", default="checkpoints/strate_iv_jax/best_agent.pkl")
    parser.add_argument("--output", default="results/backtest_strate_iv.json")
    args = parser.parse_args()

    p("=" * 75)
    p("  BACKTEST — Strate IV TD-MPC2 Agent")
    p("=" * 75)
    t_global = time.time()

    # Load buffer + meta
    p("\n[1/4] Loading test data...")
    test_data = dict(np.load(os.path.join(args.buffer_dir, "test.npz")))
    with open(os.path.join(args.buffer_dir, "meta.json")) as f:
        meta = json.load(f)

    obs_dim = meta["obs_dim"]
    n_test = len(test_data["obs"])
    pair_ids = test_data["pair_ids"]
    unique_pairs = np.unique(pair_ids)
    p("  %d test transitions, %d pairs, obs_dim=%d" % (n_test, len(unique_pairs), obs_dim))

    # Load agent
    p("\n[2/4] Loading trained agent...")
    with open(args.checkpoint, "rb") as f:
        ckpt = pickle.load(f)

    config = ckpt["config"]
    agent = TDMPC2Agent(config=config, obs_dim=obs_dim, action_dim=1)

    # Restore params
    agent.wm_params = jax.device_put(ckpt["wm_params"])
    agent.actor_params = jax.device_put(ckpt["actor_params"])
    agent.critic_params = jax.device_put(ckpt["critic_params"])
    agent.target_critic_params = jax.device_put(ckpt["target_critic_params"])

    p("  Restored from step %d (train best Sharpe=%.2f)" % (
        ckpt["step"], ckpt["best_sharpe"]))

    # Run agent on test set
    p("\n[3/4] Running MPPI planning on test set...")

    strat_names = ["TD-MPC2", "Buy & Hold"]
    per_pair_results = {s: [] for s in strat_names}

    for pid in unique_pairs:
        mask = pair_ids == pid
        if mask.sum() < 3:
            continue

        obs_pair = test_data["obs"][mask]
        rets_pair = test_data["reward"][mask]

        # TD-MPC2 agent positions
        positions = np.zeros(len(obs_pair), dtype=np.float32)
        for i in range(len(obs_pair)):
            # Extract convergence score from obs (micro convergence is at emb_dim*2 position)
            emb_dim = meta["emb_dim"]
            conv_score = float(obs_pair[i, 2 * emb_dim])  # first convergence feature
            conv_score = np.clip(conv_score, 0.0, 1.0)

            action = agent.select_action(
                jnp.array(obs_pair[i]),
                convergence_score=conv_score,
                eval_mode=True,
            )
            positions[i] = float(np.array(action)[0])

        # TD-MPC2 strategy returns
        tdmpc_rets = positions * rets_pair
        for i in range(1, len(positions)):
            delta_pos = abs(positions[i] - positions[i - 1])
            if delta_pos > 0.01:
                tdmpc_rets[i] -= TX_COST * delta_pos

        per_pair_results["TD-MPC2"].append((
            compute_sharpe(tdmpc_rets),
            compute_cum_return(tdmpc_rets),
            compute_max_drawdown(tdmpc_rets),
        ))

        # Buy & Hold baseline
        per_pair_results["Buy & Hold"].append((
            compute_sharpe(rets_pair),
            compute_cum_return(rets_pair),
            compute_max_drawdown(rets_pair),
        ))

    p("  Evaluated %d pairs" % len(per_pair_results["Buy & Hold"]))

    # Report
    p("\n[4/4] Results")
    p("=" * 75)
    p("  BACKTEST STRATE IV — TD-MPC2 Agent")
    p("=" * 75)
    p("  Model: latent=%d, hidden=%d, step=%d" % (
        config.latent_dim, config.hidden_dim, ckpt["step"]))
    p("  Data: %d test windows, %d pairs" % (n_test, len(unique_pairs)))
    p("  JEPA: %.1fM params (d=%d, step %d)" % (
        meta["jepa_params_M"], meta["d_model"], meta["jepa_checkpoint_step"]))
    p("  Tx cost: %.2f%% round-trip" % (TX_COST * 100))
    p("-" * 75)
    p("  %-20s %16s %16s %16s" % ("Strategy", "Sharpe", "CumRet (%)", "MaxDD (%)"))
    p("-" * 75)

    results = {}
    for name in strat_names:
        data = per_pair_results[name]
        sharpes = np.array([r[0] for r in data])
        cumrets = np.array([r[1] for r in data])
        maxdds = np.array([r[2] for r in data])

        sh_med = np.median(sharpes)
        sh_25, sh_75 = np.percentile(sharpes, [25, 75])
        cr_med = np.median(cumrets) * 100
        cr_25, cr_75 = np.percentile(cumrets, [25, 75]) * 100
        dd_med = np.median(maxdds) * 100
        dd_25, dd_75 = np.percentile(maxdds, [25, 75]) * 100

        p("  %-20s %5.2f [%5.2f,%5.2f] %5.1f [%5.1f,%5.1f] %5.1f [%5.1f,%5.1f]" % (
            name, sh_med, sh_25, sh_75, cr_med, cr_25, cr_75, dd_med, dd_25, dd_75))

        results[name] = {
            "sharpe_median": round(float(sh_med), 3),
            "sharpe_p25": round(float(sh_25), 3),
            "sharpe_p75": round(float(sh_75), 3),
            "cumret_median": round(float(np.median(cumrets)), 5),
            "maxdd_median": round(float(np.median(maxdds)), 5),
            "n_pairs": len(data),
        }

    p("-" * 75)
    tdmpc_sharpe = results["TD-MPC2"]["sharpe_median"]
    bh_sharpe = results["Buy & Hold"]["sharpe_median"]
    p("  TD-MPC2 vs B&H: Sharpe %.2f vs %.2f (%s)" % (
        tdmpc_sharpe, bh_sharpe,
        "BEATS" if tdmpc_sharpe > bh_sharpe else "LOSES"))

    elapsed = time.time() - t_global
    p("\n  Total time: %.1fs" % elapsed)
    p("=" * 75)

    # Export JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {
        "agent": {
            "latent_dim": config.latent_dim,
            "hidden_dim": config.hidden_dim,
            "train_step": ckpt["step"],
            "train_best_sharpe": ckpt["best_sharpe"],
        },
        "jepa": {
            "params_M": meta["jepa_params_M"],
            "d_model": meta["d_model"],
            "checkpoint_step": meta["jepa_checkpoint_step"],
        },
        "data": {
            "n_test": n_test,
            "n_pairs": len(unique_pairs),
            "obs_dim": obs_dim,
        },
        "tx_cost_roundtrip": TX_COST,
        "strategies": results,
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    p("  Results saved to %s" % args.output)


if __name__ == "__main__":
    main()
