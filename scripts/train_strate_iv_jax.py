"""Offline TD-MPC2 training on dual-scale replay buffer.

Trains world model (dynamics + reward), distributional critic, and actor
on pre-computed (obs, action, reward, next_obs, done) transitions.

Usage on TPU VM:
    PYTHONPATH=. python3 -u scripts/train_strate_iv_jax.py \
      --buffer_dir data/dual_buffer/ \
      --total_steps 50000 \
      --eval_interval 2000
"""

import argparse
import json
import os
import sys
import time

if __name__ != "__main__":
    sys.exit(0)

from src.common.env_setup import setup_tpu_env
setup_tpu_env()

import numpy as np
import jax

from src.jax_v6.strate_iv.tdmpc2 import TDMPC2Agent
from src.jax_v6.strate_iv.replay_buffer import ReplayBuffer
from src.jax_v6.config import StrateIVJAXConfig


def p(msg):
    print(msg, flush=True)


def load_config_from_yaml(path):
    """Load StrateIVJAXConfig from YAML."""
    import yaml
    from dacite import from_dict, Config as DaciteConfig
    with open(path) as f:
        raw = yaml.safe_load(f)
    strate_iv_dict = raw.get("strate_iv", raw)
    return from_dict(
        data_class=StrateIVJAXConfig,
        data=strate_iv_dict,
        config=DaciteConfig(strict=False),
    )


from src.common.metrics import compute_sharpe


def evaluate_agent(agent, test_data, pair_ids, tx_cost=0.0008):
    """Run agent in eval mode on test data, compute per-pair metrics."""
    unique_pairs = np.unique(pair_ids)
    annualize = np.sqrt(252 * 24 * 60 / 32)  # 32-min windows
    per_pair_sharpe = []

    for pid in unique_pairs:
        mask = pair_ids == pid
        if mask.sum() < 3:
            continue
        obs_pair = test_data["obs"][mask]
        rets_pair = test_data["reward"][mask]

        # Agent decides position for each window
        positions = np.zeros(len(obs_pair), dtype=np.float32)
        for i in range(len(obs_pair)):
            action = agent.select_action(
                jax.numpy.array(obs_pair[i]),
                convergence_score=0.5,
                eval_mode=True,
            )
            positions[i] = float(np.array(action)[0])

        # Strategy returns
        strat_rets = positions * rets_pair
        # Transaction costs
        for i in range(1, len(positions)):
            if abs(positions[i] - positions[i - 1]) > 0.01:
                strat_rets[i] -= tx_cost * abs(positions[i] - positions[i - 1])

        per_pair_sharpe.append(compute_sharpe(strat_rets, annualize))

    return {
        "median_sharpe": float(np.median(per_pair_sharpe)) if per_pair_sharpe else 0.0,
        "mean_sharpe": float(np.mean(per_pair_sharpe)) if per_pair_sharpe else 0.0,
        "p25_sharpe": float(np.percentile(per_pair_sharpe, 25)) if per_pair_sharpe else 0.0,
        "p75_sharpe": float(np.percentile(per_pair_sharpe, 75)) if per_pair_sharpe else 0.0,
        "n_pairs": len(per_pair_sharpe),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer_dir", default="data/dual_buffer/")
    parser.add_argument("--config", default="configs/strate_iv_dual.yaml")
    parser.add_argument("--total_steps", type=int, default=50000)
    parser.add_argument("--eval_interval", type=int, default=5000)
    parser.add_argument("--checkpoint_dir", default="checkpoints/strate_iv_jax/")
    parser.add_argument("--log_interval", type=int, default=500)
    args = parser.parse_args()

    p("=" * 65)
    p("  STRATE IV — Offline TD-MPC2 Training")
    p("=" * 65)
    t_global = time.time()

    # Load config
    config = load_config_from_yaml(args.config)
    p("\n  Config: latent_dim=%d, hidden_dim=%d, n_layers=%d" % (
        config.latent_dim, config.hidden_dim, config.n_layers))
    p("  obs_dim=%d (from config)" % config.obs_dim)

    # Load buffer
    p("\n[1/3] Loading replay buffer...")
    train_data = dict(np.load(os.path.join(args.buffer_dir, "train.npz")))
    test_data = dict(np.load(os.path.join(args.buffer_dir, "test.npz")))
    with open(os.path.join(args.buffer_dir, "meta.json")) as f:
        meta = json.load(f)

    obs_dim = meta["obs_dim"]
    n_train = len(train_data["obs"])
    n_test = len(test_data["obs"])
    p("  Train: %d transitions, Test: %d transitions" % (n_train, n_test))
    p("  obs_dim=%d, emb_dim=%d, %d pairs" % (obs_dim, meta["emb_dim"], meta["n_pairs"]))

    # Fill replay buffer
    buffer = ReplayBuffer(
        capacity=min(config.buffer_capacity, n_train),
        obs_dim=obs_dim,
        action_dim=1,
    )
    for i in range(n_train):
        buffer.add(
            train_data["obs"][i],
            train_data["action"][i],
            train_data["reward"][i],
            train_data["next_obs"][i],
            train_data["done"][i],
        )
    p("  Buffer filled: %d / %d" % (len(buffer), buffer.capacity))

    # Create agent
    p("\n[2/3] Creating TD-MPC2 agent...")
    agent = TDMPC2Agent(
        config=config,
        obs_dim=obs_dim,
        action_dim=1,
    )
    param_counts = agent.param_count()
    p("  Agent params: %s" % {k: f"{v:,}" for k, v in param_counts.items()})

    # Training loop
    p("\n[3/3] Training for %d steps..." % args.total_steps)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    metrics_log = []
    best_sharpe = -np.inf

    for step in range(1, args.total_steps + 1):
        batch = buffer.sample_async(config.batch_size)
        metrics = agent.update(batch)

        if step % args.log_interval == 0:
            p("  Step %5d/%d | consist=%.4f rew=%.4f crit=%.4f act=%.4f cql=%.4f" % (
                step, args.total_steps,
                metrics.get("loss/consistency", 0),
                metrics.get("loss/reward", 0),
                metrics.get("loss/critic", 0),
                metrics.get("loss/actor", 0),
                metrics.get("loss/cql", 0),
            ))
            metrics_log.append({"step": step, **{k: float(v) for k, v in metrics.items()}})

        if step % args.eval_interval == 0:
            p("\n  [Eval @ step %d]" % step)
            eval_res = evaluate_agent(agent, test_data, test_data["pair_ids"])
            p("  Median Sharpe: %.2f [%.2f, %.2f] (%d pairs)" % (
                eval_res["median_sharpe"], eval_res["p25_sharpe"],
                eval_res["p75_sharpe"], eval_res["n_pairs"]))

            if eval_res["median_sharpe"] > best_sharpe:
                best_sharpe = eval_res["median_sharpe"]
                # Save best checkpoint
                import pickle
                ckpt_path = os.path.join(args.checkpoint_dir, "best_agent.pkl")
                with open(ckpt_path, "wb") as f:
                    pickle.dump({
                        "wm_params": jax.device_get(agent.wm_params),
                        "actor_params": jax.device_get(agent.actor_params),
                        "critic_params": jax.device_get(agent.critic_params),
                        "target_critic_params": jax.device_get(agent.target_critic_params),
                        "step": step,
                        "best_sharpe": best_sharpe,
                        "eval": eval_res,
                        "config": config,
                    }, f)
                p("  New best! Saved to %s" % ckpt_path)

            metrics_log.append({"step": step, "eval": eval_res})

    # Final eval
    p("\n  [Final eval]")
    final_eval = evaluate_agent(agent, test_data, test_data["pair_ids"])
    p("  Median Sharpe: %.2f [%.2f, %.2f]" % (
        final_eval["median_sharpe"], final_eval["p25_sharpe"], final_eval["p75_sharpe"]))

    elapsed = time.time() - t_global
    p("\n  Total training time: %.1fs (%.1f steps/s)" % (
        elapsed, args.total_steps / elapsed))
    p("  Best median Sharpe: %.2f" % best_sharpe)

    # Save training log
    log_path = os.path.join(args.checkpoint_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "config": {
                "obs_dim": obs_dim,
                "latent_dim": config.latent_dim,
                "hidden_dim": config.hidden_dim,
                "total_steps": args.total_steps,
                "batch_size": config.batch_size,
                "lr": config.lr,
            },
            "best_sharpe": best_sharpe,
            "final_eval": final_eval,
            "metrics": metrics_log,
            "elapsed_seconds": round(elapsed, 1),
        }, f, indent=2)
    p("  Log saved to %s" % log_path)
    p("=" * 65)


if __name__ == "__main__":
    main()
