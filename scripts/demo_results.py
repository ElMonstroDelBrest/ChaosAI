"""Demo: visualize Strate IV agent behavior on a validation episode.

Loads the best PPO model, runs it on a random held-out episode, and
generates a 3-panel PNG: OHLCV price, agent position, cumulative PnL
vs Buy & Hold.

Usage:
    PYTHONPATH=. python scripts/demo_results.py
    PYTHONPATH=. python scripts/demo_results.py --model_path tb_logs/strate_iv/best_model/best_model.zip
    PYTHONPATH=. python scripts/demo_results.py --output demo.png --seed 42
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.strate_iv.config import load_config
from src.strate_iv.env import LatentCryptoEnv
from src.strate_iv.trajectory_buffer import TrajectoryBuffer


def run_episode(
    model: PPO, env, seed: int | None = None, use_vec: bool = False
) -> dict:
    """Run one episode and collect trajectory data."""
    if use_vec:
        obs = env.reset()
        inner = env.venv.envs[0]
        realized_idx = inner._realized_idx
        entry = inner._entry
    else:
        obs, info = env.reset(seed=seed)
        realized_idx = info["realized_future_idx"]
        entry = env._entry
        inner = env

    # Realized future OHLCV: (N_tgt, patch_len, 5)
    realized = entry.future_ohlcv[realized_idx].numpy()

    actions = []
    rewards = []
    positions = []
    cum_pnls = []

    n_tgt = inner.config.n_tgt
    for step in range(n_tgt):
        action, _ = model.predict(obs, deterministic=True)
        if use_vec:
            obs, reward_arr, done_arr, info_arr = env.step(action)
            reward = float(reward_arr[0])
            info = info_arr[0]
            terminated = done_arr[0]
        else:
            obs, reward, terminated, truncated, info = env.step(action)

        actions.append(float(action[0]))
        rewards.append(reward)
        positions.append(info["position"])
        cum_pnls.append(info["cumulative_pnl"])

        if terminated:
            break

    return {
        "realized_ohlcv": realized,
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "positions": np.array(positions),
        "cum_pnls": np.array(cum_pnls),
        "last_close": entry.last_close,
        "sigma_close": entry.revin_stds[0, 3].item(),
    }


def plot_demo(traj: dict, output_path: str) -> None:
    """Generate 3-panel visualization."""
    realized = traj["realized_ohlcv"]  # (N_tgt, patch_len, 5)
    n_tgt, patch_len, _ = realized.shape

    # Flatten close prices across patches
    close_flat = realized[:, :, 3].flatten()  # (N_tgt * patch_len,)
    time_candles = np.arange(len(close_flat))

    # Patch boundaries for agent actions (1 action per patch)
    patch_mids = np.arange(n_tgt) * patch_len + patch_len // 2
    patch_edges = np.arange(n_tgt + 1) * patch_len

    # Buy & Hold PnL: normalized to start at 0
    bh_returns = (close_flat - close_flat[0]) / close_flat[0]

    # Agent cumulative PnL re-derived in price-space for fair comparison
    agent_pnl_price = np.zeros(len(close_flat))
    cum = 0.0
    for t in range(n_tgt):
        action = traj["actions"][t]
        for c in range(patch_len):
            idx = t * patch_len + c
            if idx > 0:
                ret = (close_flat[idx] - close_flat[idx - 1]) / close_flat[idx - 1]
                cum += action * ret
            agent_pnl_price[idx] = cum

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1.5, 2]})

    # --- Panel 1: OHLCV Close Price ---
    ax1 = axes[0]
    ax1.plot(time_candles, close_flat, color="steelblue", linewidth=1.2, label="Close")

    # Shade patches by agent position
    for t in range(n_tgt):
        pos = traj["actions"][t]
        if pos > 0.1:
            color = "green"
            alpha = min(0.3, abs(pos) * 0.3)
        elif pos < -0.1:
            color = "red"
            alpha = min(0.3, abs(pos) * 0.3)
        else:
            color = "gray"
            alpha = 0.05
        ax1.axvspan(patch_edges[t], patch_edges[t + 1], color=color, alpha=alpha)

    ax1.set_ylabel("Close Price")
    ax1.set_title(f"Strate IV Agent Demo  |  "
                  f"$\\sigma_{{close}}$ = {traj['sigma_close']:.4f}  |  "
                  f"Last context close = {traj['last_close']:.2f}")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Agent Position ---
    ax2 = axes[1]
    ax2.bar(patch_mids, traj["actions"], width=patch_len * 0.8,
            color=["green" if a > 0 else "red" if a < 0 else "gray"
                   for a in traj["actions"]],
            alpha=0.7, edgecolor="black", linewidth=0.5)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Position")
    ax2.set_ylim(-1.15, 1.15)
    ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Cumulative PnL ---
    ax3 = axes[2]
    ax3.plot(time_candles, agent_pnl_price * 100, color="darkgreen",
             linewidth=2, label="Agent")
    ax3.plot(time_candles, bh_returns * 100, color="gray",
             linewidth=1.5, linestyle="--", label="Buy & Hold")
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.set_ylabel("Cumulative Return (%)")
    ax3.set_xlabel(f"Candles (1h)  |  {n_tgt} patches x {patch_len} candles = {n_tgt * patch_len}h")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)

    # Final stats annotation
    agent_final = agent_pnl_price[-1] * 100
    bh_final = bh_returns[-1] * 100
    alpha_val = agent_final - bh_final
    ax3.annotate(
        f"Agent: {agent_final:+.2f}%\nB&H: {bh_final:+.2f}%\nAlpha: {alpha_val:+.2f}%",
        xy=(0.98, 0.95), xycoords="axes fraction",
        ha="right", va="top",
        fontsize=10, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray"),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Strate IV demo visualization")
    parser.add_argument("--model_path", type=str,
                        default="tb_logs/strate_iv/best_model/best_model.zip")
    parser.add_argument("--buffer_dir", type=str,
                        default="data/trajectory_buffer/")
    parser.add_argument("--config", type=str, default="configs/strate_iv.yaml")
    parser.add_argument("--output", type=str, default="outputs/strate_iv_demo.png")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n_demos", type=int, default=1,
                        help="Number of demo episodes to generate")

    args = parser.parse_args()

    config = load_config(args.config)
    buffer = TrajectoryBuffer(args.buffer_dir)
    _, eval_buffer = buffer.split(val_ratio=config.buffer.val_ratio)
    print(f"Eval buffer: {len(eval_buffer)} episodes")

    # Try to load VecNormalize stats if available
    vecnorm_path = os.path.join(
        os.path.dirname(args.model_path), "..", "checkpoints", "vecnormalize.pkl"
    )

    if os.path.exists(vecnorm_path):
        vec_env = DummyVecEnv([lambda: LatentCryptoEnv(buffer=eval_buffer, config=config.env)])
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"Loaded VecNormalize stats from {vecnorm_path}")
        env = vec_env
        use_vec = True
    else:
        env = LatentCryptoEnv(buffer=eval_buffer, config=config.env)
        use_vec = False
        print("No VecNormalize stats found, using raw observations")

    model = PPO.load(args.model_path)
    print(f"Loaded model from {args.model_path}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    for i in range(args.n_demos):
        seed = (args.seed + i) if args.seed is not None else None
        traj = run_episode(model, env, seed=seed, use_vec=use_vec)

        if args.n_demos > 1:
            stem = Path(args.output).stem
            suffix = Path(args.output).suffix
            out_path = str(Path(args.output).parent / f"{stem}_{i:02d}{suffix}")
        else:
            out_path = args.output

        plot_demo(traj, out_path)

        print(f"  Episode {i}: "
              f"agent PnL = {traj['cum_pnls'][-1]:.2f}, "
              f"actions = {traj['actions'].round(2).tolist()}")


if __name__ == "__main__":
    main()
