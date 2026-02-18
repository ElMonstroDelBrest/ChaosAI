"""Train PPO agent on LatentCryptoEnv (Strate IV).

Usage:
    # Smoke test with synthetic buffer:
    PYTHONPATH=. python scripts/train_strate_iv.py --smoke_test

    # Full training from pre-computed buffer:
    PYTHONPATH=. python scripts/train_strate_iv.py \
        --config configs/strate_iv.yaml \
        --buffer_dir data/trajectory_buffer/
"""

import argparse
from pathlib import Path

import numpy as np
import torch


def smoke_test(total_timesteps: int = 1000) -> None:
    """Smoke test: train PPO on synthetic buffer for a few steps."""
    from src.strate_iv.config import EnvConfig
    from src.strate_iv.trajectory_buffer import TrajectoryBuffer, TrajectoryEntry
    from src.strate_iv.env import LatentCryptoEnv

    print("=== Smoke test: training PPO on synthetic buffer ===")

    # Create synthetic buffer (in-memory, no disk)
    n_entries = 20
    n_futures, n_tgt, patch_len, d_model = 4, 8, 16, 128

    entries = []
    for _ in range(n_entries):
        entry = TrajectoryEntry(
            context_tokens=torch.randint(0, 1024, (64,)),
            weekend_mask=torch.zeros(64),
            context_ohlcv=torch.rand(256, 5) * 100 + 50,
            future_ohlcv=torch.rand(n_futures, n_tgt, patch_len, 5) * 10 + 95,
            future_latents=torch.randn(n_futures, n_tgt, d_model),
            revin_means=torch.zeros(1, 5),
            revin_stds=torch.ones(1, 5) * 0.02,
            last_close=100.0,
            h_x_pooled=torch.randn(d_model),
        )
        entries.append(entry)

    buf = TrajectoryBuffer.from_entries(entries)
    config = EnvConfig(n_tgt=n_tgt, tc_rate=0.0005, patch_len=patch_len)
    env = LatentCryptoEnv(buffer=buf, config=config)

    # Quick manual rollout test
    print("  Running manual rollout test...")
    obs, info = env.reset()
    total_reward = 0.0
    for step in range(config.n_tgt):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            break
    print(f"  Manual rollout: {step + 1} steps, total reward = {total_reward:.4f}")

    # Train with SB3 PPO
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env

        check_env(env, warn=True)
        print("  Environment check passed.")

        log_dir = "tb_logs/strate_iv_smoke/"
        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4,
            n_steps=64,  # Small for smoke test
            batch_size=32,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=log_dir,
        )

        print(f"  Training PPO for {total_timesteps} steps...")
        model.learn(total_timesteps=total_timesteps)
        print("  Training complete.")

        # Evaluate
        obs, _ = env.reset()
        rewards = []
        for _ in range(config.n_tgt):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            if terminated:
                break
        print(f"  Evaluation: total reward = {sum(rewards):.4f}")
        print(f"  Rewards per step: {[f'{r:.4f}' for r in rewards]}")
        print(f"  All rewards finite: {all(np.isfinite(rewards))}")

    except ImportError:
        print("  stable-baselines3 not installed, skipping PPO training.")

    print("=== Smoke test passed ===")


def train(args: argparse.Namespace) -> None:
    """Full PPO training from pre-computed buffer."""
    from src.strate_iv.config import load_config
    from src.strate_iv.trajectory_buffer import TrajectoryBuffer
    from src.strate_iv.env import LatentCryptoEnv

    config = load_config(args.config)
    buffer_dir = args.buffer_dir or config.buffer.buffer_dir

    print(f"Loading buffer from {buffer_dir}...")
    full_buffer = TrajectoryBuffer(buffer_dir)
    print(f"  Loaded {len(full_buffer)} episodes")

    if len(full_buffer) == 0:
        raise RuntimeError(
            f"No episodes in {buffer_dir}. Run precompute_trajectories.py first."
        )

    # Split into train/eval for overfitting detection
    train_buffer, eval_buffer = full_buffer.split(
        val_ratio=config.buffer.val_ratio,
    )
    print(f"  Train: {len(train_buffer)} episodes, Eval: {len(eval_buffer)} episodes")

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback, CallbackList, CheckpointCallback, EvalCallback,
    )
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    # Use SubprocVecEnv for parallel rollouts (exploits multi-core CPU)
    n_envs = args.n_envs if hasattr(args, "n_envs") and args.n_envs else 8
    print(f"  Using {n_envs} parallel environments (SubprocVecEnv)")

    train_env = SubprocVecEnv([
        lambda i=i: LatentCryptoEnv(buffer=train_buffer, config=config.env)
        for i in range(n_envs)
    ])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    eval_env = DummyVecEnv([lambda: LatentCryptoEnv(buffer=eval_buffer, config=config.env)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
                            training=False)

    log_dir = Path(config.ppo.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = log_dir / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)

    # Auto-resume: check for existing model + VecNormalize stats
    last_model_path = checkpoint_dir / "ppo_strate_iv_last.zip"
    vecnorm_path = checkpoint_dir / "vecnormalize_v4.pkl"
    best_vecnorm_path = best_model_dir / "vecnormalize.pkl"
    resuming = last_model_path.exists() and not args.no_resume

    if resuming:
        print(f"Auto-resuming from {last_model_path}")
        if vecnorm_path.exists():
            train_env = VecNormalize.load(str(vecnorm_path), train_env.venv)
            print(f"  Loaded VecNormalize stats from {vecnorm_path}")
        model = PPO.load(str(last_model_path), env=train_env,
                         tensorboard_log=str(log_dir))
    else:
        model = PPO(
            "MlpPolicy", train_env,
            learning_rate=config.ppo.lr,
            n_steps=config.ppo.n_steps,
            batch_size=config.ppo.batch_size,
            n_epochs=config.ppo.n_epochs,
            gamma=config.ppo.gamma,
            gae_lambda=config.ppo.gae_lambda,
            clip_range=config.ppo.clip_range,
            ent_coef=config.ppo.ent_coef,
            vf_coef=config.ppo.vf_coef,
            max_grad_norm=config.ppo.max_grad_norm,
            target_kl=getattr(config.ppo, 'target_kl', None),
            verbose=1,
            tensorboard_log=str(log_dir),
            policy_kwargs=dict(
                net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
            ),
        )

    # Sync eval env normalization stats from training env
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms

    checkpoint_callback = CheckpointCallback(
        save_freq=5_000,
        save_path=str(checkpoint_dir),
        name_prefix="ppo_strate_iv",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(log_dir / "eval_logs"),
        eval_freq=config.ppo.eval_freq,
        n_eval_episodes=len(eval_buffer),
        deterministic=True,
        verbose=1,
    )

    class VecNormalizeSyncCallback(BaseCallback):
        """Sync VecNormalize stats to eval env and save periodically."""

        def __init__(self) -> None:
            super().__init__()
            self._last_best = None

        def _on_step(self) -> bool:
            eval_env.obs_rms = train_env.obs_rms
            eval_env.ret_rms = train_env.ret_rms
            return True

        def _on_rollout_end(self) -> None:
            train_env.save(str(vecnorm_path))
            # Also save alongside best_model whenever it's updated
            best_zip = best_model_dir / "best_model.zip"
            if best_zip.exists():
                mtime = best_zip.stat().st_mtime
                if mtime != self._last_best:
                    self._last_best = mtime
                    train_env.save(str(best_vecnorm_path))
                    print(f"  VecNormalize saved to {best_vecnorm_path}")

    sync_callback = VecNormalizeSyncCallback()
    callbacks = CallbackList([checkpoint_callback, eval_callback, sync_callback])

    total_timesteps = config.ppo.total_timesteps
    if args.total_timesteps is not None:
        total_timesteps = args.total_timesteps

    print(f"Training PPO for {total_timesteps} timesteps...")
    print(f"  Eval every {config.ppo.eval_freq} steps on {len(eval_buffer)} held-out episodes")
    print(f"  Observation normalization: ON (VecNormalize)")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=not resuming,
    )

    # Save final + last model + VecNormalize stats
    final_path = log_dir / "ppo_strate_iv_final"
    model.save(str(final_path))
    model.save(str(last_model_path))
    train_env.save(str(vecnorm_path))
    train_env.save(str(best_vecnorm_path))
    print(f"Final model saved to {final_path}")
    print(f"Last model saved to {last_model_path}")
    print(f"VecNormalize stats saved to {vecnorm_path}")
    print(f"VecNormalize also saved to {best_vecnorm_path}")
    print(f"Best model (by eval reward) saved to {best_model_dir}")


def train_tdmpc2(args: argparse.Namespace) -> None:
    """Online TD-MPC2 + CVaR training from pre-computed trajectory buffer.

    The LatentCryptoEnv samples from the pre-computed buffer for domain randomization.
    Experience is collected online and stored in a ReplayBuffer for model-based updates.
    """
    import os
    from pathlib import Path
    from torch.utils.tensorboard import SummaryWriter

    from src.strate_iv.config import load_config
    from src.strate_iv.trajectory_buffer import TrajectoryBuffer
    from src.strate_iv.env import LatentCryptoEnv
    from src.strate_iv.replay_buffer import ReplayBuffer
    from src.strate_iv.tdmpc2 import TDMPC2Agent

    config = load_config(args.config)
    cfg = config.tdmpc2
    buffer_dir = args.buffer_dir or config.buffer.buffer_dir

    print(f"Loading trajectory buffer from {buffer_dir}...")
    full_buffer = TrajectoryBuffer(buffer_dir)
    if len(full_buffer) == 0:
        raise RuntimeError(f"No episodes in {buffer_dir}. Run precompute_trajectories.py first.")
    train_buffer, eval_buffer = full_buffer.split(val_ratio=config.buffer.val_ratio)
    print(f"  Train: {len(train_buffer)} | Eval: {len(eval_buffer)} episodes")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    env = LatentCryptoEnv(buffer=train_buffer, config=config.env)
    eval_env = LatentCryptoEnv(buffer=eval_buffer, config=config.env)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"  obs_dim: {obs_dim}, action_dim: {action_dim}")

    agent = TDMPC2Agent(cfg, obs_dim=obs_dim, action_dim=action_dim, device=device)
    replay = ReplayBuffer(
        capacity=cfg.buffer_capacity, obs_dim=obs_dim, action_dim=action_dim,
    )

    # Auto-resume
    save_dir = args.save_dir or cfg.save_dir
    resumed = False
    if not getattr(args, "no_resume", False) and os.path.exists(f"{save_dir}/world_model.pt"):
        print(f"  Auto-resuming from {save_dir}")
        agent.load(save_dir)
        resumed = True

    total_timesteps = args.total_timesteps or cfg.total_timesteps
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    print(f"Training TD-MPC2 for {total_timesteps} env steps...")
    print(f"  Warmup: {cfg.warmup_steps} random steps")
    print(f"  MPPI planning: {cfg.use_planning} (H={cfg.plan_horizon}, K={cfg.plan_samples})")
    print(f"  CVaR alpha: {cfg.cvar_alpha}")

    obs, _ = env.reset()
    episode_reward = 0.0
    episode_steps = 0
    n_episodes = 0

    for step in range(total_timesteps):
        # Action selection: random during warmup, MPPI afterwards
        if step < cfg.warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, eval=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        replay.add(obs, action, reward, next_obs, done)

        episode_reward += reward
        episode_steps += 1
        obs = next_obs

        if done:
            n_episodes += 1
            writer.add_scalar("train/episode_reward", episode_reward, step)
            writer.add_scalar("train/episode_steps", episode_steps, step)
            obs, _ = env.reset()
            episode_reward = 0.0
            episode_steps = 0

        # Update
        if step >= cfg.warmup_steps and step % cfg.update_freq == 0:
            batch = replay.sample(cfg.batch_size, device=device)
            losses = agent.update(batch)
            for k, v in losses.items():
                writer.add_scalar(k, v, step)

        # Evaluation
        if step > 0 and step % cfg.eval_freq == 0:
            eval_rewards = []
            for _ in range(min(10, len(eval_buffer))):
                e_obs, _ = eval_env.reset()
                e_ep_reward = 0.0
                e_done = False
                while not e_done:
                    e_action = agent.select_action(e_obs, eval=True)
                    e_obs, e_r, e_term, e_trunc, _ = eval_env.step(e_action)
                    e_ep_reward += e_r
                    e_done = e_term or e_trunc
                eval_rewards.append(e_ep_reward)
            mean_eval = float(np.mean(eval_rewards))
            writer.add_scalar("eval/episode_reward", mean_eval, step)
            print(f"  [{step}/{total_timesteps}] eval_reward={mean_eval:.4f}  n_episodes={n_episodes}")

            # Save checkpoint
            agent.save(save_dir)
            print(f"  Checkpoint saved to {save_dir}")

    writer.close()
    agent.save(save_dir)
    print(f"\nTraining complete. Final model saved to {save_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Strate IV RL agent")
    parser.add_argument("--mode", type=str, default="ppo",
                        choices=["ppo", "tdmpc2"],
                        help="Training mode: ppo (v5, Stable-Baselines3) or tdmpc2 (v6, Phase E)")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run smoke test with synthetic data (PPO mode only)")
    parser.add_argument("--config", type=str, default="configs/strate_iv.yaml")
    parser.add_argument("--buffer_dir", type=str, default=None,
                        help="Override buffer dir from config")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Override TD-MPC2 checkpoint save dir")
    parser.add_argument("--total_timesteps", type=int, default=None,
                        help="Override total_timesteps from config")
    parser.add_argument("--no_resume", action="store_true",
                        help="Force fresh start, ignore existing checkpoints")
    parser.add_argument("--n_envs", type=int, default=8,
                        help="Number of parallel environments (PPO SubprocVecEnv workers)")

    args = parser.parse_args()

    if args.mode == "tdmpc2":
        train_tdmpc2(args)
    elif args.smoke_test:
        smoke_test(total_timesteps=args.total_timesteps or 1000)
    else:
        train(args)


if __name__ == "__main__":
    main()
