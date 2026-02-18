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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Strate IV PPO agent")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run smoke test with synthetic data")
    parser.add_argument("--config", type=str, default="configs/strate_iv.yaml")
    parser.add_argument("--buffer_dir", type=str, default=None,
                        help="Override buffer dir from config")
    parser.add_argument("--total_timesteps", type=int, default=None,
                        help="Override total_timesteps from config")
    parser.add_argument("--no_resume", action="store_true",
                        help="Force fresh start, ignore existing checkpoints")
    parser.add_argument("--n_envs", type=int, default=8,
                        help="Number of parallel environments (SubprocVecEnv workers)")

    args = parser.parse_args()

    if args.smoke_test:
        smoke_test(total_timesteps=args.total_timesteps or 1000)
    else:
        train(args)


if __name__ == "__main__":
    main()
