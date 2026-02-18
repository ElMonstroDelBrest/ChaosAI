"""Configuration dataclasses for Strate IV RL training.

Defines frozen dataclasses for the environment, buffer, PPO and TD-MPC2 hyperparameters.
Loaded from YAML via dacite with strict mode (no unknown keys allowed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dacite import from_dict, Config as DaciteConfig


@dataclass(frozen=True)
class EnvConfig:
    """RL environment parameters."""

    n_tgt: int = 8
    tc_rate: float = 0.002
    patch_len: int = 16
    dead_market_threshold: float = 1e-4


@dataclass(frozen=True)
class BufferConfig:
    """Pre-computed trajectory buffer parameters."""

    buffer_dir: str = "data/trajectory_buffer/"
    n_episodes: int = 255
    n_futures: int = 16
    val_ratio: float = 0.2


@dataclass(frozen=True)
class PPOConfig:
    """Stable-Baselines3 PPO hyperparameters."""

    lr: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    total_timesteps: int = 1_000_000
    eval_freq: int = 10_000
    log_dir: str = "tb_logs/strate_iv/"


@dataclass(frozen=True)
class TDMPC2Config:
    """TD-MPC2 + CVaR hyperparameters (Phase E).

    Args:
        latent_dim: Planning latent dimension (world model internal space).
        hidden_dim: Hidden dimension for all MLPs.
        n_layers: Depth of all MLP components.
        n_quantiles: Number of quantiles for distributional critic.
        cvar_alpha: CVaR confidence level (0.25 = worst-25% conditional expectation).
        gamma: Discount factor.
        lr: Learning rate shared by world model, actor, and critic.
        ema_tau: EMA coefficient for target critic update (0.005 = slow).
        max_grad_norm: Gradient clipping norm.
        batch_size: Replay buffer batch size per update.
        buffer_capacity: Replay buffer max capacity.
        warmup_steps: Random exploration steps before training begins.
        update_freq: Update every N environment steps.
        use_planning: Use MPPI planning at inference (True) or actor-only (False).
        plan_horizon: MPPI planning horizon (steps).
        plan_samples: Number of trajectory samples in MPPI.
        plan_iters: Number of MPPI refinement iterations.
        plan_temperature: MPPI softmax temperature for weighting.
        plan_init_std: Initial action std for MPPI sampling.
        total_timesteps: Total environment steps for training.
        eval_freq: Evaluate every N steps.
        log_dir: TensorBoard log directory.
        save_dir: Checkpoint save directory.
    """

    latent_dim: int = 256
    hidden_dim: int = 512
    n_layers: int = 2
    n_quantiles: int = 32
    cvar_alpha: float = 0.25
    gamma: float = 0.99
    lr: float = 3e-4
    ema_tau: float = 0.005
    max_grad_norm: float = 10.0
    batch_size: int = 256
    buffer_capacity: int = 100_000
    warmup_steps: int = 1_000
    update_freq: int = 1
    use_planning: bool = True
    plan_horizon: int = 5
    plan_samples: int = 512
    plan_iters: int = 6
    plan_temperature: float = 0.5
    plan_init_std: float = 0.5
    total_timesteps: int = 500_000
    eval_freq: int = 5_000
    log_dir: str = "tb_logs/strate_iv_v6/"
    save_dir: str = "checkpoints/strate_iv_tdmpc2/"


@dataclass(frozen=True)
class StrateIVConfig:
    """Root configuration composing env, buffer, PPO, and TD-MPC2 sections."""

    env: EnvConfig = field(default_factory=EnvConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    tdmpc2: TDMPC2Config = field(default_factory=TDMPC2Config)


def load_config(path: str | Path) -> StrateIVConfig:
    """Load and validate a Strate IV config from YAML."""
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)
    return from_dict(
        data_class=StrateIVConfig,
        data=config_dict,
        config=DaciteConfig(strict=True),
    )
