"""Configuration dataclasses for Strate IV RL training.

Defines frozen dataclasses for the environment, buffer, and PPO hyperparameters.
Loaded from YAML via dacite with strict mode (no unknown keys allowed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dacite import from_dict, Config as DaciteConfig


@dataclass(frozen=True)
class EnvConfig:
    """RL environment parameters.

    Args:
        n_tgt: Number of target patches per episode (episode length in steps).
        tc_rate: Transaction cost rate per unit of position change (20 bps).
        patch_len: Candles per patch (must match Strate I/II tokenizer).
        dead_market_threshold: Relative volatility below which market signals
            are zeroed out (noise gate). Prevents hallucinated positions on
            flat/dead markets where RevIN amplifies floating-point noise.
    """

    n_tgt: int = 8
    tc_rate: float = 0.002
    patch_len: int = 16
    dead_market_threshold: float = 1e-4


@dataclass(frozen=True)
class BufferConfig:
    """Pre-computed trajectory buffer parameters.

    Args:
        buffer_dir: Directory containing episode .pt files.
        n_episodes: Target number of episodes to pre-compute.
        n_futures: Number of future trajectories per episode (N).
        val_ratio: Fraction of episodes reserved for evaluation.
    """

    buffer_dir: str = "data/trajectory_buffer/"
    n_episodes: int = 255
    n_futures: int = 16
    val_ratio: float = 0.2


@dataclass(frozen=True)
class PPOConfig:
    """Stable-Baselines3 PPO hyperparameters.

    Args:
        lr: Learning rate.
        n_steps: Rollout length per environment per update.
        batch_size: Minibatch size for PPO updates.
        n_epochs: Number of SGD passes per PPO update.
        gamma: Discount factor.
        gae_lambda: GAE lambda for advantage estimation.
        clip_range: PPO clipping parameter.
        ent_coef: Entropy bonus coefficient.
        vf_coef: Value function loss coefficient.
        max_grad_norm: Gradient clipping norm.
        total_timesteps: Total training timesteps.
        eval_freq: Evaluate every N timesteps.
        log_dir: TensorBoard log directory.
    """

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
class StrateIVConfig:
    """Root configuration composing env, buffer, and PPO sections."""

    env: EnvConfig = field(default_factory=EnvConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)


def load_config(path: str | Path) -> StrateIVConfig:
    """Load and validate a Strate IV config from YAML.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated StrateIVConfig instance.
    """
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)
    return from_dict(
        data_class=StrateIVConfig,
        data=config_dict,
        config=DaciteConfig(strict=True),
    )
