"""LatentMultiverseEnv: Gymnasium environment with Multiverse Crossing.

Fork of src/strate_iv/env.py adapted for M independent multiverse perturbations.

Key difference: future_latents shape is (M, N, N_tgt, d_model) instead of
(N, N_tgt, d_model). At each step, convergence metrics across the M universes
are computed and appended to the observation vector.

Observation vector (concatenated):
  h_x_pooled:         (d_model,)      — JEPA context (static)
  future_mean_t:      (d_model,)      — mean across M×N futures at step t
  future_std_t:       (d_model,)      — std across M×N futures at step t
  close_stats:        (N_tgt * 3,)    — per-target [mean, std, skew] (static)
  revin_stds:         (5,)            — volatility regime (static)
  delta_mu:           (1,)            — macro trend (static)
  step_progress:      (1,)            — t / N_tgt (dynamic)
  realized_returns:   (N_tgt,)        — close returns so far (dynamic)
  position:           (1,)            — portfolio position (dynamic)
  cumulative_pnl:     (1,)            — running PnL (dynamic)
  convergence_score:  (1,)            — inter-universe agreement (dynamic)
  divergence_rate:    (1,)            — Δ(inter_std) (dynamic)
  inter_mv_std:       (1,)            — spread of universe means (dynamic)
  bifurcation_index:  (1,)            — soft cluster count (dynamic)
  lyapunov_proxy:     (1,)            — chaos indicator (dynamic)

obs_dim = 3*d_model + N_tgt*4 + 5 + 4 + 5  (the +5 is multiverse metrics)
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import EPS
from .multiverse_crossing import compute_convergence

# Use jax.numpy only for convergence computation (called with numpy arrays cast to jnp)
import jax.numpy as jnp


@dataclass
class MultiverseTrajectoryEntry:
    """Pre-computed trajectory data for one episode with M multiverse perturbations.

    Attributes:
        h_x_pooled:      (d_model,) np.ndarray — JEPA context mean-pool.
        future_latents:   (M, N, N_tgt, d_model) np.ndarray — M perturbed × N futures.
        future_ohlcv:     (N, N_tgt, patch_len, 5) np.ndarray — OHLCV of original futures.
        context_ohlcv:    (T, 5) np.ndarray — raw context candles.
        revin_stds:       (1, 5) np.ndarray — RevIN channel stds.
        revin_means:      (1, 5) np.ndarray — RevIN channel means.
    """

    h_x_pooled: np.ndarray
    future_latents: np.ndarray  # (M, N, N_tgt, d_model)
    future_ohlcv: np.ndarray
    context_ohlcv: np.ndarray
    revin_stds: np.ndarray
    revin_means: np.ndarray


@dataclass(frozen=True)
class MultiverseEnvConfig:
    """Environment configuration for multiverse crossing."""

    n_tgt: int = 8
    tc_rate: float = 0.002
    patch_len: int = 16
    dead_market_threshold: float = 1e-4
    perturbation_sigma: float = 0.01


class LatentMultiverseEnv(gym.Env):
    """Latent-space crypto trading env with multiverse crossing observations.

    Args:
        entries: List of MultiverseTrajectoryEntry (pre-computed offline).
        config: Environment configuration.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        entries: list[MultiverseTrajectoryEntry],
        config: MultiverseEnvConfig | None = None,
    ) -> None:
        super().__init__()
        self.entries = entries
        self.config = config or MultiverseEnvConfig()

        if len(entries) == 0:
            raise ValueError("Entry list is empty — cannot auto-detect obs_dim.")

        # Episode state
        self._entry: MultiverseTrajectoryEntry | None = None
        self._realized_idx: int = 0
        self._step_idx: int = 0
        self._position: float = 0.0
        self._cumulative_pnl: float = 0.0
        self._prev_inter_std: float = 0.0

        # Auto-detect obs_dim
        sample_obs = self._build_observation_from(entries[0])
        obs_dim = sample_obs.shape[0]

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32,
        )

    def reset(
        self, *, seed: int | None = None, options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        entry_idx = self.np_random.integers(0, len(self.entries))
        self._entry = self.entries[entry_idx]

        n_futures = self._entry.future_ohlcv.shape[0]
        self._realized_idx = self.np_random.integers(0, n_futures)

        self._step_idx = 0
        self._position = 0.0
        self._cumulative_pnl = 0.0
        self._prev_inter_std = 0.0

        obs = self._build_observation()
        return obs, {"realized_future_idx": self._realized_idx}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_val = float(np.clip(action[0], -1.0, 1.0))

        realized = self._entry.future_ohlcv[self._realized_idx]  # (N_tgt, patch_len, 5)
        close_current = realized[self._step_idx, -1, 3].item()

        if self._step_idx < self.config.n_tgt - 1:
            close_next = realized[self._step_idx + 1, -1, 3].item()
        else:
            close_next = close_current

        # Simple PnL reward: position × log_return - transaction_cost
        log_ret = np.log(close_next / (close_current + EPS) + EPS)
        tc = self.config.tc_rate * abs(action_val - self._position)
        reward = float(action_val * log_ret - tc)

        self._position = action_val
        self._cumulative_pnl += reward
        self._step_idx += 1

        terminated = self._step_idx >= self.config.n_tgt
        obs = self._build_observation()

        info = {
            "log_return": log_ret,
            "tc_penalty": tc,
            "position": self._position,
            "cumulative_pnl": self._cumulative_pnl,
            "step": self._step_idx,
        }
        return obs, reward, terminated, False, info

    def _build_observation(self) -> np.ndarray:
        return self._build_observation_from(
            self._entry, self._step_idx, self._realized_idx,
            self._position, self._cumulative_pnl, self._prev_inter_std,
        )

    def _build_observation_from(
        self,
        entry: MultiverseTrajectoryEntry,
        step_idx: int = 0,
        realized_idx: int = 0,
        position: float = 0.0,
        cum_pnl: float = 0.0,
        prev_inter_std: float = 0.0,
    ) -> np.ndarray:
        n_tgt = self.config.n_tgt
        future_latents = entry.future_latents  # (M, N, N_tgt, d_model)

        h_x_pooled = entry.h_x_pooled  # (d_model,)

        latent_step = min(step_idx, n_tgt - 1)

        # Flatten M×N for mean/std computation
        all_futures_t = future_latents[:, :, latent_step, :]  # (M, N, d_model)
        flat_futures = all_futures_t.reshape(-1, all_futures_t.shape[-1])  # (M*N, d_model)
        future_mean_t = flat_futures.mean(axis=0)  # (d_model,)
        future_std_t = flat_futures.std(axis=0)  # (d_model,)

        # Close stats (static)
        close_stats = self._compute_close_stats(entry.future_ohlcv)

        # RevIN stds (static)
        revin_stds = entry.revin_stds.flatten()  # (5,)

        # Delta mu (static)
        delta_mu = self._compute_delta_mu(entry, self.config.patch_len)

        # Step progress (dynamic)
        step_progress = np.array([step_idx / n_tgt], dtype=np.float32)

        # Realized returns (dynamic)
        realized = entry.future_ohlcv[realized_idx]  # (N_tgt, patch_len, 5)
        closes = realized[:, -1, 3]
        realized_returns = np.zeros(n_tgt, dtype=np.float32)
        for s in range(min(step_idx, n_tgt - 1)):
            realized_returns[s] = (closes[s + 1] - closes[s]) / (abs(closes[s]) + EPS)

        # Soft noise gate (same as original env)
        close_mean = abs(entry.revin_means[0, 3].item())
        close_std = revin_stds[3]
        relative_vol = close_std / (close_mean + EPS)
        temperature = 100_000.0
        gate = float(1.0 / (1.0 + np.exp(
            -(relative_vol - self.config.dead_market_threshold) * temperature
        )))
        future_mean_t = future_mean_t * gate
        future_std_t = future_std_t * gate
        close_stats = close_stats * gate
        delta_mu = delta_mu * gate
        realized_returns = realized_returns * gate

        pos = np.array([position], dtype=np.float32)
        cpnl = np.array([cum_pnl], dtype=np.float32)

        # --- Multiverse convergence metrics ---
        mv_futures_jnp = jnp.array(all_futures_t)  # (M, N, d_model)
        conv_metrics = compute_convergence(
            mv_futures_jnp,
            prev_inter_std=prev_inter_std,
            sigma=self.config.perturbation_sigma,
            t=max(step_idx, 1),
        )
        # Update prev_inter_std for next step
        self._prev_inter_std = float(conv_metrics["inter_mv_std"])

        mv_obs = np.array([
            float(conv_metrics["convergence_score"]),
            float(conv_metrics["divergence_rate"]),
            float(conv_metrics["inter_mv_std"]),
            float(conv_metrics["bifurcation_index"]),
            float(conv_metrics["lyapunov_proxy"]),
        ], dtype=np.float32)

        obs = np.concatenate([
            h_x_pooled, future_mean_t, future_std_t,
            close_stats, revin_stds, delta_mu,
            step_progress, realized_returns,
            pos, cpnl,
            mv_obs,
        ]).astype(np.float32)

        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _compute_delta_mu(entry: MultiverseTrajectoryEntry, patch_len: int) -> np.ndarray:
        context = entry.context_ohlcv  # (T, 5)
        close = context[:, 3]
        T = len(close)
        if T < 4 * patch_len:
            return np.zeros(1, dtype=np.float32)
        recent = close[-(2 * patch_len):].mean()
        earlier = close[-(4 * patch_len):-(2 * patch_len)].mean()
        sigma = close.std() + EPS
        return np.array([(recent - earlier) / sigma], dtype=np.float32)

    @staticmethod
    def _compute_close_stats(future_ohlcv: np.ndarray) -> np.ndarray:
        N, N_tgt, patch_len, _ = future_ohlcv.shape
        close_prices = future_ohlcv[:, :, -1, 3]  # (N, N_tgt)
        close_shifted = np.concatenate([
            close_prices[:, :1], close_prices[:, :-1],
        ], axis=1)
        returns = (close_prices - close_shifted) / (np.abs(close_shifted) + EPS)
        stats = []
        for t in range(N_tgt):
            r = returns[:, t]
            mean = r.mean()
            std = r.std() + EPS
            skew = ((r - mean) ** 3).mean() / (std ** 3)
            stats.extend([mean, std, skew])
        return np.array(stats, dtype=np.float32)
