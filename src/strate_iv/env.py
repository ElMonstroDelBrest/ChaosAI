"""LatentCryptoEnv: Gymnasium environment for Strate IV RL training.

The agent observes a **step-aware** vector composed of:
  - h_x_pooled (d_model): JEPA context encoder mean-pool (static per episode)
  - future_mean_t (d_model): Mean of N future latents at current step t (**dynamic**)
  - future_std_t (d_model): Std of N future latents at current step t (**dynamic**)
  - close_stats (N_tgt * 3): Per-target close return stats (mean/std/skew, static)
  - revin_stds (5): RevIN std per channel (volatility regime, static)
  - delta_mu (1): Macro trend from context patches (static)
  - step_progress (1): t / N_tgt — episode progress (**dynamic**)
  - realized_returns (N_tgt): Close returns of the realized future, masked for
    future steps (**dynamic** — fills in as the episode progresses)
  - position (1): Current portfolio position a_{t-1} (**dynamic**)
  - cumulative_pnl (1): Running PnL (**dynamic**)

Observation dim = 3 * d_model + N_tgt * 4 + 5 + 4  (auto-detected from buffer).

Action: Box([-1], [1]) — continuous position (-1=short, 0=flat, +1=long).

Episode: N_tgt steps. At reset, one future is sampled as "realized" (domain
randomization). The observation evolves at each step, giving the agent
step-specific latent expectations and the realized market trajectory so far.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import EPS
from .config import EnvConfig
from .reward import PnLReward
from .trajectory_buffer import TrajectoryBuffer, TrajectoryEntry


class LatentCryptoEnv(gym.Env):
    """Latent-space crypto trading environment for PPO training.

    Args:
        buffer: Pre-computed trajectory buffer to sample episodes from.
        config: Environment configuration (n_tgt, tc_rate, patch_len).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        buffer: TrajectoryBuffer,
        config: EnvConfig | None = None,
    ) -> None:
        super().__init__()
        self.buffer = buffer
        self.config = config or EnvConfig()
        self.reward_fn = PnLReward(tc_rate=self.config.tc_rate)

        # Episode state
        self._entry: TrajectoryEntry | None = None
        self._realized_idx: int = 0
        self._step_idx: int = 0
        self._position: float = 0.0
        self._cumulative_pnl: float = 0.0

        # Auto-detect obs_dim from buffer (no hardcoded fallback)
        if len(buffer) == 0:
            raise ValueError(
                "TrajectoryBuffer is empty — cannot auto-detect obs_dim. "
                "Run precompute_trajectories.py first."
            )
        sample_obs = self._build_observation_from(buffer.entries[0])
        obs_dim = sample_obs.shape[0]

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Use np_random for deterministic sampling when seeded
        entry_idx = self.np_random.integers(0, len(self.buffer))
        self._entry = self.buffer.entries[entry_idx]

        # Domain randomization: pick one future as "realized"
        n_futures = self._entry.future_ohlcv.shape[0]
        self._realized_idx = self.np_random.integers(0, n_futures)

        self._step_idx = 0
        self._position = 0.0
        self._cumulative_pnl = 0.0

        obs = self._build_observation()
        info = {"realized_future_idx": self._realized_idx}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_val = float(np.clip(action[0], -1.0, 1.0))

        # Get close prices for current and next step from realized future
        # future_ohlcv: (N, N_tgt, patch_len, 5)
        realized = self._entry.future_ohlcv[self._realized_idx]  # (N_tgt, patch_len, 5)

        # Close price = channel 3, last candle of each patch
        close_current = realized[self._step_idx, -1, 3].item()

        if self._step_idx < self.config.n_tgt - 1:
            close_next = realized[self._step_idx + 1, -1, 3].item()
        else:
            # Last step: use last candle close of current patch
            close_next = close_current

        reward, info = self.reward_fn.compute(
            action=action_val,
            prev_action=self._position,
            close_current=close_current,
            close_next=close_next,
        )

        self._position = action_val
        self._cumulative_pnl += reward
        self._step_idx += 1

        terminated = self._step_idx >= self.config.n_tgt
        truncated = False

        obs = self._build_observation()

        step_info = {
            "raw_pnl": info.raw_pnl,
            "tc_penalty": info.tc_penalty,
            "log_return": info.log_return,
            "position": self._position,
            "cumulative_pnl": self._cumulative_pnl,
            "step": self._step_idx,
        }

        return obs, reward, terminated, truncated, step_info

    def _build_observation(self) -> np.ndarray:
        """Build the observation vector from current episode state."""
        return self._build_observation_from(
            self._entry, self._step_idx, self._realized_idx,
            self._position, self._cumulative_pnl,
        )

    def _build_observation_from(
        self,
        entry: TrajectoryEntry,
        step_idx: int = 0,
        realized_idx: int = 0,
        position: float = 0.0,
        cum_pnl: float = 0.0,
    ) -> np.ndarray:
        """Build step-aware observation vector from a given entry.

        Components (concatenated):
            h_x_pooled:       (d_model,) — JEPA context representation (static)
            future_mean_t:    (d_model,) — mean of N future latents at step t (dynamic)
            future_std_t:     (d_model,) — std of N future latents at step t (dynamic)
            close_stats:      (N_tgt * 3,) — per-target [mean, std, skew] of close returns
            revin_stds:       (5,) — RevIN channel stds (volatility regime)
            delta_mu:         (1,) — macro trend from context
            step_progress:    (1,) — t / N_tgt, episode progress (dynamic)
            realized_returns: (N_tgt,) — realized close returns so far (dynamic)
            position:         (1,) — current portfolio position (dynamic)
            cumulative_pnl:   (1,) — running PnL (dynamic)

        Returns:
            (obs_dim,) float32 array with NaN/Inf replaced by 0.
        """
        n_tgt = self.config.n_tgt
        future_latents = entry.future_latents.numpy()  # (N, N_tgt, d_model)

        # 1. h_x_pooled (d_model) — static per episode
        h_x_pooled = entry.h_x_pooled.numpy()  # (d_model,)

        # 2. future_mean_t — mean across N futures at CURRENT step (dynamic)
        latent_step = min(step_idx, n_tgt - 1)
        future_mean_t = future_latents[:, latent_step, :].mean(axis=0)  # (d_model,)

        # 3. future_std_t — std across N futures at CURRENT step (dynamic)
        future_std_t = future_latents[:, latent_step, :].std(axis=0)  # (d_model,)

        # 4. close_stats — per-target: mean, std, skew of close returns (static)
        future_ohlcv = entry.future_ohlcv.numpy()
        close_stats = self._compute_close_stats(future_ohlcv)

        # 5. revin_stds (5) — volatility regime (static)
        revin_stds = entry.revin_stds.numpy().flatten()  # (5,)

        # 6. delta_mu (1) — macro trend (static)
        delta_mu = self._compute_delta_mu(entry, self.config.patch_len)

        # 7. step_progress (1) — episode progress (dynamic)
        step_progress = np.array(
            [step_idx / n_tgt], dtype=np.float32,
        )

        # 8. realized_returns (N_tgt) — close returns of realized path so far (dynamic)
        realized = entry.future_ohlcv[realized_idx].numpy()  # (N_tgt, patch_len, 5)
        closes = realized[:, -1, 3]  # (N_tgt,) close at end of each patch
        realized_returns = np.zeros(n_tgt, dtype=np.float32)
        for s in range(min(step_idx, n_tgt - 1)):
            realized_returns[s] = (
                (closes[s + 1] - closes[s]) / (abs(closes[s]) + EPS)
            )

        # Soft noise gate: differentiable sigmoid attenuation instead of a hard
        # Heaviside step. A hard if creates a non-differentiable discontinuity
        # in the observation manifold (Gradient Starvation risk).
        # sigmoid((vol - threshold) * T) → 1 on live markets, → 0 on dead markets,
        # with a smooth, Lipschitz-continuous transition around the threshold.
        close_mean = abs(entry.revin_means[0, 3].item())
        close_std = revin_stds[3]
        relative_vol = close_std / (close_mean + EPS)
        temperature = 100_000.0
        gate = float(
            1.0 / (1.0 + np.exp(
                -(relative_vol - self.config.dead_market_threshold) * temperature
            ))
        )
        future_mean_t = future_mean_t * gate
        future_std_t = future_std_t * gate
        close_stats = close_stats * gate
        delta_mu = delta_mu * gate
        realized_returns = realized_returns * gate

        # 9. position (1) — current portfolio position (dynamic)
        pos = np.array([position], dtype=np.float32)

        # 10. cumulative pnl (1) — running PnL (dynamic)
        cpnl = np.array([cum_pnl], dtype=np.float32)

        obs = np.concatenate([
            h_x_pooled, future_mean_t, future_std_t,
            close_stats, revin_stds, delta_mu,
            step_progress, realized_returns,
            pos, cpnl,
        ]).astype(np.float32)

        # Replace any NaN/Inf with 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        return obs

    @staticmethod
    def _compute_delta_mu(entry: TrajectoryEntry, patch_len: int) -> np.ndarray:
        """Compute macro trend signal from context OHLCV.

        Computes the normalized difference between the mean close of the
        last 2 patches vs the previous 2 patches in the context window.
        This gives the agent the "slope" of the global trend.

        Args:
            entry: Trajectory entry with context_ohlcv.
            patch_len: Candles per patch (from config).

        Returns:
            (1,) array with normalized delta_mu.
        """
        context = entry.context_ohlcv.numpy()  # (T, 5)
        close = context[:, 3]  # (T,)
        T = len(close)

        if T < 4 * patch_len:
            return np.zeros(1, dtype=np.float32)

        # Mean close of last 2 patches
        recent = close[-(2 * patch_len):].mean()
        # Mean close of the 2 patches before that
        earlier = close[-(4 * patch_len):-(2 * patch_len)].mean()

        # Normalize by overall std to keep it O(1)
        sigma = close.std() + EPS
        delta_mu = (recent - earlier) / sigma

        return np.array([delta_mu], dtype=np.float32)

    @staticmethod
    def _compute_close_stats(future_ohlcv: np.ndarray) -> np.ndarray:
        """Compute per-target close return statistics across N futures.

        Args:
            future_ohlcv: (N, N_tgt, patch_len, 5)

        Returns:
            (N_tgt * 3,) array: [mean, std, skew] per target.
        """
        N, N_tgt, patch_len, _ = future_ohlcv.shape

        # Close channel = 3, last candle of each patch
        close_prices = future_ohlcv[:, :, -1, 3]  # (N, N_tgt)

        # Returns: ratio of close at target t vs target t-1
        close_shifted = np.concatenate([
            close_prices[:, :1],  # anchor
            close_prices[:, :-1],
        ], axis=1)  # (N, N_tgt)
        returns = (close_prices - close_shifted) / (np.abs(close_shifted) + EPS)

        stats = []
        for t in range(N_tgt):
            r = returns[:, t]  # (N,)
            mean = r.mean()
            std = r.std() + EPS
            skew = ((r - mean) ** 3).mean() / (std ** 3)
            stats.extend([mean, std, skew])

        return np.array(stats, dtype=np.float32)
