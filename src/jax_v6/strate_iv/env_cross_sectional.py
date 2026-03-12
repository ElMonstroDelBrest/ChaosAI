"""Cross-Sectional Market-Neutral Portfolio Environment — JAX-pure.

Long top-k% assets, Short bottom-k%, rest Flat.
The Q-network scores each asset independently (same as single-asset DQN),
then ranking forces market neutrality by construction.

Actions are not chosen per-asset — they emerge from relative ranking.
The Q-net doesn't know it's in a portfolio: it sees (K, obs_dim) and
outputs (K, 3, n_quantiles). Scores = CVaR(Long) - CVaR(Short) per asset.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class PortfolioState(NamedTuple):
    """Mutable portfolio state (carried through episode)."""
    step: jnp.ndarray            # () int32
    weights: jnp.ndarray         # (K,) float32 — sum(|w|) ≈ 1
    prev_weights: jnp.ndarray    # (K,) float32
    cpnl: jnp.ndarray            # () float32 — portfolio cumulative PnL
    per_asset_cpnl: jnp.ndarray  # (K,) float32


def obs_dim(d_model: int) -> int:
    """Observation dim per asset: h_t + h_prev + vol + position + cpnl."""
    return 2 * d_model + 3


def make_portfolio_obs(
    h_t: jnp.ndarray,
    h_prev: jnp.ndarray,
    vol_t: jnp.ndarray,
    weights: jnp.ndarray,
    cpnl: jnp.ndarray,
) -> jnp.ndarray:
    """Construct observation matrix for K assets.

    Each row is identical format to single-asset env — the Q-net
    doesn't know it's in a portfolio.

    Args:
        h_t: (K, d_model) current JEPA embeddings
        h_prev: (K, d_model) previous JEPA embeddings
        vol_t: (K,) volatility proxies
        weights: (K,) float32 portfolio weights (soft or hard)
        cpnl: (K,) float32 per-asset cumulative PnL

    Returns:
        (K, 2*d_model + 3) observation matrix
    """
    K = h_t.shape[0]
    extra = jnp.stack([vol_t, weights, cpnl], axis=-1)  # (K, 3)
    return jnp.concatenate([h_t, h_prev, extra], axis=-1)  # (K, 2*d_model+3)


def compute_scores(
    q_net,
    q_params: dict,
    obs_K: jnp.ndarray,
    cvar_alpha: float = 0.25,
) -> jnp.ndarray:
    """Score K assets using CVaR(Long) - CVaR(Short).

    The Q-net sees obs_K as a batch of K independent observations.
    Score > 0 → prefer Long, Score < 0 → prefer Short.

    Args:
        q_net: QNetwork module (obs → (B, 3, n_quantiles))
        q_params: network parameters
        obs_K: (K, obs_dim) per-asset observations
        cvar_alpha: CVaR tail fraction

    Returns:
        (K,) directional scores
    """
    from .critic import cvar_from_quantiles

    # Forward: (K, 3, n_quantiles)
    q_vals = q_net.apply({"params": q_params}, obs_K)
    # CVaR per action per asset: (K, 3)
    cvar = jax.vmap(lambda q: cvar_from_quantiles(q, alpha=cvar_alpha))(q_vals)
    # Score = CVaR(Long=2) - CVaR(Short=0)
    return cvar[:, 2] - cvar[:, 0]


def rank_and_allocate(
    scores: jnp.ndarray,
    n_long: int,
    n_short: int,
) -> jnp.ndarray:
    """Deterministic ranking → position allocation.

    Top n_long assets → +1 (Long)
    Bottom n_short assets → -1 (Short)
    Rest → 0 (Flat)

    Market-neutral by construction: sum(positions) = n_long - n_short.
    When n_long == n_short, it's dollar-neutral.

    Args:
        scores: (K,) directional scores
        n_long: number of Long positions
        n_short: number of Short positions

    Returns:
        (K,) int32 positions ∈ {-1, 0, +1}
    """
    K = scores.shape[0]
    # Double argsort gives rank (0 = lowest score, K-1 = highest)
    ranks = jnp.argsort(jnp.argsort(scores))
    positions = jnp.zeros(K, dtype=jnp.int32)
    positions = jnp.where(ranks >= K - n_long, 1, positions)   # top → Long
    positions = jnp.where(ranks < n_short, -1, positions)      # bottom → Short
    return positions


def noisy_rank_and_allocate(
    scores: jnp.ndarray,
    key: jax.random.PRNGKey,
    noise_scale: float,
    n_long: int,
    n_short: int,
) -> jnp.ndarray:
    """Noisy ranking for exploration (Ape-X style).

    Adds Gaussian noise to scores before ranking.
    Preserves market neutrality (noise doesn't change n_long/n_short).

    Args:
        scores: (K,) directional scores
        key: PRNG key
        noise_scale: std of Gaussian noise on scores
        n_long: number of Long positions
        n_short: number of Short positions

    Returns:
        (K,) int32 positions ∈ {-1, 0, +1}
    """
    noise = noise_scale * jax.random.normal(key, scores.shape)
    noisy_scores = scores + noise
    return rank_and_allocate(noisy_scores, n_long, n_short)


def score_weighted_allocate(
    scores: jnp.ndarray,
    n_long: int,
    n_short: int,
) -> jnp.ndarray:
    """Score-weighted soft allocation.

    Same ranking as rank_and_allocate (top-k Long, bottom-k Short),
    but weights are proportional to |score| then L1-normalized.
    Market-neutral by construction (long weights + short weights cancel).

    Args:
        scores: (K,) directional scores
        n_long: number of Long positions
        n_short: number of Short positions

    Returns:
        (K,) float32 weights with sum(|w|) = 1
    """
    K = scores.shape[0]
    ranks = jnp.argsort(jnp.argsort(scores))
    direction = jnp.zeros(K, dtype=jnp.float32)
    direction = jnp.where(ranks >= K - n_long, 1.0, direction)   # top → +1
    direction = jnp.where(ranks < n_short, -1.0, direction)      # bottom → -1
    raw_w = jnp.abs(scores) * direction                          # conviction-weighted
    w_sum = jnp.sum(jnp.abs(raw_w)) + 1e-8
    return raw_w / w_sum


def noisy_score_weighted_allocate(
    scores: jnp.ndarray,
    key: jax.random.PRNGKey,
    noise_scale: float,
    n_long: int,
    n_short: int,
) -> jnp.ndarray:
    """Noisy score-weighted allocation for exploration (Ape-X style).

    Adds Gaussian noise to scores before soft allocation.
    Preserves market neutrality (noise doesn't change n_long/n_short).

    Args:
        scores: (K,) directional scores
        key: PRNG key
        noise_scale: std of Gaussian noise on scores
        n_long: number of Long positions
        n_short: number of Short positions

    Returns:
        (K,) float32 weights with sum(|w|) = 1
    """
    noise = noise_scale * jax.random.normal(key, scores.shape)
    noisy_scores = scores + noise
    return score_weighted_allocate(noisy_scores, n_long, n_short)


def reset_portfolio(
    h_wins: jnp.ndarray,
    vol_wins: jnp.ndarray,
    k_assets: int,
) -> tuple[jnp.ndarray, PortfolioState]:
    """Reset portfolio environment.

    Args:
        h_wins: (K, win_len, d_model) JEPA embeddings per asset
        vol_wins: (K, win_len) vol proxies per asset
        k_assets: number of assets K

    Returns:
        obs: (K, obs_dim) initial observations
        state: PortfolioState
    """
    K = k_assets
    state = PortfolioState(
        step=jnp.int32(0),
        weights=jnp.zeros(K, dtype=jnp.float32),
        prev_weights=jnp.zeros(K, dtype=jnp.float32),
        cpnl=jnp.float32(0.0),
        per_asset_cpnl=jnp.zeros(K, dtype=jnp.float32),
    )
    obs = make_portfolio_obs(
        h_wins[:, 0],    # (K, d_model)
        h_wins[:, 0],    # no previous at t=0
        vol_wins[:, 0],  # (K,)
        state.weights,
        state.per_asset_cpnl,
    )
    return obs, state


def step_portfolio(
    state: PortfolioState,
    weights: jnp.ndarray,
    h_wins: jnp.ndarray,
    vol_wins: jnp.ndarray,
    ret_wins: jnp.ndarray,
    episode_len: int,
    fee_rate: float = 0.0008,
    slippage_factor: float = 0.0,
    risk_parity: bool = False,
) -> tuple[jnp.ndarray, PortfolioState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Step the portfolio environment (V3 — soft allocation + quadratic slippage).

    Per-asset reward uses z-scored relative return (pure alpha signal):
        relative_ret_i = ret_i - mean(ret)
        z_ret_i = relative_ret_i / (std(ret) + eps)
        reward_i = w_i * z_ret_i - fee_rate * |delta_w_i| - slippage * delta_w_i²

    Quadratic slippage makes the agent "lazy" — penalizes large rebalances
    more than proportionally, encouraging smooth weight transitions.

    Args:
        state: current PortfolioState
        weights: (K,) float32 new weights (soft or hard)
        h_wins: (K, win_len, d_model) JEPA embeddings
        vol_wins: (K, win_len) vol proxies
        ret_wins: (K, win_len) returns
        episode_len: max steps per episode
        fee_rate: linear transaction cost coefficient
        slippage_factor: quadratic slippage coefficient
        risk_parity: if True, normalize each asset's return by its vol proxy before
            cross-sectional z-scoring. Makes the alpha signal regime-invariant:
            high-vol assets are de-levered so a BTC +3% day doesn't dominate the
            cross-sectional signal. vol_wins[:, t] (exo_clock RV proxy) is already
            in the buffer — zero extra cost.

    Returns:
        next_obs: (K, obs_dim)
        next_state: PortfolioState
        per_asset_reward: (K,) float32 — individual per-asset rewards for replay
        portfolio_reward: () float32 — mean reward for eval/logging
        done: () bool
    """
    t = state.step

    # Per-asset returns at current step
    next_returns = ret_wins[:, t + 1]   # (K,)

    # Risk-parity pre-normalization (v6.1).
    # Divides each asset's return by its vol proxy before cross-sectional z-scoring.
    # Effect: high-vol assets are de-levered; cross-sectional signal becomes
    # regime-invariant (same scale during crypto crash and quiet consolidation).
    # vol_wins[:, t] is the exo_clock RV proxy — already in buffer, zero extra cost.
    if risk_parity:
        vol_t = vol_wins[:, t] + 1e-8   # (K,) avoid div by zero
        next_returns = next_returns / vol_t

    # Z-scored cross-sectional relative return (pure alpha)
    mean_ret = next_returns.mean()
    std_ret = next_returns.std() + 1e-8
    z_relative_ret = (next_returns - mean_ret) / std_ret  # (K,)

    # Per-asset alpha reward (conviction-weighted)
    raw_alpha = weights * z_relative_ret  # (K,)

    # Transaction cost: linear + quadratic slippage
    delta = weights - state.weights       # (K,) float32
    fee = fee_rate * jnp.abs(delta) + slippage_factor * delta ** 2  # (K,)

    per_asset_reward = raw_alpha - fee  # (K,)

    # Portfolio reward for eval/logging
    portfolio_reward = per_asset_reward.mean()

    # Update state
    new_per_asset_cpnl = state.per_asset_cpnl + per_asset_reward
    new_cpnl = state.cpnl + portfolio_reward
    new_step = t + 1
    done = new_step >= episode_len

    new_state = PortfolioState(
        step=new_step,
        weights=weights,
        prev_weights=state.weights,
        cpnl=new_cpnl,
        per_asset_cpnl=new_per_asset_cpnl,
    )

    # Next observation
    next_h = h_wins[:, new_step]          # (K, d_model)
    prev_h = h_wins[:, new_step - 1]      # (K, d_model)  — safe: new_step >= 1
    next_vol = vol_wins[:, new_step]       # (K,)

    next_obs = make_portfolio_obs(next_h, prev_h, next_vol, weights, new_per_asset_cpnl)

    return next_obs, new_state, per_asset_reward, portfolio_reward, done
