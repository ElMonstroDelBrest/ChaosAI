"""All-Weather Trading Environment — JAX-pure, discrete actions.

3 actions: {0=Short, 1=Flat, 2=Long}
Observations: (h_t, h_{t-1}, vol_t, position, cpnl)
Reward: PnL - fees - risk penalty

Operates on pre-computed JEPA embeddings (h_last) from the RL buffer.
No Gymnasium dependency — pure JAX arrays for full JIT compatibility.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import NamedTuple


class EnvState(NamedTuple):
    """Mutable environment state (carried through episode)."""
    step: jnp.ndarray       # () int32 — current step in episode
    position: jnp.ndarray   # () int32 — current position (-1, 0, +1)
    prev_action: jnp.ndarray  # () int32 — previous action (0, 1, 2)
    cpnl: jnp.ndarray       # () float32 — cumulative PnL


# Action mapping: action_id → position
# 0 = Short → -1, 1 = Flat → 0, 2 = Long → +1
ACTION_TO_POSITION = jnp.array([-1, 0, 1], dtype=jnp.int32)

# Observation dim: h_t (d_model) + h_{t-1} (d_model) + vol (1) + position (1) + cpnl (1)
# = 2 * d_model + 3


def obs_dim(d_model: int) -> int:
    """Compute observation dimension from d_model."""
    return 2 * d_model + 3


def make_obs(
    h_t: jnp.ndarray,
    h_prev: jnp.ndarray,
    vol_t: jnp.ndarray,
    position: jnp.ndarray,
    cpnl: jnp.ndarray,
) -> jnp.ndarray:
    """Construct observation vector.

    Args:
        h_t: (d_model,) current JEPA embedding
        h_prev: (d_model,) previous JEPA embedding
        vol_t: () volatility proxy scalar
        position: () current position (-1, 0, +1)
        cpnl: () cumulative PnL

    Returns:
        (2*d_model + 3,) observation vector
    """
    return jnp.concatenate([
        h_t,
        h_prev,
        jnp.array([vol_t, position.astype(jnp.float32), cpnl]),
    ])


def reset(h_series: jnp.ndarray, vol_series: jnp.ndarray, start_idx: int) -> tuple[jnp.ndarray, EnvState]:
    """Reset environment at a given start index in the asset's time series.

    Args:
        h_series: (T, d_model) — full asset embedding series
        vol_series: (T,) — full asset vol proxy series
        start_idx: starting index in the series

    Returns:
        obs: (obs_dim,) initial observation
        state: EnvState
    """
    h_t = h_series[start_idx]
    h_prev = h_series[start_idx]  # no previous at t=0, use same
    vol_t = vol_series[start_idx]

    state = EnvState(
        step=jnp.int32(0),
        position=jnp.int32(0),     # start Flat
        prev_action=jnp.int32(1),  # Flat
        cpnl=jnp.float32(0.0),
    )

    observation = make_obs(h_t, h_prev, vol_t, state.position, state.cpnl)
    return observation, state


def compute_reward(
    action: jnp.ndarray,
    prev_action: jnp.ndarray,
    next_return: jnp.ndarray,
    vol_proxy: jnp.ndarray,
    fee_rate: float = 0.0008,
    risk_lambda: float = 0.5,
) -> jnp.ndarray:
    """Compute All-Weather reward.

    Args:
        action: () int32 — current action (0=Short, 1=Flat, 2=Long)
        prev_action: () int32 — previous action
        next_return: () float32 — next period log-return
        vol_proxy: () float32 — volatility proxy (z-scored)
        fee_rate: transaction cost per trade
        risk_lambda: risk penalty coefficient

    Returns:
        () float32 reward
    """
    position = (action - 1).astype(jnp.float32)  # -1, 0, +1
    raw_pnl = position * next_return
    fee = fee_rate * (action != prev_action).astype(jnp.float32)
    risk = jnp.abs(position) * risk_lambda * jnp.abs(vol_proxy)
    return raw_pnl - fee - risk


def step(
    state: EnvState,
    action: jnp.ndarray,
    h_series: jnp.ndarray,
    vol_series: jnp.ndarray,
    return_series: jnp.ndarray,
    start_idx: int,
    episode_len: int,
    fee_rate: float = 0.0008,
    risk_lambda: float = 0.5,
) -> tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray]:
    """Take one step in the environment.

    Args:
        state: current EnvState
        action: () int32 — chosen action
        h_series: (T, d_model) — asset embedding series
        vol_series: (T,) — asset vol proxy
        return_series: (T,) — asset returns
        start_idx: episode start index in series
        episode_len: max steps per episode
        fee_rate: transaction cost
        risk_lambda: risk penalty coefficient

    Returns:
        next_obs: (obs_dim,) observation
        next_state: EnvState
        reward: () float32
        done: () bool
    """
    t = state.step
    abs_t = start_idx + t

    # Compute reward using next return
    next_return = return_series[abs_t + 1]
    vol_t = vol_series[abs_t]
    reward = compute_reward(action, state.prev_action, next_return, vol_t, fee_rate, risk_lambda)

    # Update state
    new_position = ACTION_TO_POSITION[action]
    new_cpnl = state.cpnl + reward
    new_step = t + 1
    done = new_step >= episode_len

    new_state = EnvState(
        step=new_step,
        position=new_position,
        prev_action=action,
        cpnl=new_cpnl,
    )

    # Next observation
    next_abs_t = start_idx + new_step
    h_t = h_series[next_abs_t]
    h_prev = h_series[next_abs_t - 1]
    vol_next = vol_series[next_abs_t]

    next_obs = make_obs(h_t, h_prev, vol_next, new_state.position, new_cpnl)

    return next_obs, new_state, reward, done


def run_episode(
    h_series: jnp.ndarray,
    vol_series: jnp.ndarray,
    return_series: jnp.ndarray,
    start_idx: int,
    episode_len: int,
    select_action_fn,
    action_state,
    key: jax.random.PRNGKey,
    fee_rate: float = 0.0008,
    risk_lambda: float = 0.5,
) -> dict:
    """Run a full episode, collecting transitions.

    Args:
        h_series: (T, d_model) asset embeddings
        vol_series: (T,) vol proxy
        return_series: (T,) returns
        start_idx: starting index
        episode_len: number of steps
        select_action_fn: callable(action_state, obs, key) → (action, action_state)
        action_state: state for the action selector (e.g., Q-network params)
        key: PRNG key
        fee_rate: transaction cost
        risk_lambda: risk penalty weight

    Returns:
        dict with transitions: obs, action, reward, next_obs, done (all lists)
    """
    obs, env_state = reset(h_series, vol_series, start_idx)
    transitions = {"obs": [], "action": [], "reward": [], "next_obs": [], "done": []}

    for t in range(episode_len):
        key, action_key = jax.random.split(key)
        action, action_state = select_action_fn(action_state, obs, action_key)

        next_obs, env_state, reward, done = step(
            env_state, action, h_series, vol_series, return_series,
            start_idx, episode_len, fee_rate, risk_lambda,
        )

        transitions["obs"].append(obs)
        transitions["action"].append(action)
        transitions["reward"].append(reward)
        transitions["next_obs"].append(next_obs)
        transitions["done"].append(done)

        obs = next_obs

    return transitions
