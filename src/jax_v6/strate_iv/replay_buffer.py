"""Ring replay buffer for TD-MPC2 — numpy storage, JAX array output."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp


class ReplayBuffer:
    """Ring buffer storing (obs, action, reward, next_obs, done) transitions.

    All data stored as float32 numpy arrays on CPU. sample() returns
    dict[str, jnp.ndarray] for direct use in JAX JIT-compiled updates.

    Args:
        capacity: Maximum number of transitions.
        obs_dim: Observation dimension.
        action_dim: Action dimension (1 for continuous position).
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int = 1) -> None:
        self.capacity = capacity
        self._ptr = 0
        self._size = 0

        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._action = np.zeros((capacity, action_dim), dtype=np.float32)
        self._reward = np.zeros(capacity, dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._done = np.zeros(capacity, dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray | float,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self._obs[self._ptr] = obs
        self._action[self._ptr] = np.atleast_1d(action)
        self._reward[self._ptr] = reward
        self._next_obs[self._ptr] = next_obs
        self._done[self._ptr] = float(done)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, jnp.ndarray]:
        """Sample a random batch and return as JAX arrays."""
        idx = np.random.randint(0, self._size, size=batch_size)
        return {
            "obs":      jnp.array(self._obs[idx]),
            "action":   jnp.array(self._action[idx]),
            "reward":   jnp.array(self._reward[idx]),
            "next_obs": jnp.array(self._next_obs[idx]),
            "done":     jnp.array(self._done[idx]),
        }

    def __len__(self) -> int:
        return self._size
