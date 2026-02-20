"""Async ring replay buffer for TD-MPC2 — double-buffered H2D transfer.

Problem: jnp.array(numpy) is synchronous — the host thread blocks until the
Host-to-Device transfer completes, starving the TPU of work.

Solution: jax.device_put is asynchronous by default (returns a future).
Double-buffering prefetches the next batch while the TPU computes the current
one, hiding H2D latency entirely.

Typical TPU H2D bandwidth: ~20 GB/s. A 256×1000 float32 batch = ~1 MB → 50μs.
But synchronous transfer adds ~100μs of dispatch overhead per call × 5 arrays.
With async double-buffering, this overhead is completely hidden.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp

log = logging.getLogger(__name__)


class ReplayBuffer:
    """Async ring buffer: numpy storage, double-buffered JAX array output.

    All data stored as float32 numpy arrays on CPU. sample() dispatches
    asynchronous H2D transfers via jax.device_put. Call sample_async()
    to get prefetched batches with zero host-side blocking.

    Args:
        capacity: Maximum number of transitions.
        obs_dim: Observation dimension.
        action_dim: Action dimension (1 for continuous position).
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int = 1) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self._ptr = 0
        self._size = 0

        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._action = np.zeros((capacity, action_dim), dtype=np.float32)
        self._reward = np.zeros(capacity, dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._done = np.zeros(capacity, dtype=np.float32)

        # Double-buffer: prefetched batch ready for immediate consumption
        self._prefetched: dict[str, jnp.ndarray] | None = None

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

    def _dispatch_async(self, batch_size: int) -> dict[str, jnp.ndarray]:
        """Dispatch an asynchronous H2D transfer and return futures.

        jax.device_put returns immediately (non-blocking). The actual H2D
        transfer happens in the background. The returned arrays are futures
        that block only when their values are actually needed (e.g., in a
        JIT-compiled function).
        """
        idx = np.random.randint(0, self._size, size=batch_size)
        # jax.device_put is async by default — returns DeviceArray futures
        return {
            "obs":      jax.device_put(self._obs[idx]),
            "action":   jax.device_put(self._action[idx]),
            "reward":   jax.device_put(self._reward[idx]),
            "next_obs": jax.device_put(self._next_obs[idx]),
            "done":     jax.device_put(self._done[idx]),
        }

    def sample(self, batch_size: int) -> dict[str, jnp.ndarray]:
        """Sample a batch with async H2D transfer.

        Uses jax.device_put (async) instead of jnp.array (sync).
        """
        return self._dispatch_async(batch_size)

    def sample_async(self, batch_size: int) -> dict[str, jnp.ndarray]:
        """Double-buffered sample: return prefetched batch, dispatch next.

        First call: dispatches and returns (no prefetch available yet).
        Subsequent calls: returns the prefetched batch (already on device),
        and dispatches the next one in the background.

        This hides H2D latency by overlapping transfer with TPU compute:
          Step N:   TPU computes on batch N  |  Host transfers batch N+1
          Step N+1: TPU computes on batch N+1 | Host transfers batch N+2
        """
        if self._prefetched is not None:
            # Return prefetched batch, start next async transfer
            batch = self._prefetched
            self._prefetched = self._dispatch_async(batch_size)
            return batch
        else:
            # First call — no prefetch yet, dispatch two
            self._prefetched = self._dispatch_async(batch_size)
            return self._dispatch_async(batch_size)

    def __len__(self) -> int:
        return self._size
