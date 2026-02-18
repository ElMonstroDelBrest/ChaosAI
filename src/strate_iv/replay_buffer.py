"""Simple ring replay buffer for TD-MPC2 (Strate IV, Phase E)."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


class ReplayBuffer:
    """Ring buffer storing (obs, action, reward, next_obs, done) transitions.

    All data is stored as float32 numpy arrays on CPU and converted to tensors
    at sample time.

    Args:
        capacity: Maximum number of transitions to store.
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

    def sample(self, batch_size: int, device: str = "cpu") -> dict[str, Tensor]:
        idx = np.random.randint(0, self._size, size=batch_size)
        dev = torch.device(device)
        return {
            "obs":      torch.from_numpy(self._obs[idx]).to(dev),
            "action":   torch.from_numpy(self._action[idx]).to(dev),
            "reward":   torch.from_numpy(self._reward[idx]).to(dev),
            "next_obs": torch.from_numpy(self._next_obs[idx]).to(dev),
            "done":     torch.from_numpy(self._done[idx]).to(dev),
        }

    def __len__(self) -> int:
        return self._size
