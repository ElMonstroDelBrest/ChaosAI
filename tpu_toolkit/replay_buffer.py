"""Replay buffer ring avec double-buffering async H2D.

Problème: jnp.array(numpy) bloque le thread hôte ~100-500μs par appel.
Pour le RL sur TPU, sample() est appelé à chaque step → overhead cumulé.

Solution: jax.device_put() est asynchrone → retourne immédiatement un future.
Double-buffering: précharge le batch N+1 pendant que le TPU calcule N.

Latence H2D typique: ~50μs pour 1Mo (256×1000 float32).
Temps de calcul TPU: ~8s sur v6e, ~1s sur v5p.
→ L'overhead H2D est complètement masqué sur v5p (0.005% du temps total).
→ Utile aussi pour l'inférence rapide et les fine-tuning sur v5p.
"""

from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp


class ReplayBuffer:
    """Ring buffer numpy avec double-buffering H2D asynchrone.

    Storage: numpy (CPU, pas de limite de HBM).
    Output: dict de jnp.ndarray (DeviceArray futures).

    Usage typique:
        buf = ReplayBuffer(capacity=100_000, obs_dim=422, action_dim=1)
        for step in env_loop:
            buf.add(obs, action, reward, next_obs, done)
            if len(buf) >= warmup_steps:
                batch = buf.sample_async(256)  # non-bloquant
                metrics = agent.update(batch)  # bloque ici sur les valeurs
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int = 1) -> None:
        self.capacity = capacity
        self._ptr = 0
        self._size = 0
        self._obs      = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self._action   = np.zeros((capacity, action_dim), dtype=np.float32)
        self._reward   = np.zeros(capacity,               dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self._done     = np.zeros(capacity,               dtype=np.float32)
        self._prefetched: dict[str, jnp.ndarray] | None = None

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray | float,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self._obs[self._ptr]      = obs
        self._action[self._ptr]   = np.atleast_1d(action)
        self._reward[self._ptr]   = reward
        self._next_obs[self._ptr] = next_obs
        self._done[self._ptr]     = float(done)
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def _dispatch_async(self, batch_size: int) -> dict[str, jnp.ndarray]:
        """Lance un transfert H2D asynchrone, retourne immédiatement des futures."""
        idx = np.random.randint(0, self._size, size=batch_size)
        return {
            "obs":      jax.device_put(self._obs[idx]),
            "action":   jax.device_put(self._action[idx]),
            "reward":   jax.device_put(self._reward[idx]),
            "next_obs": jax.device_put(self._next_obs[idx]),
            "done":     jax.device_put(self._done[idx]),
        }

    def sample(self, batch_size: int) -> dict[str, jnp.ndarray]:
        """Sample synchrone (simple, pour debug)."""
        return self._dispatch_async(batch_size)

    def sample_async(self, batch_size: int) -> dict[str, jnp.ndarray]:
        """Sample avec double-buffering.

        Diagramme de recouvrement:
            Step N:   [TPU: calcule batch N    ] [Host: H2D batch N+1 en cours]
            Step N+1: [TPU: calcule batch N+1  ] [Host: H2D batch N+2 en cours]

        Première fois: dispatch 2 batches, retourne le premier.
        Appels suivants: retourne le prefetch, lance le suivant.
        """
        if self._prefetched is not None:
            batch = self._prefetched
            self._prefetched = self._dispatch_async(batch_size)
            return batch
        else:
            self._prefetched = self._dispatch_async(batch_size)
            return self._dispatch_async(batch_size)

    def __len__(self) -> int:
        return self._size

    def is_ready(self, warmup: int) -> bool:
        return self._size >= warmup
