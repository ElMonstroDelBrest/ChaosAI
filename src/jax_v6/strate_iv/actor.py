"""Deterministic actor for TD-MPC2 — JAX/Flax port.

z → tanh(MLP(z)) → action ∈ [-1, 1]
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
import flax.linen as nn

from .world_model import MLP, _build_dims


class Actor(nn.Module):
    """Deterministic policy: z → action ∈ [-1, 1] (tanh output)."""

    latent_dim: int
    action_dim: int
    hidden_dim: int
    n_layers: int = 2

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        dims = _build_dims(self.latent_dim, self.hidden_dim, self.action_dim, self.n_layers)
        return jnp.tanh(MLP(features=dims[1:])(z))
