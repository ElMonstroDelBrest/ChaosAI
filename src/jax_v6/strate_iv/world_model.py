"""Latent world model for TD-MPC2 planning — JAX/Flax port.

Architecture:
  LatentEncoder:  obs → z  (MLP + LayerNorm)
  LatentDynamics: (z, a) → z_next  (MLP + residual skip + LayerNorm)
  RewardHead:     (z, a) → r  (scalar)
  WorldModel:     composes the three, rollout() via jax.lax.scan
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn


def _build_dims(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int) -> list[int]:
    return [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]


class MLP(nn.Module):
    """MLP with ELU activations between every layer except the last."""

    features: Sequence[int]

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat)(x)
            if i < len(self.features) - 1:
                x = nn.elu(x)
        return x


class LatentEncoder(nn.Module):
    """obs → z with LayerNorm."""

    hidden_dim: int
    latent_dim: int
    n_layers: int = 2

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        dims = _build_dims(obs.shape[-1], self.hidden_dim, self.latent_dim, self.n_layers)
        x = MLP(features=dims[1:])(obs)
        return nn.LayerNorm()(x)


class LatentDynamics(nn.Module):
    """(z, a) → z_next with residual skip + LayerNorm."""

    hidden_dim: int
    latent_dim: int
    action_dim: int
    n_layers: int = 2

    @nn.compact
    def __call__(self, z: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        inp = jnp.concatenate([z, a], axis=-1)
        dims = _build_dims(
            self.latent_dim + self.action_dim,
            self.hidden_dim,
            self.latent_dim,
            self.n_layers,
        )
        residual = MLP(features=dims[1:])(inp)
        return nn.LayerNorm()(residual + z)


class RewardHead(nn.Module):
    """(z, a) → scalar reward."""

    hidden_dim: int
    latent_dim: int
    action_dim: int
    n_layers: int = 2

    @nn.compact
    def __call__(self, z: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        inp = jnp.concatenate([z, a], axis=-1)
        dims = _build_dims(
            self.latent_dim + self.action_dim,
            self.hidden_dim,
            1,
            self.n_layers,
        )
        return MLP(features=dims[1:])(inp).squeeze(-1)


class WorldModel(nn.Module):
    """Latent world model: encoder + residual dynamics + reward head.

    Uses jax.lax.scan for efficient H-step rollouts on TPU.
    """

    obs_dim: int
    action_dim: int
    latent_dim: int
    hidden_dim: int
    n_layers: int = 2

    def setup(self):
        self.encoder = LatentEncoder(
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            n_layers=self.n_layers,
        )
        self.dynamics = LatentDynamics(
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            n_layers=self.n_layers,
        )
        self.reward_head = RewardHead(
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            n_layers=self.n_layers,
        )

    def encode(self, obs: jnp.ndarray) -> jnp.ndarray:
        return self.encoder(obs)

    def step(self, z: jnp.ndarray, a: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Single latent step. Returns (z_next, r)."""
        return self.dynamics(z, a), self.reward_head(z, a)

    def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Encode obs then rollout H steps.

        Args:
            obs: (B, obs_dim)
            actions: (H, B, action_dim)

        Returns:
            z_seq: (H, B, latent_dim)
            r_seq: (H, B)
        """
        z0 = self.encode(obs)
        return self.rollout(z0, actions)

    def rollout(self, z0: jnp.ndarray, actions: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """H-step imagined rollout via jax.lax.scan.

        Args:
            z0: (B, latent_dim) initial latent state.
            actions: (H, B, action_dim) action sequence.

        Returns:
            z_seq: (H, B, latent_dim)
            r_seq: (H, B)
        """
        def scan_fn(z, a):
            z_next, r = self.step(z, a)
            return z_next, (z_next, r)

        _, (z_seq, r_seq) = jax.lax.scan(scan_fn, z0, actions)
        return z_seq, r_seq
