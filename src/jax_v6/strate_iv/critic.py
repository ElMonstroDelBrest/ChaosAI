"""Distributional value critics for CVaR optimization — JAX/Flax port.

Quantile regression (QR-DQN style) to learn the full return distribution.
Two-critic ensemble (TD3-style) for pessimistic value estimation.
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
import flax.linen as nn

from .world_model import MLP, _build_dims


class QuantileCritic(nn.Module):
    """Single quantile critic: (z, a) → (n_quantiles,) return distribution."""

    latent_dim: int
    action_dim: int
    hidden_dim: int
    n_quantiles: int
    n_layers: int = 2

    @nn.compact
    def __call__(self, z: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        inp = jnp.concatenate([z, a], axis=-1)
        dims = _build_dims(
            self.latent_dim + self.action_dim,
            self.hidden_dim,
            self.n_quantiles,
            self.n_layers,
        )
        return MLP(features=dims[1:])(inp)  # (B, n_quantiles)


class EnsembleCritic(nn.Module):
    """Two-critic ensemble for pessimistic value estimation (TD3-style)."""

    latent_dim: int
    action_dim: int
    hidden_dim: int
    n_quantiles: int
    n_layers: int = 2

    def setup(self):
        kw = dict(
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            n_quantiles=self.n_quantiles,
            n_layers=self.n_layers,
        )
        self.q1 = QuantileCritic(**kw)
        self.q2 = QuantileCritic(**kw)

    def __call__(
        self, z: jnp.ndarray, a: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.q1(z, a), self.q2(z, a)

    def min(self, z: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        """Element-wise minimum of the two critics. Returns (B, n_quantiles)."""
        q1, q2 = self(z, a)
        return jnp.minimum(q1, q2)


def cvar_from_quantiles(quantiles: jnp.ndarray, alpha: float | jnp.ndarray = 0.1) -> jnp.ndarray:
    """CVaR_alpha from quantile estimates.

    CVaR_alpha = E[Z | Z <= Q_alpha(Z)] = mean of bottom alpha*100% quantiles.

    Args:
        quantiles: (*, n_quantiles) — unsorted quantile values.
        alpha: Confidence level (0.25 = worst-25% conditional expectation).
            Can be a scalar or a jnp.ndarray for dynamic alpha.

    Returns:
        (*,) CVaR values.
    """
    sorted_q = jnp.sort(quantiles, axis=-1)
    n_q = quantiles.shape[-1]
    k = jnp.maximum(jnp.int32(jnp.floor(alpha * n_q)), 1)
    # Create a mask for the bottom-k quantiles
    indices = jnp.arange(n_q)
    mask = (indices < k).astype(jnp.float32)
    return (sorted_q * mask).sum(axis=-1) / k


def quantile_huber_loss(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    taus: jnp.ndarray,
    kappa: float = 1.0,
) -> jnp.ndarray:
    """Asymmetric Quantile Huber loss for distributional RL (QR-DQN style).

    Args:
        pred:   (B, n_quantiles) predicted quantile values.
        target: (B, n_target_quantiles) TD target quantile values.
        taus:   (n_quantiles,) fixed quantile fractions in (0, 1).
        kappa:  Huber threshold.

    Returns:
        Scalar loss.
    """
    # (B, n_q, 1) vs (B, 1, n_tgt) → delta (B, n_q, n_tgt)
    delta = target[:, None, :] - pred[:, :, None]
    abs_delta = jnp.abs(delta)

    # Huber kernel
    huber = jnp.where(
        abs_delta <= kappa,
        0.5 * delta ** 2,
        kappa * (abs_delta - 0.5 * kappa),
    )

    # Asymmetric weight: |I(delta < 0) − tau_i|
    weight = jnp.abs((delta < 0).astype(jnp.float32) - taus[None, :, None])
    loss = (weight * huber).mean(axis=2).sum(axis=1).mean()
    return loss
