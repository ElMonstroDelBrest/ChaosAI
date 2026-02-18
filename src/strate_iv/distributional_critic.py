"""Distributional value critics for CVaR optimization (Strate IV, Phase E).

Uses quantile regression (QR-DQN style) to learn the full return distribution.
Two-critic ensemble (like TD3) for pessimistic value estimation.
"""

import torch
import torch.nn as nn
from torch import Tensor


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int) -> nn.Sequential:
    """MLP with ELU between layers, no activation after last."""
    dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class QuantileCritic(nn.Module):
    """Single quantile critic: (z, a) → (n_quantiles,) return distribution."""

    def __init__(
        self, latent_dim: int, action_dim: int, hidden_dim: int,
        n_quantiles: int, n_layers: int = 2,
    ):
        super().__init__()
        self.net = _mlp(latent_dim + action_dim, hidden_dim, n_quantiles, n_layers)

    def forward(self, z: Tensor, a: Tensor) -> Tensor:
        return self.net(torch.cat([z, a], dim=-1))  # (B, n_quantiles)


class EnsembleCritic(nn.Module):
    """Two-critic ensemble for pessimistic value estimation (like TD3)."""

    def __init__(
        self, latent_dim: int, action_dim: int, hidden_dim: int,
        n_quantiles: int, n_layers: int = 2,
    ):
        super().__init__()
        self.q1 = QuantileCritic(latent_dim, action_dim, hidden_dim, n_quantiles, n_layers)
        self.q2 = QuantileCritic(latent_dim, action_dim, hidden_dim, n_quantiles, n_layers)

    def forward(self, z: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
        return self.q1(z, a), self.q2(z, a)

    def min(self, z: Tensor, a: Tensor) -> Tensor:
        """Element-wise minimum of the two critics. Returns (B, n_quantiles)."""
        q1, q2 = self(z, a)
        return torch.min(q1, q2)


def cvar_from_quantiles(quantiles: Tensor, alpha: float = 0.1) -> Tensor:
    """CVaR_alpha from quantile estimates.

    CVaR_alpha = E[Z | Z <= Q_alpha(Z)] = mean of bottom alpha*100% quantiles.

    Args:
        quantiles: (*, n_quantiles) — unsorted quantile values.
        alpha: Confidence level (0.25 = worst-25% conditional expectation).

    Returns:
        (*,) CVaR values.
    """
    sorted_q = quantiles.sort(dim=-1).values
    k = max(1, int(alpha * quantiles.shape[-1]))
    return sorted_q[..., :k].mean(dim=-1)


def quantile_huber_loss(
    pred: Tensor,
    target: Tensor,
    taus: Tensor,
    kappa: float = 1.0,
) -> Tensor:
    """Asymmetric Quantile Huber loss for distributional RL (QR-DQN style).

    Args:
        pred:   (B, n_quantiles) predicted quantile values.
        target: (B, n_target_quantiles) TD target quantile values (detached).
        taus:   (n_quantiles,) fixed quantile fractions in (0, 1).
        kappa:  Huber threshold.

    Returns:
        Scalar loss.
    """
    # (B, n_q, 1) vs (B, 1, n_tgt) → delta (B, n_q, n_tgt)
    delta = target.unsqueeze(1) - pred.unsqueeze(2)
    abs_delta = delta.abs()

    # Huber kernel
    huber = torch.where(
        abs_delta <= kappa,
        0.5 * delta.pow(2),
        kappa * (abs_delta - 0.5 * kappa),
    )

    # Asymmetric weight: |I(delta < 0) − tau_i|
    weight = (delta.detach() < 0).float() - taus.view(1, -1, 1)
    loss = (weight.abs() * huber).mean(dim=2).sum(dim=1).mean()
    return loss
