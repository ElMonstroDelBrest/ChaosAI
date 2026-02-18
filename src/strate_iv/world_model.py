"""Latent world model for TD-MPC2 planning in Strate IV (Phase E).

Architecture:
  LatentEncoder:  obs → z  (obs_dim → latent_dim, + LayerNorm)
  LatentDynamics: (z, a) → z_next  (+ residual skip + LayerNorm)
  RewardHead:     (z, a) → r  (scalar)
  WorldModel:     composes the three above.
"""

import torch
import torch.nn as nn
from torch import Tensor


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int) -> nn.Sequential:
    """Build MLP with ELU activations between every layer EXCEPT the last."""
    dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class LatentEncoder(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, latent_dim: int, n_layers: int = 2):
        super().__init__()
        self.net = _mlp(obs_dim, hidden_dim, latent_dim, n_layers)
        self.ln = nn.LayerNorm(latent_dim)

    def forward(self, obs: Tensor) -> Tensor:
        return self.ln(self.net(obs))


class LatentDynamics(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int, n_layers: int = 2):
        super().__init__()
        self.net = _mlp(latent_dim + action_dim, hidden_dim, latent_dim, n_layers)
        self.ln = nn.LayerNorm(latent_dim)

    def forward(self, z: Tensor, a: Tensor) -> Tensor:
        inp = torch.cat([z, a], dim=-1)
        # Residual skip: helps long-horizon rollouts stay on manifold
        return self.ln(self.net(inp) + z)


class RewardHead(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int, n_layers: int = 2):
        super().__init__()
        self.net = _mlp(latent_dim + action_dim, hidden_dim, 1, n_layers)

    def forward(self, z: Tensor, a: Tensor) -> Tensor:
        return self.net(torch.cat([z, a], dim=-1)).squeeze(-1)  # (B,)


class WorldModel(nn.Module):
    """Latent world model: encoder + dynamics (residual) + reward head.

    Args:
        obs_dim: Raw observation dimension.
        action_dim: Action dimension.
        latent_dim: Planning latent dimension.
        hidden_dim: Hidden width for all sub-networks.
        n_layers: Depth of all sub-networks.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
    ):
        super().__init__()
        self.encoder = LatentEncoder(obs_dim, hidden_dim, latent_dim, n_layers)
        self.dynamics = LatentDynamics(latent_dim, action_dim, hidden_dim, n_layers)
        self.reward_head = RewardHead(latent_dim, action_dim, hidden_dim, n_layers)

    def encode(self, obs: Tensor) -> Tensor:
        return self.encoder(obs)

    def step(self, z: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
        """Single latent step. Returns (z_next, r)."""
        return self.dynamics(z, a), self.reward_head(z, a)

    def rollout(self, z0: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        """H-step imagined rollout.

        Args:
            z0: (B, latent_dim) initial latent state.
            actions: (H, B, action_dim) action sequence.

        Returns:
            z_seq: (H, B, latent_dim)
            r_seq: (H, B)
        """
        H = actions.shape[0]
        z = z0
        z_list: list[Tensor] = []
        r_list: list[Tensor] = []
        for h in range(H):
            z_next, r = self.step(z, actions[h])
            z_list.append(z_next)
            r_list.append(r)
            z = z_next
        return torch.stack(z_list), torch.stack(r_list)
