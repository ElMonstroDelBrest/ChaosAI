"""TD-MPC2 agent with CVaR objective for Strate IV (Phase E).

Architecture:
  - WorldModel:      latent encoder + residual dynamics + reward head
  - EnsembleCritic:  distributional quantile critics (pessimistic ensemble)
  - Actor:           deterministic policy π(z) → action ∈ [-1, 1]
  - Target critic:   EMA copy of critic for stable Bellman targets

Planning:
  MPPI (Model Predictive Path Integral) using world model + CVaR critic scoring.
  Each planning call refines a Gaussian action sequence toward the CVaR-optimal trajectory.

Training (online, from ReplayBuffer):
  1. Consistency:  ||dynamics(z, a) − encode(next_obs)||²
  2. Reward:       ||reward_head(z, a) − r||²
  3. Critic:       QR-Huber TD loss with distributional Bellman targets
  4. Actor:        maximize CVaR_alpha of ensemble critic
"""

from __future__ import annotations

import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .world_model import WorldModel
from .distributional_critic import EnsembleCritic, cvar_from_quantiles, quantile_huber_loss


class Actor(nn.Module):
    """Deterministic policy: z → action ∈ [-1, 1] (tanh output)."""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int, n_layers: int = 2):
        super().__init__()
        dims = [latent_dim] + [hidden_dim] * (n_layers - 1) + [action_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ELU())
        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        return torch.tanh(self.net(z))


class TDMPC2Agent:
    """TD-MPC2 with distributional CVaR critic for risk-aware latent planning.

    Not an nn.Module — manages multiple sub-networks and their optimizers.

    Args:
        config: TDMPC2Config instance.
        obs_dim: Observation space dimension (auto-detected from buffer).
        action_dim: Action space dimension (1 for continuous position).
        device: 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        config,
        obs_dim: int,
        action_dim: int = 1,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device)

        self.world_model = WorldModel(
            obs_dim, action_dim, config.latent_dim, config.hidden_dim, config.n_layers,
        ).to(self.device)

        self.actor = Actor(
            config.latent_dim, action_dim, config.hidden_dim, config.n_layers,
        ).to(self.device)

        self.critic = EnsembleCritic(
            config.latent_dim, action_dim, config.hidden_dim, config.n_quantiles, config.n_layers,
        ).to(self.device)

        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad = False

        # Fixed quantile fractions (midpoints of n_quantiles equal-width intervals)
        self.taus = (
            (torch.arange(config.n_quantiles, dtype=torch.float32) + 0.5)
            / config.n_quantiles
        ).to(self.device)

        self.wm_optim = torch.optim.Adam(self.world_model.parameters(), lr=config.lr)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=config.lr)

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _mppi(self, z: Tensor) -> Tensor:
        """MPPI planning from a single latent state.

        Args:
            z: (latent_dim,) current latent state.

        Returns:
            (action_dim,) first action of the CVaR-optimal sequence.
        """
        cfg = self.config
        H, K = cfg.plan_horizon, cfg.plan_samples
        gamma = cfg.gamma

        # Warm-start: actor action repeated for H steps
        mu = self.actor(z.unsqueeze(0)).expand(H, -1).clone()   # (H, action_dim)
        sigma = torch.ones_like(mu) * cfg.plan_init_std

        for _ in range(cfg.plan_iters):
            # Sample K action sequences
            eps = torch.randn(H, K, self.action_dim, device=self.device)
            actions = (mu.unsqueeze(1) + sigma.unsqueeze(1) * eps).clamp(-1, 1)  # (H, K, act)

            # Imagined rollout
            z0 = z.unsqueeze(0).expand(K, -1)        # (K, latent_dim)
            z_seq, r_seq = self.world_model.rollout(z0, actions)  # (H,K,D), (H,K)

            # Discounted returns (no terminal bootstrapping first)
            discount = gamma ** torch.arange(H, device=self.device, dtype=torch.float32)
            returns = (r_seq * discount.unsqueeze(1)).sum(0)  # (K,)

            # Terminal value via CVaR of target critic
            z_T = z_seq[-1]                               # (K, latent_dim)
            a_T = self.actor(z_T)                         # (K, action_dim)
            q_T = self.target_critic.min(z_T, a_T)        # (K, n_quantiles)
            terminal_cvar = cvar_from_quantiles(q_T, cfg.cvar_alpha)  # (K,)
            returns = returns + (gamma ** H) * terminal_cvar

            # MPPI weights: softmax re-weighting
            w = torch.softmax(returns / cfg.plan_temperature, dim=0)  # (K,)

            # Update Gaussian parameters
            mu = (w.view(1, K, 1) * actions).sum(1)       # (H, action_dim)
            sigma = (w.view(1, K, 1) * (actions - mu.unsqueeze(1)).pow(2)).sum(1).sqrt()
            sigma = sigma.clamp(min=1e-3)

        return mu[0].clamp(-1, 1)  # (action_dim,)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, eval: bool = False) -> np.ndarray:
        """Select action from a raw observation.

        Args:
            obs: (obs_dim,) observation from the gym environment.
            eval: If True, use deterministic actor without MPPI.

        Returns:
            (action_dim,) action as numpy array.
        """
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().to(self.device)
            z = self.world_model.encode(obs_t.unsqueeze(0)).squeeze(0)
            if eval or not self.config.use_planning:
                action = self.actor(z.unsqueeze(0)).squeeze(0)
            else:
                action = self._mppi(z)
        return action.cpu().numpy()

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def update(self, batch: dict[str, Tensor]) -> dict[str, float]:
        """Joint update of world model, critic, and actor from one replay batch.

        Args:
            batch: Dict with keys obs, action, reward, next_obs, done.

        Returns:
            Dict of scalar losses for logging.
        """
        obs = batch["obs"].to(self.device)
        action = batch["action"].to(self.device)
        reward = batch["reward"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        done = batch["done"].to(self.device)

        # ---- 1. World model ----
        z = self.world_model.encode(obs)
        with torch.no_grad():
            z_next = self.world_model.encode(next_obs)

        z_next_pred = self.world_model.dynamics(z, action)
        r_pred = self.world_model.reward_head(z, action)

        consistency_loss = F.mse_loss(z_next_pred, z_next)
        reward_loss = F.mse_loss(r_pred, reward)
        wm_loss = consistency_loss + reward_loss

        self.wm_optim.zero_grad()
        wm_loss.backward()
        nn.utils.clip_grad_norm_(self.world_model.parameters(), self.config.max_grad_norm)
        self.wm_optim.step()

        # ---- 2. Critic ----
        with torch.no_grad():
            a_next = self.actor(z_next)
            q_next = self.target_critic.min(z_next, a_next)  # (B, n_quantiles)
            # Distributional Bellman target: shift entire distribution by r + γ * (1-done)
            target_q = reward.unsqueeze(1) + self.config.gamma * (1 - done).unsqueeze(1) * q_next

        q1, q2 = self.critic(z.detach(), action)
        critic_loss = (
            quantile_huber_loss(q1, target_q, self.taus)
            + quantile_huber_loss(q2, target_q, self.taus)
        )

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_optim.step()

        # ---- 3. Actor: maximize CVaR of critic ----
        a_pred = self.actor(z.detach())
        q_actor = self.critic.min(z.detach(), a_pred)  # (B, n_quantiles)
        cvar_val = cvar_from_quantiles(q_actor, self.config.cvar_alpha)  # (B,)
        actor_loss = -cvar_val.mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_optim.step()

        # ---- 4. EMA update target critic ----
        tau = self.config.ema_tau
        with torch.no_grad():
            for p_tgt, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                p_tgt.data.mul_(tau).add_(p.data, alpha=1.0 - tau)

        return {
            "loss/consistency": consistency_loss.item(),
            "loss/reward": reward_loss.item(),
            "loss/critic": critic_loss.item(),
            "loss/actor": actor_loss.item(),
            "metrics/cvar": cvar_val.mean().item(),
        }

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.world_model.state_dict(), f"{path}/world_model.pt")
        torch.save(self.actor.state_dict(), f"{path}/actor.pt")
        torch.save(self.critic.state_dict(), f"{path}/critic.pt")
        torch.save(self.target_critic.state_dict(), f"{path}/target_critic.pt")

    def load(self, path: str) -> None:
        dev = self.device
        self.world_model.load_state_dict(torch.load(f"{path}/world_model.pt", map_location=dev))
        self.actor.load_state_dict(torch.load(f"{path}/actor.pt", map_location=dev))
        self.critic.load_state_dict(torch.load(f"{path}/critic.pt", map_location=dev))
        self.target_critic.load_state_dict(torch.load(f"{path}/target_critic.pt", map_location=dev))
