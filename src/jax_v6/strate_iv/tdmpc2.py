"""TD-MPC2 agent with dynamic CVaR + Multiverse Crossing — full JAX/Flax port.

Architecture:
  - WorldModel:      latent encoder + residual dynamics + reward head (Flax)
  - EnsembleCritic:  distributional quantile critics (Flax)
  - Actor:           deterministic policy (Flax)
  - Target critic:   EMA copy for stable Bellman targets

Planning:
  MPPI via jax.lax.scan (rollout) + jax.vmap (K samples).
  CVaR alpha dynamically modulated by multiverse convergence score.

Training:
  JIT-compiled update step with optax.adam for all 3 sub-networks.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from .world_model import WorldModel
from .critic import EnsembleCritic, cvar_from_quantiles, quantile_huber_loss
from .actor import Actor
from .multiverse_crossing import dynamic_cvar_alpha

PyTree = Any


def _ema_update(target_params: PyTree, online_params: PyTree, tau: float) -> PyTree:
    """EMA: target = tau * target + (1 - tau) * online."""
    return jax.tree.map(
        lambda t, o: tau * t + (1.0 - tau) * o,
        target_params,
        online_params,
    )


class TDMPC2Agent:
    """TD-MPC2 with distributional dynamic-CVaR critic for risk-aware latent planning.

    Pure JAX/Flax — no PyTorch. All computation is JIT-compiled.

    Args:
        config: StrateIVJAXConfig instance.
        obs_dim: Observation space dimension.
        action_dim: Action space dimension (1 for continuous position).
        rng_key: JAX PRNG key.
    """

    def __init__(
        self,
        config,
        obs_dim: int,
        action_dim: int = 1,
        rng_key: jax.random.PRNGKey | None = None,
    ) -> None:
        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        # --- Initialize Flax modules ---
        k1, k2, k3, self._rng = jax.random.split(rng_key, 4)

        self.world_model = WorldModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
        )
        self.actor_module = Actor(
            latent_dim=config.latent_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
        )
        self.critic_module = EnsembleCritic(
            latent_dim=config.latent_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            n_quantiles=config.n_quantiles,
            n_layers=config.n_layers,
        )

        # Init params with dummy inputs
        dummy_obs = jnp.zeros((1, obs_dim))
        dummy_action = jnp.zeros((1, action_dim))
        dummy_z = jnp.zeros((1, config.latent_dim))

        self.wm_params = self.world_model.init(k1, dummy_obs, jnp.zeros((1, 1, action_dim)))
        self.actor_params = self.actor_module.init(k2, dummy_z)
        self.critic_params = self.critic_module.init(k3, dummy_z, dummy_action)
        self.target_critic_params = jax.tree.map(jnp.copy, self.critic_params)

        # Fixed quantile fractions (midpoints)
        n_q = config.n_quantiles
        self.taus = (jnp.arange(n_q, dtype=jnp.float32) + 0.5) / n_q

        # Optimizers
        self.wm_opt = optax.adam(config.lr)
        self.actor_opt = optax.adam(config.lr)
        self.critic_opt = optax.adam(config.lr)

        self.wm_opt_state = self.wm_opt.init(self.wm_params)
        self.actor_opt_state = self.actor_opt.init(self.actor_params)
        self.critic_opt_state = self.critic_opt.init(self.critic_params)

    # ------------------------------------------------------------------
    # Planning (MPPI)
    # ------------------------------------------------------------------

    def _mppi(
        self,
        wm_params: PyTree,
        actor_params: PyTree,
        critic_params: PyTree,
        z: jnp.ndarray,
        convergence_score: float,
        rng: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """MPPI planning from a single latent state with dynamic CVaR.

        Args:
            wm_params: WorldModel parameters.
            actor_params: Actor parameters.
            critic_params: Target critic parameters.
            z: (latent_dim,) current latent state.
            convergence_score: Multiverse convergence ∈ [0, 1].
            rng: PRNG key.

        Returns:
            (action_dim,) first action of the CVaR-optimal sequence.
        """
        cfg = self.config
        H, K = cfg.plan_horizon, cfg.plan_samples
        gamma = cfg.gamma

        # Dynamic CVaR alpha based on multiverse convergence
        alpha = dynamic_cvar_alpha(
            jnp.float32(convergence_score), cfg.cvar_alpha_min, cfg.cvar_alpha_max,
        )

        # Warm-start: actor action repeated for H steps
        a0 = self.actor_module.apply(actor_params, z[None, :]).squeeze(0)  # (action_dim,)
        mu = jnp.broadcast_to(a0, (H, self.action_dim))
        sigma = jnp.ones((H, self.action_dim)) * cfg.plan_init_std

        def mppi_iter(carry, _):
            mu, sigma, rng = carry
            rng, sub = jax.random.split(rng)

            # Sample K action sequences: (H, K, action_dim)
            eps = jax.random.normal(sub, shape=(H, K, self.action_dim))
            actions = jnp.clip(mu[:, None, :] + sigma[:, None, :] * eps, -1.0, 1.0)

            # Imagined rollout for all K samples
            z0 = jnp.broadcast_to(z[None, :], (K, cfg.latent_dim))
            z_seq, r_seq = self.world_model.apply(wm_params, z0, actions, method=WorldModel.rollout)

            # Discounted returns
            discount = gamma ** jnp.arange(H, dtype=jnp.float32)
            returns = (r_seq * discount[:, None]).sum(axis=0)  # (K,)

            # Terminal value via CVaR of target critic
            z_T = z_seq[-1]  # (K, latent_dim)
            a_T = self.actor_module.apply(actor_params, z_T)  # (K, action_dim)
            q_T = self.critic_module.apply(critic_params, z_T, a_T, method=EnsembleCritic.min)
            terminal_cvar = cvar_from_quantiles(q_T, alpha)  # (K,)
            returns = returns + (gamma ** H) * terminal_cvar

            # MPPI softmax re-weighting
            w = jax.nn.softmax(returns / cfg.plan_temperature)  # (K,)

            mu_new = (w[None, :, None] * actions).sum(axis=1)  # (H, action_dim)
            sigma_new = jnp.sqrt(
                (w[None, :, None] * (actions - mu_new[:, None, :]) ** 2).sum(axis=1)
            )
            sigma_new = jnp.maximum(sigma_new, 1e-3)

            return (mu_new, sigma_new, rng), None

        (mu, sigma, _), _ = jax.lax.scan(
            mppi_iter, (mu, sigma, rng), None, length=cfg.plan_iters,
        )

        return jnp.clip(mu[0], -1.0, 1.0)  # first action

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        obs: jnp.ndarray,
        convergence_score: float = 0.5,
        eval_mode: bool = False,
    ) -> jnp.ndarray:
        """Select action from observation.

        Args:
            obs: (obs_dim,) observation.
            convergence_score: Multiverse convergence from env.
            eval_mode: If True, use deterministic actor without MPPI.

        Returns:
            (action_dim,) action as jnp array.
        """
        z = self.world_model.apply(
            self.wm_params, obs[None, :], method=WorldModel.encode,
        ).squeeze(0)

        if eval_mode or not self.config.use_planning:
            return self.actor_module.apply(self.actor_params, z[None, :]).squeeze(0)

        self._rng, sub = jax.random.split(self._rng)
        return self._mppi(
            self.wm_params,
            self.actor_params,
            self.target_critic_params,
            z,
            convergence_score,
            sub,
        )

    # ------------------------------------------------------------------
    # Training step (JIT-compiled)
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def _update_jit(
        self,
        wm_params: PyTree,
        actor_params: PyTree,
        critic_params: PyTree,
        target_critic_params: PyTree,
        wm_opt_state: PyTree,
        actor_opt_state: PyTree,
        critic_opt_state: PyTree,
        taus: jnp.ndarray,
        batch: dict[str, jnp.ndarray],
    ) -> tuple[PyTree, PyTree, PyTree, PyTree, PyTree, PyTree, PyTree, dict[str, jnp.ndarray]]:
        """JIT-compiled joint update of world model, critic, and actor."""
        cfg = self.config
        obs = batch["obs"]
        action = batch["action"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        # ---- 1. World model loss ----
        def wm_loss_fn(wm_p):
            z = self.world_model.apply(wm_p, obs, method=WorldModel.encode)
            z_next_target = jax.lax.stop_gradient(
                self.world_model.apply(wm_p, next_obs, method=WorldModel.encode)
            )
            z_next_pred = self.world_model.apply(
                wm_p, z, action, method=WorldModel.step,
            )[0]  # dynamics output
            r_pred = self.world_model.apply(
                wm_p, z, action, method=WorldModel.step,
            )[1]  # reward output
            consistency = jnp.mean((z_next_pred - z_next_target) ** 2)
            reward_loss = jnp.mean((r_pred - reward) ** 2)
            return consistency + reward_loss, (consistency, reward_loss, z)

        (wm_loss, (consistency_loss, reward_loss, z_detached)), wm_grads = jax.value_and_grad(
            wm_loss_fn, has_aux=True,
        )(wm_params)
        wm_grads = optax.clip_by_global_norm(cfg.max_grad_norm).update(wm_grads, None)[0]
        wm_updates, wm_opt_state = self.wm_opt.update(wm_grads, wm_opt_state, wm_params)
        wm_params = optax.apply_updates(wm_params, wm_updates)

        # Recompute z with updated params (for critic/actor)
        z = jax.lax.stop_gradient(
            self.world_model.apply(wm_params, obs, method=WorldModel.encode)
        )
        z_next = jax.lax.stop_gradient(
            self.world_model.apply(wm_params, next_obs, method=WorldModel.encode)
        )

        # ---- 2. Critic loss ----
        a_next = jax.lax.stop_gradient(
            self.actor_module.apply(actor_params, z_next)
        )
        q_next = self.critic_module.apply(
            target_critic_params, z_next, a_next, method=EnsembleCritic.min,
        )
        target_q = reward[:, None] + cfg.gamma * (1.0 - done[:, None]) * q_next

        def critic_loss_fn(c_p):
            q1, q2 = self.critic_module.apply(c_p, z, action)
            return (
                quantile_huber_loss(q1, target_q, taus)
                + quantile_huber_loss(q2, target_q, taus)
            )

        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(critic_params)
        critic_grads = optax.clip_by_global_norm(cfg.max_grad_norm).update(critic_grads, None)[0]
        c_updates, critic_opt_state = self.critic_opt.update(critic_grads, critic_opt_state, critic_params)
        critic_params = optax.apply_updates(critic_params, c_updates)

        # ---- 3. Actor loss: maximize CVaR ----
        # Use midpoint alpha for training (agent learns a mixed strategy)
        train_alpha = (cfg.cvar_alpha_min + cfg.cvar_alpha_max) / 2.0

        def actor_loss_fn(a_p):
            a_pred = self.actor_module.apply(a_p, z)
            q_actor = self.critic_module.apply(
                critic_params, z, a_pred, method=EnsembleCritic.min,
            )
            cvar_val = cvar_from_quantiles(q_actor, train_alpha)
            return -cvar_val.mean()

        actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(actor_params)
        actor_grads = optax.clip_by_global_norm(cfg.max_grad_norm).update(actor_grads, None)[0]
        a_updates, actor_opt_state = self.actor_opt.update(actor_grads, actor_opt_state, actor_params)
        actor_params = optax.apply_updates(actor_params, a_updates)

        # ---- 4. EMA update target critic ----
        target_critic_params = _ema_update(target_critic_params, critic_params, cfg.ema_tau)

        metrics = {
            "loss/consistency": consistency_loss,
            "loss/reward": reward_loss,
            "loss/critic": critic_loss,
            "loss/actor": actor_loss,
        }

        return (
            wm_params, actor_params, critic_params, target_critic_params,
            wm_opt_state, actor_opt_state, critic_opt_state,
            metrics,
        )

    def update(self, batch: dict[str, jnp.ndarray]) -> dict[str, float]:
        """Joint update from one replay batch. Delegates to JIT-compiled _update_jit."""
        (
            self.wm_params,
            self.actor_params,
            self.critic_params,
            self.target_critic_params,
            self.wm_opt_state,
            self.actor_opt_state,
            self.critic_opt_state,
            metrics,
        ) = self._update_jit(
            self.wm_params,
            self.actor_params,
            self.critic_params,
            self.target_critic_params,
            self.wm_opt_state,
            self.actor_opt_state,
            self.critic_opt_state,
            self.taus,
            batch,
        )
        return {k: float(v) for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # Param count
    # ------------------------------------------------------------------

    def param_count(self) -> dict[str, int]:
        """Count parameters per sub-network."""
        def _count(params):
            return sum(x.size for x in jax.tree.leaves(params))

        return {
            "world_model": _count(self.wm_params),
            "actor": _count(self.actor_params),
            "critic": _count(self.critic_params),
            "total": (
                _count(self.wm_params)
                + _count(self.actor_params)
                + _count(self.critic_params)
            ),
        }
