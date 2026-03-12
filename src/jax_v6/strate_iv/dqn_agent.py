"""Quantile Regression DQN with CVaR action selection — JAX/Flax.

Discrete 3-action agent (Short/Flat/Long) for All-Weather trading.
Uses QR-DQN to learn the full return distribution per action,
then selects actions by CVaR (worst-alpha% conditional expected return).

Components:
  - QNetwork: obs → (n_actions, n_quantiles) distributional Q-values
  - ReplayBuffer: ring buffer for offline transitions (numpy)
  - DQNAgent: epsilon-greedy exploration, QR-DQN loss, EMA target network
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax

from .critic import cvar_from_quantiles, quantile_huber_loss


class QNetwork(nn.Module):
    """Quantile Q-network: obs → (n_actions, n_quantiles).

    Two hidden layers with ReLU, output reshaped to (n_actions, n_quantiles).
    hidden_dim=512 is MXU-aligned (512/4=128).
    """
    hidden_dim: int = 512
    n_actions: int = 3
    n_quantiles: int = 32

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            obs: (B, obs_dim) observation vector

        Returns:
            (B, n_actions, n_quantiles) quantile Q-values
        """
        x = nn.relu(nn.Dense(self.hidden_dim)(obs))
        x = nn.relu(nn.Dense(self.hidden_dim // 2)(x))
        x = nn.Dense(self.n_actions * self.n_quantiles)(x)
        return x.reshape(-1, self.n_actions, self.n_quantiles)


class ReplayBuffer:
    """Ring buffer for RL transitions (numpy, CPU-side).

    Stores (obs, action, reward, next_obs, done) tuples.
    Sampling returns JAX arrays on the default device.
    """

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        """Add a single transition."""
        idx = self.pos % self.capacity
        self.obs[idx] = np.asarray(obs)
        self.action[idx] = int(action)
        self.reward[idx] = float(reward)
        self.next_obs[idx] = np.asarray(next_obs)
        self.done[idx] = float(done)
        self.pos += 1
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, obs, action, reward, next_obs, done):
        """Add a batch of transitions."""
        n = len(obs)
        for i in range(n):
            self.add(obs[i], action[i], reward[i], next_obs[i], done[i])

    def sample(self, batch_size: int, key: jax.random.PRNGKey) -> dict:
        """Sample a random batch, return as JAX arrays."""
        indices = jax.random.randint(key, (batch_size,), 0, self.size)
        indices_np = np.asarray(indices)
        return {
            "obs": jnp.array(self.obs[indices_np]),
            "action": jnp.array(self.action[indices_np]),
            "reward": jnp.array(self.reward[indices_np]),
            "next_obs": jnp.array(self.next_obs[indices_np]),
            "done": jnp.array(self.done[indices_np]),
        }


class DQNState(NamedTuple):
    """Immutable agent state for JIT-compatible updates."""
    q_params: dict
    target_params: dict
    opt_state: optax.OptState
    step: jnp.ndarray  # () int32


def create_agent(
    obs_dim: int,
    hidden_dim: int = 512,
    n_actions: int = 3,
    n_quantiles: int = 32,
    lr: float = 3e-4,
    key: jax.random.PRNGKey = None,
) -> tuple[QNetwork, DQNState, optax.GradientTransformation]:
    """Initialize QNetwork, target network, and optimizer.

    Returns:
        q_net: QNetwork module
        agent_state: DQNState with params, target, opt_state
        tx: optimizer
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    q_net = QNetwork(hidden_dim=hidden_dim, n_actions=n_actions, n_quantiles=n_quantiles)

    dummy_obs = jnp.zeros((1, obs_dim))
    q_params = q_net.init(key, dummy_obs)["params"]
    target_params = jax.tree.map(lambda x: x.copy(), q_params)

    tx = optax.adam(lr)
    opt_state = tx.init(q_params)

    agent_state = DQNState(
        q_params=q_params,
        target_params=target_params,
        opt_state=opt_state,
        step=jnp.int32(0),
    )
    return q_net, agent_state, tx


def select_action(
    q_net: QNetwork,
    q_params: dict,
    obs: jnp.ndarray,
    key: jax.random.PRNGKey,
    epsilon: float,
    cvar_alpha: float = 0.25,
) -> jnp.ndarray:
    """Epsilon-greedy action selection with CVaR.

    With probability epsilon: random action.
    Otherwise: argmax CVaR over actions.

    Args:
        q_net: QNetwork module
        q_params: network parameters
        obs: (obs_dim,) single observation
        key: PRNG key
        epsilon: exploration rate
        cvar_alpha: CVaR tail fraction

    Returns:
        () int32 action
    """
    key_rand, key_eps = jax.random.split(key)

    # Q-values: (1, n_actions, n_quantiles)
    q_vals = q_net.apply({"params": q_params}, obs[None])
    # CVaR per action: (n_actions,)
    cvar_vals = cvar_from_quantiles(q_vals[0], alpha=cvar_alpha)

    greedy_action = jnp.argmax(cvar_vals)
    random_action = jax.random.randint(key_rand, (), 0, 3)

    # Epsilon-greedy
    explore = jax.random.uniform(key_eps) < epsilon
    return jnp.where(explore, random_action, greedy_action).astype(jnp.int32)


def compute_epsilon(step: int, eps_start: float, eps_end: float, eps_decay_steps: int) -> float:
    """Linear epsilon decay."""
    frac = jnp.clip(step / max(eps_decay_steps, 1), 0.0, 1.0)
    return eps_start + (eps_end - eps_start) * frac


def update_step(
    q_net: QNetwork,
    agent_state: DQNState,
    tx: optax.GradientTransformation,
    batch: dict,
    gamma: float = 0.99,
    cvar_alpha: float = 0.25,
    n_quantiles: int = 32,
) -> tuple[DQNState, dict]:
    """Single QR-DQN update step.

    Args:
        q_net: Q-network module
        agent_state: current DQNState
        tx: optimizer
        batch: dict with obs, action, reward, next_obs, done
        gamma: discount factor
        cvar_alpha: CVaR alpha for target action selection
        n_quantiles: number of quantiles

    Returns:
        new_agent_state: updated DQNState
        metrics: dict with loss, mean_q, etc.
    """
    taus = (jnp.arange(n_quantiles, dtype=jnp.float32) + 0.5) / n_quantiles

    def loss_fn(q_params):
        # Current Q-values: (B, n_actions, n_quantiles)
        q_all = q_net.apply({"params": q_params}, batch["obs"])
        # Select quantiles for taken actions: (B, n_quantiles)
        B = batch["obs"].shape[0]
        action_idx = batch["action"].astype(jnp.int32)
        q_pred = q_all[jnp.arange(B), action_idx]  # (B, n_quantiles)

        # Target Q-values (no gradient)
        q_next_all = q_net.apply({"params": agent_state.target_params}, batch["next_obs"])
        # Select best action by CVaR
        cvar_next = jax.vmap(lambda q: cvar_from_quantiles(q, alpha=cvar_alpha))(q_next_all)
        best_actions = jnp.argmax(cvar_next, axis=-1)  # (B,)
        q_next = q_next_all[jnp.arange(B), best_actions]  # (B, n_quantiles)

        # TD targets: r + gamma * (1 - done) * q_next
        targets = batch["reward"][:, None] + gamma * (1.0 - batch["done"][:, None]) * q_next

        loss = quantile_huber_loss(q_pred, targets, taus)
        mean_q = q_pred.mean()
        return loss, mean_q

    (loss, mean_q), grads = jax.value_and_grad(loss_fn, has_aux=True)(agent_state.q_params)

    updates, new_opt_state = tx.update(grads, agent_state.opt_state, agent_state.q_params)
    new_q_params = optax.apply_updates(agent_state.q_params, updates)

    new_state = DQNState(
        q_params=new_q_params,
        target_params=agent_state.target_params,
        opt_state=new_opt_state,
        step=agent_state.step + 1,
    )

    metrics = {"loss": loss, "mean_q": mean_q}
    return new_state, metrics


def update_target(agent_state: DQNState, tau: float = 0.005) -> DQNState:
    """Soft update target network: target = tau * q + (1-tau) * target."""
    new_target = jax.tree.map(
        lambda t, q: tau * q + (1.0 - tau) * t,
        agent_state.target_params,
        agent_state.q_params,
    )
    return agent_state._replace(target_params=new_target)
