"""Adapted QNet: QR-DQN with a residual embedding adapter for RL fine-tuning.

The adapter is a 2-layer residual MLP applied to the JEPA embedding dims
of the observation (first 2*d_model features: h_t and h_prev).
Zero-initialized output → acts as identity at training start, enabling
warm-start from a pretrained DQN checkpoint.

Architecture:
    obs (2*d_model + 3)
     ├─ h_t   (d_model)  ──→ adapt(h_t)   = h_t   + Dense(relu(Dense(h_t)))
     ├─ h_prev (d_model) ──→ adapt(h_prev) = h_prev + Dense(relu(Dense(h_prev)))
     └─ scalars (3) ─────────────────────────────────────────────────────────┐
                                                                              │
    concat(adapt_h_t, adapt_h_prev, scalars)  →  QR-DQN head  →  (n_a, n_q)

The adapter params (adapt_fc, adapt_out) use a small LR (1e-5).
The DQN head params (fc1, fc2, fc_out) use the normal LR (1e-3).
Both sets of params can be initialized with optax.multi_transform.
"""

import jax.numpy as jnp
import flax.linen as nn


class AdaptedQNet(nn.Module):
    """QR-DQN with a residual embedding adapter.

    Drop-in replacement for QNetwork in the cross-sectional DQN training
    pipeline. All existing collect/eval/update functions work unchanged.

    Args:
        d_model:     JEPA embedding dim (obs[:d_model]=h_t, obs[d:2d]=h_prev)
        hidden_dim:  Q-network hidden dim (MXU-aligned: use 512)
        n_actions:   number of discrete actions (default 3)
        n_quantiles: number of quantile atoms (default 32)
        adapter_dim: hidden dim of the adapter MLP (default 256)
    """
    d_model: int
    hidden_dim: int = 512
    n_actions: int = 3
    n_quantiles: int = 32
    adapter_dim: int = 256

    def setup(self):
        # Adapter — shared across h_t and h_prev; zero-init output = identity
        self.adapt_fc  = nn.Dense(self.adapter_dim, name="adapt_fc")
        self.adapt_out = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="adapt_out",
        )
        # QR-DQN head — same topology as original QNetwork
        self.fc1    = nn.Dense(self.hidden_dim,           name="fc1")
        self.fc2    = nn.Dense(self.hidden_dim // 2,      name="fc2")
        self.fc_out = nn.Dense(self.n_actions * self.n_quantiles, name="fc_out")

    def _adapt(self, h: jnp.ndarray) -> jnp.ndarray:
        """Residual adapter: h + MLP(h).  Identity at initialization."""
        return h + self.adapt_out(nn.relu(self.adapt_fc(h)))

    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            obs: (..., 2*d_model + 3) observation (h_t ‖ h_prev ‖ scalars)

        Returns:
            (..., n_actions, n_quantiles) quantile Q-values
        """
        d = self.d_model
        h_t     = obs[..., :d]
        h_prev  = obs[..., d:2 * d]
        scalars = obs[..., 2 * d:]

        h_t_a    = self._adapt(h_t)
        h_prev_a = self._adapt(h_prev)

        x = jnp.concatenate([h_t_a, h_prev_a, scalars], axis=-1)
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        x = self.fc_out(x)
        # reshape handles arbitrary leading batch dims
        return x.reshape(*x.shape[:-1], self.n_actions, self.n_quantiles)
