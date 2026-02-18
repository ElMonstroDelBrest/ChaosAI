"""OT-CFM (Optimal Transport Continuous Flow Matching) predictor in JAX.

Port of strate_ii/flow_predictor.py with two major changes:
1. ODE integration via diffrax (correct adjoint backprop)
2. Sinkhorn OT replaces Hungarian (JIT-compatible, no scipy dependency)

Training: t ~ U[0,1], x_0 ~ N(0,I), pi = Sinkhorn(x_0, h_y_tgt),
          x_t = (1-t)*x_0[pi] + t*h_y_tgt, L = ||v_theta - (h_y_tgt - x_0)||^2

Inference: Euler via diffrax.diffeqsolve from x_0 ~ N(0,I) to x_1.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import Array
import diffrax


# --------------------------------------------------------------------------
# Sinkhorn OT (JIT-compatible replacement for scipy.linear_sum_assignment)
# --------------------------------------------------------------------------

def sinkhorn_coupling(x0: Array, x1: Array, epsilon: float = 0.05, n_iter: int = 50) -> Array:
    """Sinkhorn optimal transport coupling (JIT-compatible).

    Computes a soft coupling matrix via entropy-regularized OT, then extracts
    a hard permutation via argmax per row.

    Args:
        x0: (B, D_flat) source samples (noise).
        x1: (B, D_flat) target samples.
        epsilon: Entropic regularization (smaller = closer to exact OT).
        n_iter: Number of Sinkhorn iterations.

    Returns:
        (B,) int32 permutation indices. x0[perm] is OT-coupled to x1.
    """
    # Cost matrix: C[i,j] = ||x0[i] - x1[j]||^2
    x0_sq = jnp.sum(x0 ** 2, axis=1, keepdims=True)  # (B, 1)
    x1_sq = jnp.sum(x1 ** 2, axis=1, keepdims=True)  # (B, 1)
    C = x0_sq + x1_sq.T - 2.0 * (x0 @ x1.T)          # (B, B)

    # Sinkhorn iterations in log-space for stability
    log_K = -C / epsilon
    log_u = jnp.zeros(C.shape[0])
    log_v = jnp.zeros(C.shape[1])

    def sinkhorn_step(carry, _):
        log_u, log_v = carry
        log_u = -jax.nn.logsumexp(log_K + log_v[None, :], axis=1)
        log_v = -jax.nn.logsumexp(log_K + log_u[:, None], axis=0)
        return (log_u, log_v), None

    (log_u, log_v), _ = jax.lax.scan(sinkhorn_step, (log_u, log_v), None, length=n_iter)

    # Transport plan: P[i,j] = exp(log_u[i] + log_K[i,j] + log_v[j])
    log_P = log_u[:, None] + log_K + log_v[None, :]

    # Hard assignment: argmax per row
    return jnp.argmax(log_P, axis=1)


# --------------------------------------------------------------------------
# Sinusoidal time embedding
# --------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for continuous time t in [0, 1]."""
    d_model: int

    @nn.compact
    def __call__(self, t: Array) -> Array:
        """(B,) -> (B, d_model)."""
        n_freq = self.d_model // 2
        exponent = jnp.arange(n_freq, dtype=jnp.float32) / max(n_freq - 1, 1)
        freqs = 10_000.0 ** exponent  # (n_freq,)

        args = t.astype(jnp.float32)[:, None] * freqs[None, :]  # (B, n_freq)
        emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)  # (B, d_model)

        return nn.Dense(self.d_model, name="time_proj")(nn.gelu(
            nn.Dense(self.d_model, name="time_linear")(emb)
        ))


# --------------------------------------------------------------------------
# Flow predictor
# --------------------------------------------------------------------------

class FlowPredictor(nn.Module):
    """OT-CFM predictor: learns v_theta(x_t, t, context) -> velocity field.

    Uses diffrax for ODE integration at inference and Sinkhorn OT for
    JIT-compatible optimal transport coupling during training.
    """
    d_model: int = 128
    hidden_dim: int = 256
    n_layers: int = 2
    seq_len: int = 128
    dropout: float = 0.1
    ot: bool = True
    ot_epsilon: float = 0.05
    ot_sinkhorn_iters: int = 50

    @nn.compact
    def __call__(
        self,
        h_x: Array,             # (B, S, d_model)
        target_positions: Array,  # (B, N_tgt) int64
        h_y_tgt: Array,         # (B, N_tgt, d_model) = x_1 (target)
        key: Array,             # PRNGKey for t and x_0 sampling
        deterministic: bool = False,
    ) -> tuple[Array, Array]:
        """OT-CFM training step.

        Returns:
            v_pred: (B, N_tgt, d_model) predicted velocity.
            v_tgt:  (B, N_tgt, d_model) target velocity = h_y_tgt - x_0.
        """
        B = h_y_tgt.shape[0]
        key_t, key_x0 = jax.random.split(key)

        t = jax.random.uniform(key_t, (B,), dtype=h_y_tgt.dtype)  # (B,)
        x_0 = jax.random.normal(key_x0, h_y_tgt.shape, dtype=h_y_tgt.dtype)

        # Sinkhorn OT coupling
        if self.ot and B > 1:
            x0_flat = x_0.reshape(B, -1)
            x1_flat = h_y_tgt.reshape(B, -1)
            perm = sinkhorn_coupling(x0_flat, x1_flat, self.ot_epsilon, self.ot_sinkhorn_iters)
            x_0 = x_0[perm]

        t_e = t[:, None, None]  # (B, 1, 1)
        x_t = (1.0 - t_e) * x_0 + t_e * h_y_tgt
        v_tgt = h_y_tgt - x_0

        v_pred = self._forward_velocity(x_t, t, h_x, target_positions, deterministic)
        return v_pred, v_tgt

    def _forward_velocity(
        self,
        x_t: Array,
        t: Array,
        h_x: Array,
        target_positions: Array,
        deterministic: bool = False,
    ) -> Array:
        """Compute predicted velocity v_theta(x_t, t, context).

        Returns (B, N_tgt, d_model).
        """
        N_tgt = x_t.shape[1]

        t_emb = SinusoidalTimeEmbedding(
            self.d_model, name="time_embed"
        )(t)[:, None, :]  # (B, 1, d_model)
        t_emb = jnp.broadcast_to(t_emb, (x_t.shape[0], N_tgt, self.d_model))

        ctx = h_x[:, -1:, :]  # (B, 1, d_model)
        ctx = jnp.broadcast_to(ctx, (x_t.shape[0], N_tgt, self.d_model))

        pos = nn.Embed(
            num_embeddings=self.seq_len,
            features=self.d_model,
            name="pos_embed",
        )(target_positions)  # (B, N_tgt, d_model)

        inp = jnp.concatenate([x_t, t_emb, ctx, pos], axis=-1)  # (B, N_tgt, 4*d_model)

        # Velocity-field MLP
        x = inp
        for i in range(self.n_layers):
            out_dim = self.hidden_dim if i < self.n_layers - 1 else self.d_model
            x = nn.Dense(out_dim, name=f"vf_{i}")(x)
            if i < self.n_layers - 1:
                x = nn.gelu(x)
                x = nn.Dropout(rate=self.dropout, deterministic=deterministic)(x)

        return x

    def sample(
        self,
        params: dict,
        h_x: Array,
        target_positions: Array,
        key: Array,
        n_steps: int = 2,
    ) -> Array:
        """Euler integration from x_0 ~ N(0,I) to x_1 via diffrax.

        Args:
            params: Flax params dict for this module.
            h_x: (B, S, d_model) context encoder output.
            target_positions: (B, N_tgt) int64.
            key: PRNGKey.
            n_steps: Number of Euler steps.

        Returns:
            (B, N_tgt, d_model) sampled predictions.
        """
        B, N_tgt = target_positions.shape
        x0 = jax.random.normal(key, (B, N_tgt, self.d_model), dtype=h_x.dtype)

        def vector_field(t, y, args):
            """ODE right-hand side: v_theta(y, t, context)."""
            t_batch = jnp.full((B,), t)
            return self.apply(
                {'params': params},
                y, t_batch, h_x, target_positions, True,
                method=self._forward_velocity,
            )

        term = diffrax.ODETerm(vector_field)
        solver = diffrax.Euler()
        sol = diffrax.diffeqsolve(
            term, solver,
            t0=0.0, t1=1.0, dt0=1.0 / n_steps,
            y0=x0,
        )
        return sol.ys[-1] if sol.ys.ndim > 3 else sol.ys
