"""Depth-attention prediction head.

Port from ../JEPA/src/model.py:333-359 (MLPHead with AttnRes over encoder depth).

Instead of consuming only the final encoder layer, the head softmax-attends over
the last-position representation of every layer (h_0, h_1, ..., h_N). Each
output dimension (e.g. a responder, or a return value) can thus choose the
abstraction depth it needs:
  - shallow layers carry lexical / codebook-local structure
  - deep layers carry long-range / regime structure

Mechanism per position t:
  V_t  = stack_i h_i[:, t, :]        # (L+1, B, d_model)
  K_t  = RMSNorm(V_t)
  a    = softmax_i(<q, K_t>)         # (L+1, B)
  y_t  = sum_i a_i * V_t_i           # (B, d_model)

Then Dense -> SiLU -> Dense -> out_dim.

Design choices kept from the PyTorch reference:
  - depth_query is zero-init → uniform attention at step 0 (softmax over zeros)
  - RMSNorm on keys (not layernorm) — cheaper, stabilises logit magnitudes
  - SiLU activation between the two Dense layers
  - Output Dense is normal-init (std 0.01) + zero bias → small initial outputs
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn
from jax import Array


class DepthAttnHead(nn.Module):
    """Depth-attention head: attend over encoder layer stack, then MLP.

    Args:
        hidden_dim: width of the internal MLP.
        out_dim: number of output features.
        apply_per_timestep: when True (default), the depth-attention runs
            independently per timestep and the head outputs (B, S, out_dim).
            When False, it consumes only the last timestep and outputs
            (B, out_dim) — matches the original PyTorch API.
    """

    hidden_dim: int = 64
    out_dim: int = 1
    apply_per_timestep: bool = True

    @nn.compact
    def __call__(self, hidden_states: list[Array]) -> Array:
        # hidden_states: list of (B, S, d_model) — length L+1
        V = jnp.stack(hidden_states, axis=0)  # (L+1, B, S, D)
        L_plus1, B, S, D = V.shape

        depth_query = self.param(
            "depth_query", nn.initializers.zeros, (D,)
        )
        key_norm = nn.RMSNorm(name="depth_keynorm")

        if self.apply_per_timestep:
            K = key_norm(V)  # (L+1, B, S, D)
            logits = jnp.einsum("d,lbtd->lbt", depth_query, K)  # (L+1, B, S)
            alpha = nn.softmax(logits, axis=0)
            x = jnp.einsum("lbt,lbtd->btd", alpha, V)  # (B, S, D)
        else:
            V_last = V[:, :, -1, :]  # (L+1, B, D)
            K = key_norm(V_last)     # (L+1, B, D)
            logits = jnp.einsum("d,lbd->lb", depth_query, K)  # (L+1, B)
            alpha = nn.softmax(logits, axis=0)
            x = jnp.einsum("lb,lbd->bd", alpha, V_last)  # (B, D)

        x = nn.RMSNorm(name="norm")(x)
        x = nn.Dense(self.hidden_dim, name="fc1")(x)
        x = nn.silu(x)
        x = nn.Dense(
            self.out_dim,
            name="fc2",
            kernel_init=nn.initializers.normal(stddev=0.01),
            bias_init=nn.initializers.zeros,
        )(x)
        return x
