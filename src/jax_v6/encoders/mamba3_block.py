"""Mamba-3 block (SISO) — port from ../JEPA/src/model.py:434-528.

Key changes over Mamba-2 (arXiv 2603.15569):
  - Exponential-trapezoidal discretization: lambda_t in [0, 1] blends
    current/previous state-input → 2nd-order accurate recurrence.
  - Complex-valued SSM via data-dependent RoPE on B and C — rotational state
    tracking, captures periodicities / oscillations that real SSMs cannot.
  - BCNorm (RMSNorm on B/C projections) + learnable biases — replaces
    post-gate RMSNorm (§3.4).
  - No short causal conv1d (the trapezoidal B-term replaces it).

For finance/markets the complex SSM is the main motivation: price dynamics
have rotational/oscillatory structure (cycles, mean-reversion) that real SSMs
smear out. The trapezoidal term smooths noisy state updates.

Parameter count is comparable to Mamba-2 (no conv; extra theta_proj d_model -> d_state/2
and BC norms, offset by the absent conv). MXU 128x128 alignment: head_dim
(d_model * expand / n_heads) must still equal 128 to saturate MXU tiles.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import Array

from .ssd_matrix import ssd_matrix_scan


def _apply_rope(x: Array, phi: Array) -> Array:
    """Data-dependent RoPE rotation (Mamba-3 §3.2, Proposition 3).

    Split-half pairing (first half <-> second half), same convention as LLaMA
    RoPE and as ../JEPA — avoids the interleave pattern whose backward is
    flagged "Reduction Function not modeled" by XLA on v6e.

    x:   (..., N)    — N must be even
    phi: (..., N/2)  — cumulative rotation angles
    """
    half = x.shape[-1] // 2
    x_r = x[..., :half]
    x_i = x[..., half:]
    cos_p = jnp.cos(phi)
    sin_p = jnp.sin(phi)
    return jnp.concatenate(
        [x_r * cos_p - x_i * sin_p, x_r * sin_p + x_i * cos_p],
        axis=-1,
    )


def _dt_init(key, shape, dtype=jnp.float32):
    """Standard Mamba dt-bias init: U(log(1e-3), log(1e-1)) + softplus^{-1}."""
    u = jax.random.uniform(
        key, shape, dtype=dtype, minval=math.log(1e-3), maxval=math.log(1e-1)
    )
    dt_vals = jnp.clip(jnp.exp(u), min=1e-4)
    # softplus^{-1}(x) = log(exp(x) - 1); for stability compute as dt_vals + log(-expm1(-dt_vals))
    return dt_vals + jnp.log(-jnp.expm1(-dt_vals))


def _a_log_init(key, shape, dtype=jnp.float32):
    """A = -exp(A_log) so that A < 0; init so -exp(A_log) = -(1..n_heads)."""
    n_heads = shape[0]
    return jnp.log(jnp.arange(1, n_heads + 1, dtype=dtype))


class Mamba3Block(nn.Module):
    """Mamba-3 block (SISO, trapezoidal, complex SSM via RoPE on B/C).

    Same plumbing as Mamba2Block (LayerNorm + residual + gate + out_proj)
    but replaces the scan with trapezoidal-exp + RoPE on B, C. No conv.
    """

    d_model: int = 128
    d_state: int = 16            # MUST be even (RoPE pair half)
    n_heads: int = 2
    expand_factor: int = 2
    dt_max_delta: float = 2.0    # clock modulation bound

    @nn.compact
    def __call__(
        self,
        u: Array,
        weekend_mask: Array | None = None,
        vol_clock: Array | None = None,
        exo_clock: Array | None = None,
    ) -> Array:
        assert self.d_state % 2 == 0, "d_state must be even for RoPE pairing"
        d_inner = self.d_model * self.expand_factor
        head_dim = d_inner // self.n_heads
        assert d_inner % self.n_heads == 0

        residual = u
        x_in = nn.LayerNorm(name="norm")(u)

        # in_proj: x, z, B, C, dt, lam — no conv slot
        in_proj_size = (
            d_inner          # x branch
            + d_inner        # z (gate)
            + self.d_state   # B (SISO, shared across heads)
            + self.d_state   # C (SISO)
            + self.n_heads   # dt (one per head)
            + self.n_heads   # lam (one per head)
        )
        proj = nn.Dense(in_proj_size, use_bias=False, name="in_proj")(x_in)

        idx = 0
        x = proj[..., idx:idx + d_inner];                idx += d_inner
        z = proj[..., idx:idx + d_inner];                idx += d_inner
        B_raw = proj[..., idx:idx + self.d_state];       idx += self.d_state
        C_raw = proj[..., idx:idx + self.d_state];       idx += self.d_state
        dt_raw = proj[..., idx:idx + self.n_heads];      idx += self.n_heads
        lam_raw = proj[..., idx:idx + self.n_heads]

        # A_log -> A < 0
        A_log = self.param(
            "A_log", _a_log_init, (self.n_heads,)
        )
        A = -jnp.exp(A_log)                              # (H,) negative

        # dt_bias (standard Mamba init)
        dt_bias = self.param(
            "dt_bias", _dt_init, (self.n_heads,)
        )
        dt = jax.nn.softplus(dt_raw + dt_bias)           # (B, L, H)

        # Clock modulation on dt (pre-softplus): use raw + bias path, same as Mamba-2 block
        # We apply the clock bias AFTER softplus-based dt is computed, by folding into the
        # raw pre-softplus path and recomputing softplus — simpler to keep clock bounded here.
        if exo_clock is not None:
            exo_raw = nn.Dense(
                self.n_heads, use_bias=True, name="exo_proj",
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
            )(exo_clock)
            dt = jax.nn.softplus(dt_raw + dt_bias + self.dt_max_delta * jnp.tanh(exo_raw))
        elif vol_clock is not None:
            vol_raw = nn.Dense(
                self.n_heads, use_bias=True, name="vol_proj",
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
            )(vol_clock[..., None])
            dt = jax.nn.softplus(dt_raw + dt_bias + self.dt_max_delta * jnp.tanh(vol_raw))

        # Weekend gating: zero-out dt on weekend positions so the SSM stops evolving.
        if weekend_mask is not None:
            gate = 1.0 - weekend_mask[..., None].astype(dt.dtype)
            dt = dt * gate

        # Trapezoidal mix lambda in [0, 1]
        lam = jax.nn.sigmoid(lam_raw)                    # (B, L, H)

        # BCNorm + learnable bias (§3.4) — replaces post-gate RMSNorm
        B_ssm = nn.RMSNorm(name="bc_norm_B")(B_raw)
        C_ssm = nn.RMSNorm(name="bc_norm_C")(C_raw)
        B_bias = self.param("B_bias", nn.initializers.zeros, (self.d_state,))
        C_bias = self.param("C_bias", nn.initializers.zeros, (self.d_state,))
        B_ssm = B_ssm + B_bias
        C_ssm = C_ssm + C_bias

        # Data-dependent RoPE: cumulative phase phi_t = sum_{s<=t} dt_s * theta_s
        # (§3.2). f32 accumulation to prevent bf16 drift over long sequences.
        theta = nn.Dense(
            self.d_state // 2, use_bias=False, name="theta_proj",
            kernel_init=nn.initializers.normal(stddev=0.01),
        )(u)                                             # (B, L, d_state//2)
        dt_scalar = jnp.mean(dt, axis=-1, keepdims=True) # (B, L, 1)
        phi = jnp.cumsum(
            dt_scalar.astype(jnp.float32) * theta.astype(jnp.float32),
            axis=1,
        ).astype(B_ssm.dtype)                            # (B, L, d_state//2)
        B_ssm = _apply_rope(B_ssm, phi)
        C_ssm = _apply_rope(C_ssm, phi)

        # x heads: (B, L, d_inner) -> (B, L, H, D_h) + SiLU
        x = jax.nn.silu(x)
        x_heads = x.reshape(*x.shape[:-1], self.n_heads, head_dim)

        # SSD L x L scan (single chunk)
        y_heads = ssd_matrix_scan(x_heads, B_ssm, C_ssm, dt, A, lam)

        # D skip (residual on x through identity) — per-head learnable scalar
        D = self.param("D", nn.initializers.ones, (self.n_heads,))
        y_heads = y_heads + x_heads * D[None, None, :, None]

        # Gate (no post-gate RMSNorm — BCNorm replaces it, §3.4)
        y = y_heads.reshape(*y_heads.shape[:-2], d_inner) * jax.nn.silu(z)

        # Output projection + residual
        y = nn.Dense(
            self.d_model, use_bias=False, name="out_proj",
            kernel_init=nn.initializers.zeros,  # zero-init -> identity-like residual at step 0
        )(y)
        return y + residual
