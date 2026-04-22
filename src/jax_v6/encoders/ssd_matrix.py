"""Mamba-3 SSD single-chunk matrix scan (L x L formulation).

Port from ../JEPA/src/model.py:140-231 (_chunked_mamba3_scan_pytorch,
SSD single-chunk path).

SSD identity:
    y[t] = sum_{s<=t} decay[t, s] * (C[t] . B[s]) * coef[s] * x[s]

Where (C[t] . B[s]) is the (L, L) inner-product matrix — shared across heads
because B, C are SISO in Mamba-3. The M-matrix is O(B*L^2*H) vs outer-product
O(B*L*H*D_h*N). At L=128, H=8, D_h=128, N=16 this is ~30x less HBM, enabling
seq_len extension without OOM on v6e (32 GB/chip).

Exponential-trapezoidal discretization: each x[s] contributes through two
paths,
    coef_c[s]     = lambda[s] * delta[s]       via decay[t, s]   (t >= s)
    coef_p[s+1]   = (1-lambda[s+1]) * alpha[s+1] * delta[s+1] via decay[t, s+1]
                                                              (t > s)
The effective weight is (coef_c . decay) + (coef_p_shifted . decay_shifted).

Numerical stability (same strategy as ssd.py):
  - log_dA, log_P, phi accumulate in float32 (bf16 mantissa drifts over 128
    steps)
  - exp + einsum in bf16 for MXU speed
  - clamp log_rel at 0 before exp to prevent bf16 overflow on invalid
    (t < s) positions, then mask via the causal triangle matrix
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def ssd_matrix_scan(
    x: Array,       # (B, L, H, D_h)
    B: Array,       # (B, L, N)   — post-RoPE, SISO (shared across heads)
    C: Array,       # (B, L, N)   — post-RoPE, SISO (shared across heads)
    dt: Array,      # (B, L, H)   — post-softplus, weekend-gated
    A: Array,       # (H,)        — negative (decay), NOT log-space
    lam: Array,     # (B, L, H)   — trapezoidal mix in [0, 1]
) -> Array:
    """Single-chunk SSD L x L scan with trapezoidal discretization.

    Uses bf16 einsums for speed + float32 accumulations for stability.

    Args:
        x:   (B, L, H, D_h)
        B:   (B, L, N)
        C:   (B, L, N)
        dt:  (B, L, H)  post-softplus delta
        A:   (H,)       per-head decay rate (negative)
        lam: (B, L, H)  trapezoidal mix

    Returns:
        y: (B, L, H, D_h), same dtype as x.
    """
    batch, L, H, D_h = x.shape
    N = B.shape[2]

    # Promote temporal accumulators to f32.
    dt_f32 = dt.astype(jnp.float32)
    A_f32 = A.astype(jnp.float32)
    lam_f32 = lam.astype(jnp.float32)

    # log_dA[b, t, h] = dt[b, t, h] * A[h]  (negative since A < 0)
    log_dA = dt_f32 * A_f32[None, None, :]              # (B, L, H)
    log_P = jnp.cumsum(log_dA, axis=1)                  # (B, L, H), monotone decreasing

    # Relative log decay: log_rel[b, t, s, h] = log_P[b, t, h] - log_P[b, s, h]
    # For valid (t >= s): <= 0 → exp <= 1 (safe)
    # For invalid (t < s): > 0 → clamp to 0 before exp, then zero via mask
    log_rel = log_P[:, :, None, :] - log_P[:, None, :, :]  # (B, L_t, L_s, H)

    # Causal masks
    idx = jnp.arange(L)
    mask_ge = (idx[:, None] >= idx[None, :]).astype(log_rel.dtype)  # (L_t, L_s), t >= s
    mask_gt = (idx[:, None] >  idx[None, :]).astype(log_rel.dtype)  # (L_t, L_s), t >  s

    # Decay from s (coef_c path): clamped exp * causal mask
    decay_c = jnp.exp(jnp.minimum(log_rel, 0.0)) * mask_ge[None, :, :, None]

    # Decay from s+1 (coef_p path): build log_rel_next, clamp, exp, mask
    # log_P_next[b, s, h] = log_P[b, s+1, h]; last slot = 0 (dummy — masked out by coef_p_next)
    log_P_next = jnp.concatenate(
        [log_P[:, 1:], jnp.zeros_like(log_P[:, :1])], axis=1
    )                                                      # (B, L, H)
    log_rel_next = log_P[:, :, None, :] - log_P_next[:, None, :, :]  # (B, L_t, L_s, H)
    decay_p = jnp.exp(jnp.minimum(log_rel_next, 0.0)) * mask_gt[None, :, :, None]

    # Coefficients (f32)
    coef_c = lam_f32 * dt_f32                              # (B, L, H)
    dA = jnp.exp(log_dA)                                   # (B, L, H)
    coef_p = (1.0 - lam_f32) * dA * dt_f32                 # (B, L, H)
    # Shift coef_p: coef_p_at_s+1; last slot zeroed
    coef_p_next = jnp.concatenate(
        [coef_p[:, 1:], jnp.zeros_like(coef_p[:, :1])], axis=1
    )                                                      # (B, L, H)

    # Effective (non-negative) weight: (B, L_t, L_s, H), f32
    W = decay_c * coef_c[:, None, :, :] + decay_p * coef_p_next[:, None, :, :]

    # C . B inner-product (shared across heads since SISO): (B, L_t, L_s), f32
    CB = jnp.einsum("btn,bsn->bts", C.astype(jnp.float32), B.astype(jnp.float32))

    # M[b, t, s, h] = W[b, t, s, h] * CB[b, t, s]
    M = W * CB[:, :, :, None]                              # (B, L, L, H), f32

    # y[b, t, h, :] = sum_s M[b, t, s, h] * x[b, s, h, :]
    # Promote x to f32 for accumulation (mantissa > bf16).
    y = jnp.einsum("btsh,bshd->bthd", M, x.astype(jnp.float32))  # (B, L, H, D_h) f32

    return y.astype(x.dtype)
