"""Chunked SSD (State Space Duality) — the core TPU kernel.

Replaces associative_scan with dense matmuls that fill the MXU 128x128 tiles.

For seq_len=128 and chunk_size=128: exactly 1 chunk, zero inter-chunk overhead.
The L matrix construction is (128, 64) @ (64, 128) = one 128x128 MXU tile.
The output matmul is (128, 128) @ (128, P) = one MXU tile per head.

Numerical stability:
  A < 0 and cumsum(dt) >= 0, so exp(A * cs) <= 1 (decaying, safe).
  But exp(-A * cs) in B_hat can overflow. We use log-sum-exp normalization:
    offset = max(-A * cs)
    C_hat = C * exp(A*cs + offset)
    B_hat = dt * B * exp(-A*cs - offset)
  The offset cancels in C_hat @ B_hat^T, keeping magnitudes bounded.
"""

import jax
import jax.numpy as jnp
from jax import Array
from functools import partial


def naive_sequential_scan(
    x: Array,              # (B, L, H, P)
    dt_raw: Array,         # (B, L, H)
    A_log: Array,          # (H, N)
    B: Array,              # (B, L, H, N)
    C: Array,              # (B, L, H, N)
    weekend_mask: Array | None = None,
) -> Array:
    """Pure sequential scan for correctness testing. O(L) sequential steps.

    Same interface as chunked_ssd but uses a simple loop.
    """
    Bs, L, H, P = x.shape
    N = A_log.shape[1]
    A = -jnp.exp(A_log)  # (H, N) negative

    dt = jax.nn.softplus(dt_raw)  # (B, L, H)
    if weekend_mask is not None:
        gate = 1.0 - weekend_mask[..., None]  # (B, L, 1)
        dt = dt * gate

    def scan_fn(h, t):
        # h: (B, H, N, P)
        dt_t = dt[:, t, :]       # (B, H)
        B_t = B[:, t, :, :]      # (B, H, N)
        C_t = C[:, t, :, :]      # (B, H, N)
        x_t = x[:, t, :, :]      # (B, H, P)

        # A_bar = exp(A * dt) — (H, N) broadcast with (B, H) -> (B, H, N)
        A_bar = jnp.exp(A[None, :, :] * dt_t[:, :, None])  # (B, H, N)

        # B_bar = dt * B — (B, H, N)
        B_bar = dt_t[:, :, None] * B_t  # (B, H, N)

        # h = A_bar * h + B_bar * x  — (B, H, N, P)
        h = A_bar[:, :, :, None] * h + B_bar[:, :, :, None] * x_t[:, :, None, :]

        # y = C^T h — (B, H, P)
        y_t = jnp.einsum("bhn,bhnp->bhp", C_t, h)

        return h, y_t

    h0 = jnp.zeros((Bs, H, N, P), dtype=x.dtype)
    _, ys = jax.lax.scan(scan_fn, h0, jnp.arange(L))
    # ys: (L, B, H, P) -> (B, L, H, P)
    return jnp.transpose(ys, (1, 0, 2, 3))


def chunked_ssd(
    x: Array,              # (B, L, H, P) input [head_dim P]
    dt_raw: Array,         # (B, L, H) raw delta (pre-softplus)
    A_log: Array,          # (H, N) log-space SSM A [d_state N]
    B: Array,              # (B, L, H, N) input matrix
    C: Array,              # (B, L, H, N) output matrix
    weekend_mask: Array | None = None,  # (B, L)
    chunk_size: int = 128,
) -> Array:                # (B, L, H, P) output
    """Chunked SSD: dense matmuls that fill MXU 128x128 tiles.

    Algorithm per chunk of size Cs:
      1. dt = softplus(dt_raw) * (1 - weekend_mask)
      2. Reshape to chunks: (B, L, ...) -> (B, n_chunks, Cs, ...)
      3. Per chunk (vectorized over B and H):
         - Cumulative dt for decay factors
         - L = tril(C_hat @ B_hat^T)  — 128x128 MXU matmul
         - y_intra = L @ x            — 128xP MXU matmul
         - y_from_state via inter-chunk scan
      4. jax.lax.scan over chunks for state propagation
    """
    Bs, L, H, P = x.shape
    N = A_log.shape[1]
    A = -jnp.exp(A_log)  # (H, N) — negative (decay)

    # 1. Prepare dt
    dt = jax.nn.softplus(dt_raw)  # (B, L, H)
    if weekend_mask is not None:
        gate = 1.0 - weekend_mask[..., None]  # (B, L, 1)
        dt = dt * gate

    # Pad L to multiple of chunk_size if needed
    n_chunks = (L + chunk_size - 1) // chunk_size
    pad_len = n_chunks * chunk_size - L
    if pad_len > 0:
        x = jnp.pad(x, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        dt = jnp.pad(dt, ((0, 0), (0, pad_len), (0, 0)))
        B = jnp.pad(B, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        C = jnp.pad(C, ((0, 0), (0, pad_len), (0, 0), (0, 0)))

    L_padded = n_chunks * chunk_size
    Cs = chunk_size

    # 2. Reshape to chunks: (B, n_chunks, Cs, ...)
    x_c = x.reshape(Bs, n_chunks, Cs, H, P)
    dt_c = dt.reshape(Bs, n_chunks, Cs, H)
    B_c = B.reshape(Bs, n_chunks, Cs, H, N)
    C_c = C.reshape(Bs, n_chunks, Cs, H, N)

    # Cumulative dt per chunk: (B, n_chunks, Cs, H)
    cs = jnp.cumsum(dt_c, axis=2)  # cumulative within each chunk

    # A broadcast: (H, N) -> will be used as A[h, n]
    # decay_factor[b, chunk, t, h, n] = A[h,n] * cs[b, chunk, t, h]
    # We compute per (b, chunk, h) the matrices needed

    # 3. Per-chunk computation (vectorized via vmap)
    # Process one chunk for one batch element and one head
    def process_chunk(x_ch, dt_ch, B_ch, C_ch, cs_ch, A_h, h_init):
        """Process one chunk for one head.

        Args:
            x_ch: (Cs, P)
            dt_ch: (Cs,)
            B_ch: (Cs, N)
            C_ch: (Cs, N)
            cs_ch: (Cs,) cumulative dt
            A_h: (N,) SSM A for this head (negative)
            h_init: (N, P) state from previous chunk

        Returns:
            y: (Cs, P) output
            h_final: (N, P) state for next chunk
        """
        # decay_factors[t, n] = A_h[n] * cs_ch[t]
        decay = A_h[None, :] * cs_ch[:, None]  # (Cs, N)

        # Log-sum-exp normalization for numerical stability
        # log_B = -A * cs (positive, can be large)
        log_B = -decay  # (Cs, N) — positive values
        offset = jnp.max(log_B, axis=0, keepdims=True)  # (1, N)

        # C_hat[t, n] = C[t, n] * exp(A*cs[t] + offset[n])
        C_hat = C_ch * jnp.exp(decay + offset)  # (Cs, N)

        # B_hat[t, n] = dt[t] * B[t, n] * exp(-A*cs[t] - offset[n])
        B_hat = (dt_ch[:, None] * B_ch) * jnp.exp(-decay - offset)  # (Cs, N)

        # L = tril(C_hat @ B_hat^T) — (Cs, Cs) MXU matmul!
        L_mat = jnp.tril(C_hat @ B_hat.T)  # (Cs, Cs)

        # y_intra = L @ x — (Cs, P) MXU matmul!
        y_intra = L_mat @ x_ch  # (Cs, P)

        # Contribution from previous chunk's state h_init
        # y_from_state[t] = C_hat_raw[t] @ h_init where C_hat_raw uses no offset
        C_hat_raw = C_ch * jnp.exp(decay)  # (Cs, N) — stable since decay <= 0
        y_from_state = C_hat_raw @ h_init  # (Cs, P)

        y = y_intra + y_from_state

        # Final state for this chunk: h_final = decay_end * h_init + sum_t(B_hat_raw[t] * x[t])
        decay_end = jnp.exp(A_h * cs_ch[-1])  # (N,) — exp(negative * positive) <= 1
        h_final = decay_end[:, None] * h_init  # (N, P)

        # Accumulate: each timestep t contributes B_hat_end[t] * x[t]
        # where B_hat_end[t] = dt[t] * B[t] * exp(A * (cs_end - cs[t]))
        # = dt[t] * B[t] * exp(A * cs_end) * exp(-A * cs[t])
        # For stability, use the same offset trick
        residual_decay = A_h[None, :] * (cs_ch[-1] - cs_ch)[:, None]  # (Cs, N) — negative * positive = negative
        B_hat_end = (dt_ch[:, None] * B_ch) * jnp.exp(residual_decay)  # (Cs, N)
        h_final = h_final + B_hat_end.T @ x_ch  # (N, P)

        return y, h_final

    # Vectorize over heads and batch
    # vmap over H (head dimension)
    process_chunk_vH = jax.vmap(
        process_chunk,
        in_axes=(1, 0, 1, 1, 0, 0, 0),  # x:(Cs,H,P)->P=1, dt:(Cs,H)->0, etc.
        out_axes=(1, 0),  # y:(Cs,H,P)->1, h:(H,N,P)->0
    )

    # Scan over chunks with per-head state propagation
    def scan_chunk(carry, chunk_inputs):
        """carry: h_state (B, H, N, P), chunk_inputs: per-chunk tensors."""
        h_state = carry  # (B, H, N, P)
        x_ch, dt_ch, B_ch, C_ch, cs_ch = chunk_inputs

        # Process per batch element (vmap over B)
        def process_one_batch(x_b, dt_b, B_b, C_b, cs_b, h_b):
            # x_b: (Cs, H, P), dt_b: (Cs, H), B_b: (Cs, H, N), C_b: (Cs, H, N)
            # cs_b: (Cs, H), h_b: (H, N, P)
            return process_chunk_vH(x_b, dt_b, B_b, C_b, cs_b, A, h_b)

        process_batch = jax.vmap(
            process_one_batch,
            in_axes=(0, 0, 0, 0, 0, 0),
            out_axes=(0, 0),
        )

        y_chunk, h_new = process_batch(x_ch, dt_ch, B_ch, C_ch, cs_ch, h_state)
        return h_new, y_chunk

    # Prepare scan inputs: move chunk axis to leading position for lax.scan
    # x_c: (B, n_chunks, Cs, H, P) -> scan over dim 1
    scan_inputs = (
        jnp.transpose(x_c, (1, 0, 2, 3, 4)),   # (n_chunks, B, Cs, H, P)
        jnp.transpose(dt_c, (1, 0, 2, 3)),      # (n_chunks, B, Cs, H)
        jnp.transpose(B_c, (1, 0, 2, 3, 4)),    # (n_chunks, B, Cs, H, N)
        jnp.transpose(C_c, (1, 0, 2, 3, 4)),    # (n_chunks, B, Cs, H, N)
        jnp.transpose(cs, (1, 0, 2, 3)),         # (n_chunks, B, Cs, H)
    )

    h0 = jnp.zeros((Bs, H, N, P), dtype=x.dtype)
    _, y_chunks = jax.lax.scan(scan_chunk, h0, scan_inputs)
    # y_chunks: (n_chunks, B, Cs, H, P)

    # Reassemble: (n_chunks, B, Cs, H, P) -> (B, L_padded, H, P)
    y = jnp.transpose(y_chunks, (1, 0, 2, 3, 4)).reshape(Bs, L_padded, H, P)

    # Remove padding
    if pad_len > 0:
        y = y[:, :L, :, :]

    return y
