"""KAN (Kolmogorov-Arnold Networks) layer for JAX/Flax — TPU v5p optimized.

v3 (Chebyshev basis) over v2 (B-spline):
  - Chebyshev polynomial basis (T_0, T_1, ..., T_n) replaces B-spline Cox-de Boor
  - Chebyshev polynomials are orthogonal on [-1,1]: no structural inter-dim correlation
  - This prevents the KAN-specific dimensional collapse that B-splines cause via
    co-adaptation of basis functions across output dimensions
  - Same compute cost: 2 muls + 1 sub per step vs Cox-de Boor (minor overhead)
  - tanh normalization maps input to [-1,1] (required for Chebyshev domain)

v2 optimizations retained:
  1. Accumulated matmuls: out += T_k @ W[k] — no concat intermediate
  2. Low-rank factored polynomial: W[k] = U·diag(s_k)·V^T, rank=128 for MXU
  3. RMSNorm: 2× fewer reductions than LayerNorm
  4. update_grid: no-op (Chebyshev doesn't need knot positions)

XLA / TPU notes
---------------
- All shapes are static at trace time.
- Python for-loop over n_order → XLA fully unrolls into a fused kernel.
- MXU alignment: out_features should be multiples of 128.
- Low-rank (rank=128): D_in×rank + rank×D_out + n_order×rank vs n_order×D_in×D_out
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import Array

# ---------------------------------------------------------------------------
# Grid update — no-op for Chebyshev KAN (no knot positions needed)
# ---------------------------------------------------------------------------

def update_grid(
    params: dict,
    x_samples: Array,
    grid_size: int,
    margin: float = 0.01,
) -> dict:
    """No-op for Chebyshev KAN (tanh normalization makes knots unnecessary)."""
    return params


# ---------------------------------------------------------------------------
# KANLayer
# ---------------------------------------------------------------------------

class KANLayer(nn.Module):
    """KAN layer with Chebyshev polynomial basis — accumulated matmuls.

    Replaces B-spline Cox-de Boor with Chebyshev recursion:
        T_0 = 1,  T_1 = x,  T_k = 2x·T_{k-1} - T_{k-2}

    Orthogonal on [-1,1] → no structural inter-dim correlation → no collapse.
    tanh normalization maps input to [-1,1] (required for Chebyshev domain).

    Full rank (rank=0):
        output = SiLU(x) @ W_base + Σ_k T_k(tanh(x)) @ W_poly[k]

    Low-rank (rank>0):
        output = SiLU(x) @ W_base + (Σ_k T_k(tanh(x)) @ U * s_k) @ V
        Use rank=128 for TPU MXU tile alignment.

    TPU alignment:
        out_features MUST be a multiple of 128 for full MXU tile utilization.
    """
    in_features: int
    out_features: int
    grid_size: int = 10     # n_order = grid_size + spline_order (Chebyshev terms)
    spline_order: int = 3
    rank: int = 0           # 0=full rank, >0=low-rank factored polynomial
    scale_noise: float = 0.1
    scale_base: float = 1.0

    @nn.compact
    def __call__(self, x: Array) -> Array:
        orig_dtype = x.dtype
        prefix = x.shape[:-1]
        B = 1
        for d in prefix:
            B *= d
        x_2d = x.reshape(B, self.in_features)
        n_order = self.grid_size + self.spline_order

        base_weight = self.param(
            'base_weight',
            nn.initializers.normal(stddev=self.scale_base / jnp.sqrt(float(self.in_features))),
            (self.in_features, self.out_features),
        )

        # Base path: bf16 for MXU efficiency
        x_c = x_2d.astype(jnp.bfloat16)
        out = jax.nn.silu(x_c) @ base_weight.astype(jnp.bfloat16)

        # Chebyshev path: float32 for stable backward pass.
        # Backward gradients through T_k = 2x·T_{k-1}−T_{k-2} scale as (2x)^k —
        # bf16 loses 3+ bits per step at n_order=8. Float32 recursion, bf16 matmuls.
        x_safe = jnp.tanh(x_2d.astype(jnp.float32))

        if self.rank > 0:
            # Low-rank: poly_out = (Σ_k T_k(x) @ U * s_k) @ V
            poly_U = self.param(
                'poly_U',
                nn.initializers.normal(stddev=self.scale_noise / jnp.sqrt(float(self.in_features))),
                (self.in_features, self.rank),
            )
            poly_V = self.param(
                'poly_V',
                nn.initializers.normal(stddev=1.0 / jnp.sqrt(float(self.rank))),
                (self.rank, self.out_features),
            )
            poly_S = self.param(
                'poly_S',
                nn.initializers.normal(stddev=self.scale_noise / jnp.sqrt(float(n_order))),
                (n_order, self.rank),
            )
            U_bf = poly_U.astype(jnp.bfloat16)
            V_bf = poly_V.astype(jnp.bfloat16)
            S_bf = poly_S.astype(jnp.bfloat16)

            T_prev = jnp.ones_like(x_safe)   # float32
            T_curr = x_safe                   # float32
            acc = (T_prev.astype(jnp.bfloat16) @ U_bf) * S_bf[0]
            if n_order > 1:
                acc = acc + (T_curr.astype(jnp.bfloat16) @ U_bf) * S_bf[1]
            for k in range(2, n_order):
                T_next = 2.0 * x_safe * T_curr - T_prev  # float32 arithmetic
                acc = acc + (T_next.astype(jnp.bfloat16) @ U_bf) * S_bf[k]
                T_prev = T_curr
                T_curr = T_next

            out = out + acc @ V_bf
        else:
            # Full rank: Σ_k T_k(x) @ W_poly[k]
            poly_weight = self.param(
                'poly_weight',
                nn.initializers.normal(stddev=self.scale_noise / jnp.sqrt(float(self.in_features * n_order))),
                (n_order, self.in_features, self.out_features),
            )
            T_prev = jnp.ones_like(x_safe)   # float32
            T_curr = x_safe                   # float32
            out = out + T_prev.astype(jnp.bfloat16) @ poly_weight[0].astype(jnp.bfloat16)
            if n_order > 1:
                out = out + T_curr.astype(jnp.bfloat16) @ poly_weight[1].astype(jnp.bfloat16)
            for k in range(2, n_order):
                T_next = 2.0 * x_safe * T_curr - T_prev  # float32 arithmetic
                out = out + T_next.astype(jnp.bfloat16) @ poly_weight[k].astype(jnp.bfloat16)
                T_prev = T_curr
                T_curr = T_next

        # RMSNorm (cheaper than LayerNorm: no mean subtraction)
        out = nn.RMSNorm(name='rms_norm')(out.astype(jnp.float32))
        return out.astype(orig_dtype).reshape(prefix + (self.out_features,))


# ---------------------------------------------------------------------------
# KANPredictor — drop-in for predictors/predictor.py::Predictor
# ---------------------------------------------------------------------------

class KANPredictor(nn.Module):
    """KAN-based JEPA stochastic predictor. Drop-in replacement for Predictor.

    API-compatible: identical field names and __call__ signature.
    Dense+GELU layers replaced with KANLayer (B-spline activations).

    Usage::

        # In FinJEPA.from_config or training script:
        predictor = KANPredictor(
            d_model=config.mamba2.d_model,
            hidden_dim=config.predictor.hidden_dim,  # should be multiple of 128
            n_layers=config.predictor.n_layers,
            seq_len=config.embedding.seq_len,
            z_dim=config.predictor.pred_z_dim,
        )
    """
    d_model: int = 128
    hidden_dim: int = 256    # Multiple of 128 for MXU alignment (required)
    n_layers: int = 2
    seq_len: int = 128
    dropout: float = 0.1     # Kept for API compat; KAN uses LayerNorm instead
    z_dim: int = 32
    grid_size: int = 10
    spline_order: int = 3

    @nn.compact
    def __call__(
        self,
        h_x: Array,               # (B, S, d_model)
        target_positions: Array,  # (B, N_tgt) int64
        z: Array | None = None,  # (B, N_tgt, z_dim) or None
        deterministic: bool = False,
    ) -> Array:
        """Predict target representations from context encoder output.

        Returns:
            (B, N_tgt, d_model)
        """
        B = h_x.shape[0]
        N_tgt = target_positions.shape[1]

        # Last hidden state — causally richest (identical to Predictor)
        ctx_mean = h_x[:, -1:, :]   # (B, 1, d_model)

        # Positional embeddings for target positions
        pos = nn.Embed(
            num_embeddings=self.seq_len,
            features=self.d_model,
            name='pos_embed',
        )(target_positions)          # (B, N_tgt, d_model)

        ctx_expanded = jnp.broadcast_to(ctx_mean, (B, N_tgt, self.d_model))

        parts = [ctx_expanded, pos]
        if self.z_dim > 0:
            if z is None:
                z = jnp.zeros((B, N_tgt, self.z_dim), dtype=h_x.dtype)
            parts.append(z)

        x = jnp.concatenate(parts, axis=-1)
        # shape: (B, N_tgt, 2*d_model + z_dim)

        # Build per-layer in/out dims matching Predictor's MLP structure
        in_dim_0 = self.d_model * 2 + self.z_dim
        in_dims = [in_dim_0] + [self.hidden_dim] * (self.n_layers - 1)
        out_dims = [self.hidden_dim] * (self.n_layers - 1) + [self.d_model]

        for i in range(self.n_layers):
            x = KANLayer(
                in_features=in_dims[i],
                out_features=out_dims[i],
                grid_size=self.grid_size,
                spline_order=self.spline_order,
                name=f'kan_{i}',
            )(x)

        return x  # (B, N_tgt, d_model)


# ---------------------------------------------------------------------------
# KANFlowPredictor — drop-in for flow_predictor.py::FlowPredictor
# ---------------------------------------------------------------------------

class KANFlowPredictor(nn.Module):
    """KAN-based OT-CFM velocity field predictor. Drop-in for FlowPredictor.

    Only _forward_velocity is modified (Dense+GELU → KANLayer).
    __call__ (OT coupling + interpolation), SinusoidalTimeEmbedding,
    sinkhorn_coupling, and sample() are byte-for-byte identical to
    FlowPredictor — only the MLP backbone changes.

    Usage::

        flow_pred = KANFlowPredictor(
            d_model=config.mamba2.d_model,
            hidden_dim=256,   # multiple of 128
            n_layers=2,
            seq_len=config.embedding.seq_len,
        )
    """
    d_model: int = 128
    hidden_dim: int = 256    # Multiple of 128 for MXU
    n_layers: int = 2
    seq_len: int = 128
    dropout: float = 0.1     # Kept for API compat
    ot: bool = True
    ot_epsilon: float = 0.05
    ot_sinkhorn_iters: int = 50
    grid_size: int = 10
    spline_order: int = 3

    @nn.compact
    def __call__(
        self,
        h_x: Array,
        target_positions: Array,
        h_y_tgt: Array,
        key: Array,
        deterministic: bool = False,
    ) -> tuple[Array, Array]:
        """OT-CFM training step — identical to FlowPredictor.__call__.

        Returns:
            v_pred: (B, N_tgt, d_model) predicted velocity.
            v_tgt:  (B, N_tgt, d_model) target velocity = h_y_tgt - x_0.
        """
        B = h_y_tgt.shape[0]
        key_t, key_x0 = jax.random.split(key)

        t = jax.random.uniform(key_t, (B,), dtype=h_y_tgt.dtype)
        x_0 = jax.random.normal(key_x0, h_y_tgt.shape, dtype=h_y_tgt.dtype)

        if self.ot and B > 1:
            from .flow_predictor import sinkhorn_coupling
            x0_flat = x_0.reshape(B, -1)
            x1_flat = h_y_tgt.reshape(B, -1)
            perm = sinkhorn_coupling(
                x0_flat, x1_flat, self.ot_epsilon, self.ot_sinkhorn_iters
            )
            x_0 = x_0[perm]

        t_e = t[:, None, None]
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
        """Velocity field v_theta(x_t, t, context) via KAN layers.

        Input tensor: (B, N_tgt, 4*d_model) — same as FlowPredictor.
        Output:       (B, N_tgt, d_model)   — same as FlowPredictor.
        """
        B = x_t.shape[0]
        N_tgt = x_t.shape[1]

        from .flow_predictor import SinusoidalTimeEmbedding
        t_emb = SinusoidalTimeEmbedding(self.d_model, name='time_embed')(t)
        t_emb = jnp.broadcast_to(t_emb[:, None, :], (B, N_tgt, self.d_model))

        ctx = jnp.broadcast_to(h_x[:, -1:, :], (B, N_tgt, self.d_model))

        pos = nn.Embed(
            num_embeddings=self.seq_len,
            features=self.d_model,
            name='pos_embed',
        )(target_positions)

        inp = jnp.concatenate([x_t, t_emb, ctx, pos], axis=-1)
        # (B, N_tgt, 4 * d_model)

        in_dim_0 = 4 * self.d_model
        in_dims = [in_dim_0] + [self.hidden_dim] * (self.n_layers - 1)
        out_dims = [self.hidden_dim] * (self.n_layers - 1) + [self.d_model]

        x = inp
        for i in range(self.n_layers):
            x = KANLayer(
                in_features=in_dims[i],
                out_features=out_dims[i],
                grid_size=self.grid_size,
                spline_order=self.spline_order,
                name=f'vf_kan_{i}',
            )(x)

        return x  # (B, N_tgt, d_model)

    def sample(
        self,
        params: dict,
        h_x: Array,
        target_positions: Array,
        key: Array,
        n_steps: int = 2,
    ) -> Array:
        """Euler ODE integration from x_0 ~ N(0,I) to x_1 via diffrax.

        Identical to FlowPredictor.sample — only the velocity backbone changed.

        Args:
            params:           Flax params dict for this module.
            h_x:              (B, S, d_model) context encoder output.
            target_positions: (B, N_tgt) int64.
            key:              PRNGKey for x_0 sampling.
            n_steps:          Number of Euler steps (2 is sufficient for CFM).

        Returns:
            (B, N_tgt, d_model) sampled future latents.
        """
        B, N_tgt = target_positions.shape
        x0 = jax.random.normal(key, (B, N_tgt, self.d_model), dtype=h_x.dtype)

        def vector_field(t, y, args):
            t_batch = jnp.full((B,), t)
            return self.apply(
                {'params': params},
                y, t_batch, h_x, target_positions, True,
                method=self._forward_velocity,
            )

        import diffrax
        term = diffrax.ODETerm(vector_field)
        solver = diffrax.Euler()
        sol = diffrax.diffeqsolve(
            term, solver,
            t0=0.0, t1=1.0, dt0=1.0 / n_steps,
            y0=x0,
        )
        return sol.ys[-1] if sol.ys.ndim > 3 else sol.ys
