"""Mamba-2 block with chunked SSD, weekend gating, and volume-clock modulation.

Port of strate_ii/mamba2_block.py:249-376 to Flax nn.Module.

Key differences from PyTorch:
  - A_log is a Flax param with custom initializer
  - vol_proj, exo_proj are nn.Dense with zeros init
  - Uses chunked_ssd() instead of selective_scan()
  - All shapes are channels-last (B, L, D) — no transposes for conv
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import Array

from .causal_conv import CausalConv1d
from .ssd import chunked_ssd


class Mamba2Block(nn.Module):
    """Single Mamba-2 block with chunked SSD and weekend gating.

    Architecture:
        x -> LayerNorm -> Linear -> split(x_branch, z_gate, B, C, dt)
        x_branch -> CausalConv1d -> SiLU -> chunked_ssd -> y
        y = y * SiLU(z_gate)
        y -> Linear -> + residual
    """
    d_model: int = 128
    d_state: int = 16
    n_heads: int = 2
    expand_factor: int = 2
    conv_kernel: int = 4
    chunk_size: int = 128

    def setup(self):
        self.d_inner = self.d_model * self.expand_factor
        self.head_dim = self.d_inner // self.n_heads
        assert self.d_inner % self.n_heads == 0, "d_inner must be divisible by n_heads"

        self.in_proj_size = (
            self.d_inner          # x_branch
            + self.d_inner        # z_gate
            + self.n_heads * self.d_state  # B
            + self.n_heads * self.d_state  # C
            + self.n_heads        # dt
        )

    @nn.compact
    def __call__(
        self,
        x: Array,
        weekend_mask: Array | None = None,
        vol_clock: Array | None = None,
        exo_clock: Array | None = None,
    ) -> Array:
        """Forward pass with optional weekend gating and clock modulation.

        Args:
            x: (B, L, d_model) input sequence.
            weekend_mask: (B, L) float {0.0, 1.0} where 1.0 = weekend.
            vol_clock: (B, L) float endogenous L2 volatility proxy.
            exo_clock: (B, L, 2) float exogenous [RV, Volume] clock signals.

        Returns:
            (B, L, d_model) output with residual connection.
        """
        d_inner = self.d_model * self.expand_factor
        head_dim = d_inner // self.n_heads

        residual = x
        x = nn.LayerNorm()(x)

        # Input projection
        in_proj_size = (
            d_inner + d_inner
            + self.n_heads * self.d_state
            + self.n_heads * self.d_state
            + self.n_heads
        )
        proj = nn.Dense(in_proj_size, use_bias=False, name="in_proj")(x)

        # Split projections
        idx = 0
        x_branch = proj[..., idx:idx + d_inner]
        idx += d_inner
        z_gate = proj[..., idx:idx + d_inner]
        idx += d_inner
        B_proj = proj[..., idx:idx + self.n_heads * self.d_state]
        idx += self.n_heads * self.d_state
        C_proj = proj[..., idx:idx + self.n_heads * self.d_state]
        idx += self.n_heads * self.d_state
        dt = proj[..., idx:idx + self.n_heads]

        # A_log parameter: log(arange(1, d_state+1)) per head
        def a_log_init(key, shape, dtype=jnp.float32):
            return jnp.broadcast_to(
                jnp.log(jnp.arange(1, self.d_state + 1, dtype=dtype))[None, :],
                shape,
            )
        A_log = self.param("A_log", a_log_init, (self.n_heads, self.d_state))

        # Clock modulation: bias dt (pre-softplus) to adjust temporal grain
        # Exogenous clock takes priority over endogenous vol_clock
        if exo_clock is not None:
            exo_bias = nn.Dense(
                self.n_heads, use_bias=True, name="exo_proj",
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
            )(exo_clock)  # (B, L, 2) -> (B, L, n_heads)
            dt = dt + exo_bias
        elif vol_clock is not None:
            vol_bias = nn.Dense(
                self.n_heads, use_bias=True, name="vol_proj",
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
            )(vol_clock[..., None])  # (B, L, 1) -> (B, L, n_heads)
            dt = dt + vol_bias

        # Reshape B, C: (B, L, n_heads * d_state) -> (B, L, n_heads, d_state)
        B_proj = B_proj.reshape(*B_proj.shape[:-1], self.n_heads, self.d_state)
        C_proj = C_proj.reshape(*C_proj.shape[:-1], self.n_heads, self.d_state)

        # Causal conv on x_branch: (B, L, D) -> (B, L, D), channels-last
        x_branch = CausalConv1d(d_inner, self.conv_kernel, name="conv")(x_branch)
        x_branch = jax.nn.silu(x_branch)

        # Reshape x_branch for SSD: (B, L, D) -> (B, L, H, P)
        x_heads = x_branch.reshape(*x_branch.shape[:-1], self.n_heads, head_dim)

        # Chunked SSD
        y = chunked_ssd(
            x_heads, dt, A_log, B_proj, C_proj,
            weekend_mask=weekend_mask,
            chunk_size=self.chunk_size,
        )  # (B, L, H, P)

        # Reshape back: (B, L, H, P) -> (B, L, D)
        y = y.reshape(*y.shape[:-2], d_inner)

        # Gated output
        y = y * jax.nn.silu(z_gate)

        # Output projection + residual
        y = nn.Dense(self.d_model, use_bias=False, name="out_proj")(y)
        return y + residual
