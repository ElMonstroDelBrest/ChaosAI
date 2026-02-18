"""Depthwise causal 1D convolution in Flax.

JAX Conv uses channels-last (B, L, D) by default — no transpose needed.
"""

import jax.numpy as jnp
from flax import linen as nn


class CausalConv1d(nn.Module):
    """Causal 1D depthwise convolution with left padding.

    Input/output: (B, L, D) — channels-last.
    """
    d_inner: int
    kernel_size: int = 4

    @nn.compact
    def __call__(self, x):
        """(B, L, D) -> (B, L, D)."""
        # Left-pad for causal: only see past tokens
        x = jnp.pad(x, ((0, 0), (self.kernel_size - 1, 0), (0, 0)))
        return nn.Conv(
            features=self.d_inner,
            kernel_size=(self.kernel_size,),
            feature_group_count=self.d_inner,  # depthwise
            padding="VALID",
        )(x)
