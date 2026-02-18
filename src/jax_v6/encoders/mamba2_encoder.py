"""Mamba-2 Encoder: token embedding + positional embedding + Mamba-2 stack.

Port of strate_ii/encoder.py:20-175 to Flax nn.Module.

Key differences from PyTorch:
  - Codebook: nn.Embed + jax.lax.stop_gradient for freezing
  - Pos embed: sinusoidal constant (not a param, not a buffer — just jnp.array)
  - vol_clock: computed with stop_gradient (no torch.no_grad context)
  - load_codebook: direct variable assignment
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import Array

from .mamba2_block import Mamba2Block


class Mamba2Encoder(nn.Module):
    """Encoder stack for Fin-JEPA.

    Embeds token indices using frozen codebook, projects to d_model,
    adds sinusoidal positional embeddings, then applies N Mamba-2 blocks.
    """
    num_codes: int = 1024
    codebook_dim: int = 64
    d_model: int = 128
    d_state: int = 16
    n_layers: int = 6
    n_heads: int = 2
    expand_factor: int = 2
    conv_kernel: int = 4
    seq_len: int = 128
    chunk_size: int = 128

    @staticmethod
    def _sinusoidal_embed(seq_len: int, d_model: int) -> Array:
        """Generate sinusoidal positional embeddings."""
        pos = jnp.arange(seq_len, dtype=jnp.float32)[:, None]
        dim = jnp.arange(0, d_model, 2, dtype=jnp.float32)
        angle = pos / (10000.0 ** (dim / d_model))
        embed = jnp.zeros((seq_len, d_model))
        embed = embed.at[:, 0::2].set(jnp.sin(angle))
        embed = embed.at[:, 1::2].set(jnp.cos(angle))
        return embed[None, :, :]  # (1, S, d_model)

    @staticmethod
    def _compute_vol_clock(x_embed: Array) -> Array:
        """Compute realized-volatility proxy from codebook embedding transitions.

        vol_clock_t = ||embed_t - embed_{t-1}||_2, z-scored per sequence.
        Detached from gradient via stop_gradient.

        Args:
            x_embed: (B, S, codebook_dim) raw codebook embeddings.

        Returns:
            (B, S) float32 z-scored volatility proxy.
        """
        diffs = x_embed[:, 1:, :] - x_embed[:, :-1, :]
        dist = jnp.sqrt(jnp.sum(diffs ** 2, axis=-1))  # (B, S-1)

        # Prepend sequence mean for position 0
        dist0 = jnp.concatenate(
            [jnp.mean(dist, axis=1, keepdims=True), dist], axis=1
        )  # (B, S)

        # Z-score per sequence
        mu = jnp.mean(dist0, axis=1, keepdims=True)
        sigma = jnp.maximum(jnp.std(dist0, axis=1, keepdims=True), 1e-6)
        return jax.lax.stop_gradient((dist0 - mu) / sigma)

    @nn.compact
    def __call__(
        self,
        token_indices: Array,
        weekend_mask: Array | None = None,
        block_mask: Array | None = None,
        exo_clock: Array | None = None,
    ) -> Array:
        """Encode a sequence of token indices.

        Args:
            token_indices: (B, S) int64 token indices [0, K-1].
            weekend_mask: (B, S) float {0.0, 1.0} weekend indicator.
            block_mask: (B, S) bool where True = masked (target) positions.
            exo_clock: (B, S, 2) float exogenous [RV, Volume] clock signals.

        Returns:
            (B, S, d_model) encoded representations.
        """
        B, S = token_indices.shape

        # Token embedding: frozen codebook lookup + projection
        codebook = nn.Embed(
            num_embeddings=self.num_codes,
            features=self.codebook_dim,
            name="codebook_embed",
        )
        x_embed = jax.lax.stop_gradient(codebook(token_indices))  # (B, S, codebook_dim)
        x = nn.Dense(self.d_model, name="input_proj")(x_embed)  # (B, S, d_model)

        # Clock signals: exogenous takes priority, L2 is fallback
        vol_clock = None
        if exo_clock is None:
            vol_clock = self._compute_vol_clock(x_embed)  # (B, S)

        # Learned [MASK] embedding for masked positions
        mask_embed = self.param(
            "mask_embed",
            nn.initializers.normal(stddev=0.02),
            (self.d_model,),
        )

        # Replace masked positions with [MASK] embedding
        if block_mask is not None:
            mask_expanded = block_mask[..., None].astype(x.dtype)  # (B, S, 1)
            x = x * (1.0 - mask_expanded) + mask_embed * mask_expanded

        # Add sinusoidal positional embedding (constant, not learned)
        pos_embed = self._sinusoidal_embed(self.seq_len, self.d_model)
        x = x + pos_embed[:, :S, :]

        # Pass through Mamba-2 stack with clock conditioning
        for i in range(self.n_layers):
            x = Mamba2Block(
                d_model=self.d_model,
                d_state=self.d_state,
                n_heads=self.n_heads,
                expand_factor=self.expand_factor,
                conv_kernel=self.conv_kernel,
                chunk_size=self.chunk_size,
                name=f"layers_{i}",
            )(x, weekend_mask=weekend_mask, vol_clock=vol_clock, exo_clock=exo_clock)

        # Final layer norm
        return nn.LayerNorm(name="norm")(x)
