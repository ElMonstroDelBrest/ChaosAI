"""Stochastic MLP predictor for Fin-JEPA in JAX/Flax.

Port of strate_ii/predictor.py. Takes context encoder output at target
positions, optional latent noise z, and predicts target representations.

Noise z is passed via jax.random.normal(key, shape) from the training loop.
"""

import jax.numpy as jnp
from flax import linen as nn
from jax import Array


class Predictor(nn.Module):
    """Stochastic MLP predictor: maps context representations at target
    positions (+ optional noise z) to predicted target representations.
    """
    d_model: int = 128
    hidden_dim: int = 256
    n_layers: int = 2
    seq_len: int = 128
    dropout: float = 0.1
    z_dim: int = 32

    @nn.compact
    def __call__(
        self,
        h_x: Array,                # (B, S, d_model)
        target_positions: Array,    # (B, N_tgt) int64
        z: Array | None = None,    # (B, N_tgt, z_dim) or None
        deterministic: bool = False,
    ) -> Array:
        """Predict target representations from context encoder output.

        Args:
            h_x: (B, S, d_model) context encoder output (full sequence).
            target_positions: (B, N_tgt) int64 indices of target positions.
            z: (B, N_tgt, z_dim) latent noise. None -> zeros (deterministic).
            deterministic: If True, disable dropout.

        Returns:
            (B, N_tgt, d_model) predicted target representations.
        """
        B = h_x.shape[0]
        N_tgt = target_positions.shape[1]

        # Use last hidden state (causally richest)
        ctx_mean = h_x[:, -1:, :]  # (B, 1, d_model)

        # Positional embeddings for target positions
        pos = nn.Embed(
            num_embeddings=self.seq_len,
            features=self.d_model,
            name="pos_embed",
        )(target_positions)  # (B, N_tgt, d_model)

        # Expand context to match target positions
        ctx_expanded = jnp.broadcast_to(ctx_mean, (B, N_tgt, self.d_model))

        # Concatenate context + position [+ noise]
        parts = [ctx_expanded, pos]

        if self.z_dim > 0:
            if z is None:
                z = jnp.zeros((B, N_tgt, self.z_dim), dtype=h_x.dtype)
            parts.append(z)

        x = jnp.concatenate(parts, axis=-1)  # (B, N_tgt, 2*d_model + z_dim)

        # MLP layers
        in_dim = self.d_model * 2 + self.z_dim
        for i in range(self.n_layers):
            out_dim = self.hidden_dim if i < self.n_layers - 1 else self.d_model
            x = nn.Dense(out_dim, name=f"mlp_{i}")(x)
            if i < self.n_layers - 1:
                x = nn.gelu(x)
                x = nn.Dropout(rate=self.dropout, deterministic=deterministic)(x)

        return x  # (B, N_tgt, d_model)
