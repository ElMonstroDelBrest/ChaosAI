"""JEPA encoder factory — creates Mamba2Encoder from config and wraps encode_batch."""

import functools


def create_encoder_from_config(config):
    """Instantiate a Mamba2Encoder from a StrateIIConfig.

    Args:
        config: StrateIIConfig (JAX).

    Returns:
        (encoder, encode_batch_fn) where encode_batch_fn(target_params, token_indices, exo_clock)
        returns (B, 2*d_model) embeddings [mean_pool || last_token].
    """
    import jax
    import jax.numpy as jnp
    from src.jax_v6.encoders.mamba2_encoder import Mamba2Encoder

    d_model = config.mamba2.d_model
    encoder = Mamba2Encoder(
        num_codes=config.embedding.num_codes,
        codebook_dim=config.embedding.codebook_dim,
        d_model=d_model,
        d_state=config.mamba2.d_state,
        n_layers=config.mamba2.n_layers,
        n_heads=config.mamba2.n_heads,
        expand_factor=config.mamba2.expand_factor,
        conv_kernel=config.mamba2.conv_kernel,
        seq_len=config.embedding.seq_len,
        chunk_size=config.mamba2.chunk_size,
        use_remat=config.mamba2.use_remat,
        gnn_dim=config.mamba2.gnn_dim,
    )

    @jax.jit
    def encode_batch(target_params, token_indices, exo_clock):
        h = encoder.apply(
            {"params": target_params},
            token_indices,
            weekend_mask=None,
            block_mask=None,
            exo_clock=exo_clock,
        )
        return jnp.concatenate([h.mean(axis=1), h[:, -1, :]], axis=-1)

    return encoder, encode_batch
