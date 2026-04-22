"""Tests for AttnRes port from ../JEPA into Mamba2Encoder."""
import pytest


def test_config_has_use_attn_res():
    from src.jax_v6.config import Mamba2Config
    cfg = Mamba2Config()
    assert cfg.use_attn_res is False
    assert cfg.use_depth_attn_ret_head is False


def test_encoder_has_attn_res_fields():
    from src.jax_v6.encoders.mamba2_encoder import Mamba2Encoder
    # dataclass fields — use Flax annotations
    fields = Mamba2Encoder.__dataclass_fields__
    assert "use_attn_res" in fields
    assert "return_hidden_states" in fields


def test_jepa_has_attn_res_fields():
    from src.jax_v6.jepa import FinJEPA
    fields = FinJEPA.__dataclass_fields__
    assert "use_attn_res" in fields
    assert "use_depth_attn_ret_head" in fields


def test_attn_res_forward():
    """Smoke test: encoder with AttnRes on runs and returns correct shape."""
    try:
        import jax
        import jax.numpy as jnp
        from src.jax_v6.encoders.mamba2_encoder import Mamba2Encoder
    except ImportError:
        pytest.skip("JAX not installed — AST-only validation")

    enc = Mamba2Encoder(
        num_codes=16, codebook_dim=8, d_model=32, d_state=4,
        n_layers=3, n_heads=2, expand_factor=2, conv_kernel=4,
        seq_len=16, chunk_size=16, use_attn_res=True,
    )
    B, S = 2, 16
    key = jax.random.PRNGKey(0)
    tokens = jnp.zeros((B, S), dtype=jnp.int32)
    params = enc.init(key, tokens)["params"]
    assert "attn_queries" in params
    assert "attnres_key_norm" in params
    out = enc.apply({"params": params}, tokens)
    assert out.shape == (B, S, 32)


def test_return_hidden_states():
    """When return_hidden_states=True, encoder returns (out, [h_0..h_N])."""
    try:
        import jax
        import jax.numpy as jnp
        from src.jax_v6.encoders.mamba2_encoder import Mamba2Encoder
    except ImportError:
        pytest.skip("JAX not installed")

    enc = Mamba2Encoder(
        num_codes=16, codebook_dim=8, d_model=32, d_state=4,
        n_layers=3, n_heads=2, expand_factor=2, conv_kernel=4,
        seq_len=16, chunk_size=16, return_hidden_states=True,
    )
    B, S = 2, 16
    key = jax.random.PRNGKey(0)
    tokens = jnp.zeros((B, S), dtype=jnp.int32)
    params = enc.init(key, tokens)["params"]
    out, hidden = enc.apply({"params": params}, tokens)
    assert out.shape == (B, S, 32)
    # h_0 (pre-blocks) + N layer outputs
    assert len(hidden) == 4
    for h in hidden:
        assert h.shape == (B, S, 32)
