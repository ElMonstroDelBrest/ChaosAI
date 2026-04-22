"""Tests for Mamba-3 block + SSD L x L matrix scan (port from ../JEPA)."""
import pytest


def test_imports():
    from src.jax_v6.encoders.mamba3_block import Mamba3Block
    from src.jax_v6.encoders.ssd_matrix import ssd_matrix_scan
    assert Mamba3Block is not None
    assert ssd_matrix_scan is not None


def test_encoder_type_dispatch():
    """Mamba2Encoder accepts encoder_type='mamba3'."""
    from src.jax_v6.encoders.mamba2_encoder import Mamba2Encoder
    fields = Mamba2Encoder.__dataclass_fields__
    assert "encoder_type" in fields


def test_jepa_encoder_type_field():
    from src.jax_v6.jepa import FinJEPA
    assert "encoder_type" in FinJEPA.__dataclass_fields__


def test_ssd_matrix_scan_shape():
    """Smoke test: output shape matches input shape."""
    try:
        import jax
        import jax.numpy as jnp
        from src.jax_v6.encoders.ssd_matrix import ssd_matrix_scan
    except ImportError:
        pytest.skip("JAX not installed")

    B, L, H, D_h, N = 2, 8, 2, 16, 4
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (B, L, H, D_h))
    dt = jax.nn.softplus(jax.random.normal(jax.random.fold_in(key, 1), (B, L, H)))
    Bm = jax.random.normal(jax.random.fold_in(key, 2), (B, L, N))
    Cm = jax.random.normal(jax.random.fold_in(key, 3), (B, L, N))
    A = -jnp.arange(1, H + 1, dtype=jnp.float32)
    lam = jax.nn.sigmoid(jax.random.normal(jax.random.fold_in(key, 4), (B, L, H)))

    y = ssd_matrix_scan(x, Bm, Cm, dt, A, lam)
    assert y.shape == (B, L, H, D_h)
    assert jnp.all(jnp.isfinite(y))


def test_mamba3_block_forward():
    try:
        import jax
        import jax.numpy as jnp
        from src.jax_v6.encoders.mamba3_block import Mamba3Block
    except ImportError:
        pytest.skip("JAX not installed")

    B, L, d_model = 2, 8, 32
    block = Mamba3Block(d_model=d_model, d_state=8, n_heads=2, expand_factor=2)
    key = jax.random.PRNGKey(0)
    u = jax.random.normal(key, (B, L, d_model))
    params = block.init(key, u)["params"]
    # Expected params: in_proj, norm, bc_norm_B, bc_norm_C, theta_proj, out_proj + A_log, dt_bias, B_bias, C_bias, D
    for name in ("A_log", "dt_bias", "B_bias", "C_bias", "D"):
        assert name in params, f"missing param {name}"
    y = block.apply({"params": params}, u)
    assert y.shape == (B, L, d_model)
    assert jnp.all(jnp.isfinite(y))


def test_mamba3_encoder_e2e():
    """Full Mamba2Encoder with encoder_type='mamba3' runs and returns shape (B, S, D)."""
    try:
        import jax
        import jax.numpy as jnp
        from src.jax_v6.encoders.mamba2_encoder import Mamba2Encoder
    except ImportError:
        pytest.skip("JAX not installed")

    enc = Mamba2Encoder(
        num_codes=16, codebook_dim=8, d_model=32, d_state=8,
        n_layers=2, n_heads=2, expand_factor=2, conv_kernel=4,
        seq_len=8, chunk_size=8, encoder_type="mamba3",
    )
    B, S = 2, 8
    key = jax.random.PRNGKey(0)
    tokens = jnp.zeros((B, S), dtype=jnp.int32)
    params = enc.init(key, tokens)["params"]
    out = enc.apply({"params": params}, tokens)
    assert out.shape == (B, S, 32)
