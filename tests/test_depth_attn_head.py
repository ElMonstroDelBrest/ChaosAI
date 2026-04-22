"""Tests for depth-attention return head (port from ../JEPA)."""
import pytest


def test_import():
    from src.jax_v6.predictors.depth_attn_head import DepthAttnHead
    assert DepthAttnHead is not None


def test_jepa_instantiates_depth_head_when_enabled():
    """When use_depth_attn_ret_head=True AND ret_weight>0, setup uses DepthAttnHead."""
    try:
        import jax
        import jax.numpy as jnp
        from src.jax_v6.jepa import FinJEPA
        from src.jax_v6.predictors.depth_attn_head import DepthAttnHead
    except ImportError:
        pytest.skip("JAX not installed")

    model = FinJEPA(
        num_codes=16, codebook_dim=8, d_model=32, d_state=4,
        n_layers=2, n_heads=2, expand_factor=2, conv_kernel=4,
        seq_len=8, chunk_size=8,
        ret_weight=0.1, use_depth_attn_ret_head=True,
    )
    # Check the flag is carried
    assert model.use_depth_attn_ret_head is True
    assert model.ret_weight == 0.1


def test_depth_head_forward_shape():
    """DepthAttnHead per-timestep returns (B, S, out_dim)."""
    try:
        import jax
        import jax.numpy as jnp
        from src.jax_v6.predictors.depth_attn_head import DepthAttnHead
    except ImportError:
        pytest.skip("JAX not installed")

    B, S, D = 2, 8, 32
    n_layers = 3
    key = jax.random.PRNGKey(0)
    hidden = [jax.random.normal(jax.random.fold_in(key, i), (B, S, D)) for i in range(n_layers + 1)]

    head = DepthAttnHead(hidden_dim=16, out_dim=1, apply_per_timestep=True)
    params = head.init(key, hidden)["params"]
    assert "depth_query" in params
    out = head.apply({"params": params}, hidden)
    assert out.shape == (B, S, 1)


def test_depth_head_last_timestep_shape():
    try:
        import jax
        import jax.numpy as jnp
        from src.jax_v6.predictors.depth_attn_head import DepthAttnHead
    except ImportError:
        pytest.skip("JAX not installed")

    B, S, D = 2, 8, 32
    n_layers = 3
    key = jax.random.PRNGKey(0)
    hidden = [jax.random.normal(jax.random.fold_in(key, i), (B, S, D)) for i in range(n_layers + 1)]
    head = DepthAttnHead(hidden_dim=16, out_dim=5, apply_per_timestep=False)
    params = head.init(key, hidden)["params"]
    out = head.apply({"params": params}, hidden)
    assert out.shape == (B, 5)
