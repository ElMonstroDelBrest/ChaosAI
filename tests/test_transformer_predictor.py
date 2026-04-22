"""Tests for asymmetric Transformer JEPA predictor (port from ../JEPA)."""
import pytest


def test_import():
    from src.jax_v6.predictors.transformer_predictor import (
        TransformerPredictor, PredictorBlock,
    )
    assert TransformerPredictor is not None
    assert PredictorBlock is not None


def test_config_fields():
    from src.jax_v6.config import PredictorConfig
    cfg = PredictorConfig()
    assert cfg.predictor_type == "mlp"
    assert cfg.d_pred == 64
    assert cfg.n_pred_layers == 6
    assert cfg.n_pred_heads == 4
    assert cfg.pred_use_attn_res is True


def test_jepa_fields():
    from src.jax_v6.jepa import FinJEPA
    fields = FinJEPA.__dataclass_fields__
    for name in ("predictor_type", "d_pred", "n_pred_layers", "n_pred_heads", "pred_use_attn_res"):
        assert name in fields


def test_predictor_block_forward():
    try:
        import jax
        import jax.numpy as jnp
        from src.jax_v6.predictors.transformer_predictor import PredictorBlock
    except ImportError:
        pytest.skip("JAX not installed")

    B, Tq, Tk, d_pred = 2, 8, 16, 32
    block = PredictorBlock(d_pred=d_pred, n_heads=4)
    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (B, Tq, d_pred))
    kv = jax.random.normal(jax.random.fold_in(key, 1), (B, Tk, d_pred))
    params = block.init(key, q, kv)["params"]
    out = block.apply({"params": params}, q, kv)
    assert out.shape == (B, Tq, d_pred)


def test_transformer_predictor_shape():
    try:
        import jax
        import jax.numpy as jnp
        from src.jax_v6.predictors.transformer_predictor import TransformerPredictor
    except ImportError:
        pytest.skip("JAX not installed")

    B, S, d_model = 2, 16, 64
    pred = TransformerPredictor(
        d_model=d_model, d_pred=32, tgt_len=S, n_layers=2, n_heads=4, use_attn_res=True,
    )
    key = jax.random.PRNGKey(0)
    ctx = jax.random.normal(key, (B, S, d_model))
    params = pred.init(key, ctx)["params"]
    assert "pos_queries" in params
    assert "attnres_queries" in params
    out = pred.apply({"params": params}, ctx)
    assert out.shape == (B, S, d_model)


def test_transformer_predictor_attn_res_off():
    """Without AttnRes, attnres_queries/keynorm params should be absent."""
    try:
        import jax
        import jax.numpy as jnp
        from src.jax_v6.predictors.transformer_predictor import TransformerPredictor
    except ImportError:
        pytest.skip("JAX not installed")

    pred = TransformerPredictor(
        d_model=32, d_pred=16, tgt_len=8, n_layers=2, n_heads=2, use_attn_res=False,
    )
    key = jax.random.PRNGKey(0)
    ctx = jax.random.normal(key, (2, 8, 32))
    params = pred.init(key, ctx)["params"]
    assert "attnres_queries" not in params
    out = pred.apply({"params": params}, ctx)
    assert out.shape == (2, 8, 32)
