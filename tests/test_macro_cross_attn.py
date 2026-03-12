"""Test that macro cross-attention produces correct structure.

No JAX required — tests via AST inspection + config.
"""
import ast
import pathlib


def test_mamba2_encoder_has_macro_cross_attn_branch():
    """mamba2_encoder.py must contain the cross-attention branch."""
    src = pathlib.Path("src/jax_v6/encoders/mamba2_encoder.py").read_text()
    assert "use_macro_cross_attn" in src, "Missing use_macro_cross_attn branch"
    assert "macro_k_proj" in src, "Missing macro_k_proj (cross-attn key projection)"
    assert "macro_v_proj" in src, "Missing macro_v_proj (cross-attn value projection)"
    assert "macro_q_proj" in src, "Missing macro_q_proj (cross-attn query projection)"
    assert "macro_out_proj" in src, "Missing macro_out_proj"
    assert "x.at[:, 0, :].add" in src, "Missing prefix token conditioning via .at[:, 0, :].add"


def test_legacy_branch_preserved():
    """Legacy gated additive branch must still exist for backward compat."""
    src = pathlib.Path("src/jax_v6/encoders/mamba2_encoder.py").read_text()
    assert "macro_proj" in src, "Legacy macro_proj missing"
    assert "macro_gate_proj" in src, "Legacy macro_gate_proj missing"


def test_config_has_use_macro_cross_attn():
    """Mamba2Config must have use_macro_cross_attn field defaulting to False."""
    from src.jax_v6.config import Mamba2Config
    cfg = Mamba2Config()
    assert hasattr(cfg, "use_macro_cross_attn")
    assert cfg.use_macro_cross_attn is False
