"""Test risk-parity reward normalization in cross-sectional env."""
import numpy as np


def _step_rewards_np(next_returns, vol_t, weights, prev_weights,
                     risk_parity=False, fee_rate=0.0008, slippage=0.001):
    """Pure numpy reimplementation of step_portfolio reward logic."""
    rets = next_returns.copy()
    if risk_parity:
        rets = rets / (vol_t + 1e-8)
    mean_ret = rets.mean()
    std_ret = rets.std() + 1e-8
    z = (rets - mean_ret) / std_ret
    raw_alpha = weights * z
    delta = weights - prev_weights
    fee = fee_rate * np.abs(delta) + slippage * delta ** 2
    return raw_alpha - fee


def test_risk_parity_output_finite():
    """Risk-parity rewards must be finite floats."""
    rng = np.random.default_rng(0)
    K = 16
    high_vol_ret = rng.standard_normal(K).astype(np.float32) * 0.05
    high_vol_t = np.abs(high_vol_ret) + 0.01
    weights = rng.standard_normal(K).astype(np.float32)
    weights /= np.abs(weights).sum()

    r_no_rp = _step_rewards_np(high_vol_ret, high_vol_t, weights, weights * 0, risk_parity=False)
    r_rp = _step_rewards_np(high_vol_ret, high_vol_t, weights, weights * 0, risk_parity=True)

    assert np.all(np.isfinite(r_no_rp)), "Non-finite without risk_parity"
    assert np.all(np.isfinite(r_rp)), "Non-finite with risk_parity"


def test_risk_parity_flag_in_env():
    """env_cross_sectional.py must have risk_parity parameter in step_portfolio."""
    import pathlib
    src = pathlib.Path("src/jax_v6/strate_iv/env_cross_sectional.py").read_text()
    assert "risk_parity" in src, "Missing risk_parity in step_portfolio"
    assert "vol_t = vol_wins[:, t]" in src or "vol_t = vol_wins" in src, "Missing vol_t normalization"


def test_config_has_risk_parity():
    """CrossSectionalConfig must have risk_parity field."""
    from src.jax_v6.config import CrossSectionalConfig
    cfg = CrossSectionalConfig()
    assert hasattr(cfg, "risk_parity")
    assert cfg.risk_parity is False
