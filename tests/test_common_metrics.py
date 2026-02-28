"""Tests for src/common/metrics — pure numpy, no framework deps."""

import numpy as np
import pytest

from src.common.metrics import compute_sharpe, compute_max_drawdown, compute_cum_return


class TestComputeSharpe:
    def test_positive_returns(self):
        rets = [0.01, 0.02, 0.01, 0.03, 0.01]
        s = compute_sharpe(rets, annualize=1.0)
        assert s > 0

    def test_zero_vol(self):
        rets = [0.01, 0.01, 0.01]
        assert compute_sharpe(rets) == 0.0

    def test_empty(self):
        assert compute_sharpe([]) == 0.0

    def test_annualize_factor(self):
        rets = [0.01, -0.005, 0.02, -0.01, 0.015]
        s1 = compute_sharpe(rets, annualize=1.0)
        s2 = compute_sharpe(rets, annualize=np.sqrt(252))
        assert abs(s2 / s1 - np.sqrt(252)) < 1e-6

    def test_negative_returns(self):
        rets = [-0.01, -0.02, -0.01]
        assert compute_sharpe(rets) < 0


class TestComputeMaxDrawdown:
    def test_monotonic_up(self):
        rets = [0.01, 0.02, 0.03]
        assert compute_max_drawdown(rets) == 0.0

    def test_drawdown(self):
        rets = [0.10, -0.20, 0.05]
        dd = compute_max_drawdown(rets)
        assert dd < 0
        assert dd > -1.0

    def test_empty(self):
        assert compute_max_drawdown([]) == 0.0

    def test_large_loss(self):
        # cumprod([0.5, 0.35]), peak=[0.5, 0.5], dd=[0, -0.3]
        rets = [-0.5, -0.3]
        dd = compute_max_drawdown(rets)
        assert dd == pytest.approx(-0.3)


class TestComputeCumReturn:
    def test_simple(self):
        rets = [0.10, 0.10]
        expected = 1.1 * 1.1 - 1  # 0.21
        assert compute_cum_return(rets) == pytest.approx(expected, abs=1e-10)

    def test_zero(self):
        assert compute_cum_return([0.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_negative(self):
        assert compute_cum_return([-0.5]) == pytest.approx(-0.5)

    def test_empty(self):
        # prod of empty = 1, so 1-1 = 0
        assert compute_cum_return([]) == pytest.approx(0.0)

    def test_numpy_array_input(self):
        rets = np.array([0.01, 0.02, -0.01])
        r = compute_cum_return(rets)
        assert isinstance(r, float)
