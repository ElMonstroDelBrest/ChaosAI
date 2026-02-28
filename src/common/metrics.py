"""Trading performance metrics — numpy only, no framework deps."""

import numpy as np


def compute_sharpe(returns, annualize=1.0):
    """Annualized Sharpe ratio from a 1-D array of period returns.

    Args:
        returns: Array-like of period returns.
        annualize: Annualization factor (e.g. sqrt(252*24*60/32) for 32-min windows).

    Returns:
        float: Sharpe ratio (0.0 if insufficient data or zero vol).
    """
    returns = np.asarray(returns, dtype=np.float64)
    if len(returns) == 0 or np.std(returns) < 1e-12:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * annualize)


def compute_max_drawdown(returns):
    """Maximum drawdown from a 1-D array of period returns.

    Returns:
        float: Worst peak-to-trough decline (negative number, 0.0 if empty).
    """
    returns = np.asarray(returns, dtype=np.float64)
    if len(returns) == 0:
        return 0.0
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / np.maximum(peak, 1e-15)
    return float(np.min(dd))


def compute_cum_return(returns):
    """Cumulative return from a 1-D array of period returns.

    Returns:
        float: Total compounded return.
    """
    returns = np.asarray(returns, dtype=np.float64)
    return float(np.prod(1 + returns) - 1)
