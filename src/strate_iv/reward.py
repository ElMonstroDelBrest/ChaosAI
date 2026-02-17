"""PnL reward with transaction costs for Strate IV RL.

reward = action * log_return(close_next / close_current) - tc_rate * |Δaction|

The reward is symmetric for long/short — no volatility scaling. The agent
sees raw market dynamics and can learn to follow trends or cut losses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import EPS


@dataclass
class RewardInfo:
    """Breakdown of a single-step reward.

    Args:
        raw_pnl: Directional PnL component (action * log_return).
        tc_penalty: Transaction cost penalty (tc_rate * |Δaction|).
        log_return: Log-return of the underlying close price.
    """

    raw_pnl: float
    tc_penalty: float
    log_return: float


class PnLReward:
    """Compute step reward as position-weighted log-return minus turnover cost.

    Args:
        tc_rate: Transaction cost per unit of position change (default 5 bps).
    """

    def __init__(self, tc_rate: float = 0.0005) -> None:
        self.tc_rate = tc_rate

    def compute(
        self,
        action: float,
        prev_action: float,
        close_current: float,
        close_next: float,
    ) -> tuple[float, RewardInfo]:
        """Compute reward for a single step.

        Args:
            action: Current position a_t in [-1, 1].
            prev_action: Previous position a_{t-1} in [-1, 1].
            close_current: Close price at current step.
            close_next: Close price at next step.

        Returns:
            (reward, info) tuple.
        """
        log_ret = math.log(max(close_next, EPS) / max(close_current, EPS))
        raw_pnl = action * log_ret
        tc_penalty = self.tc_rate * abs(action - prev_action)
        reward = raw_pnl - tc_penalty
        return reward, RewardInfo(
            raw_pnl=raw_pnl,
            tc_penalty=tc_penalty,
            log_return=log_ret,
        )


# Backward-compatible alias
AsymmetricReward = PnLReward
