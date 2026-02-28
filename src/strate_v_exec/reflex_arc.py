"""ReflexArc — flash crash circuit breaker."""

import time
import logging
from collections import deque

log = logging.getLogger(__name__)


class ReflexArc:
    """Detects flash crashes via rolling price window.

    If price drops more than `threshold` (default 5%) within
    `window_seconds`, triggers circuit breaker.
    """

    def __init__(self, threshold: float = 0.05, window_seconds: float = 60.0):
        self.threshold = threshold
        self.window_seconds = window_seconds
        self._prices: deque[tuple[float, float]] = deque()  # (timestamp, price)
        self.triggered = False
        self.trigger_count = 0

    def update(self, price: float, now: float | None = None) -> bool:
        """Feed a new price tick. Returns True if circuit breaker triggers.

        Args:
            price: Current price.
            now: Timestamp (default: time.time()).

        Returns:
            True if flash crash detected (circuit breaker active).
        """
        if now is None:
            now = time.time()

        self._prices.append((now, price))

        # Evict old prices outside window
        cutoff = now - self.window_seconds
        while self._prices and self._prices[0][0] < cutoff:
            self._prices.popleft()

        if len(self._prices) < 2:
            return False

        # Check max price in window vs current
        max_price = max(p for _, p in self._prices)
        if max_price <= 0:
            return False

        drop = (max_price - price) / max_price
        if drop >= self.threshold:
            if not self.triggered:
                log.warning("FLASH CRASH detected: %.2f%% drop (threshold=%.2f%%)",
                            drop * 100, self.threshold * 100)
                self.trigger_count += 1
            self.triggered = True
            return True

        self.triggered = False
        return False

    def reset(self) -> None:
        """Reset the circuit breaker state."""
        self._prices.clear()
        self.triggered = False
