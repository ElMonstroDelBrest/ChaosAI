"""SynapticVector — signal from Strate IV agent to execution layer."""

from dataclasses import dataclass, field
import time
import uuid


@dataclass
class SynapticVector:
    """Signal carrying trade intent from brain to execution.

    Attributes:
        pair: Trading pair (e.g. 'BTCUSDT').
        target_pos: Target position in [-1, 1] (short to long).
        urgency: Execution urgency in [0, 1] (0=patient TWAP, 1=immediate).
        ttl: Time-to-live in seconds (signal expires after this).
        confidence: Model confidence in [0, 1].
        convergence_score: Multiverse convergence from Strate IV.
        trace_id: Unique ID for observability.
        timestamp: Creation time (auto-filled).
    """
    pair: str
    target_pos: float
    urgency: float = 0.5
    ttl: float = 60.0
    confidence: float = 0.5
    convergence_score: float = 0.5
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)

    def is_expired(self, now: float | None = None) -> bool:
        if now is None:
            now = time.time()
        return (now - self.timestamp) > self.ttl

    def to_dict(self) -> dict:
        return {
            "pair": self.pair,
            "target_pos": self.target_pos,
            "urgency": self.urgency,
            "ttl": self.ttl,
            "confidence": self.confidence,
            "convergence_score": self.convergence_score,
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
        }
