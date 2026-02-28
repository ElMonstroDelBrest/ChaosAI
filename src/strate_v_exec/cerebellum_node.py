"""CerebellumNode — smooth execution via TWAP with cancellation."""

import time
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class TWAPState:
    """State for an in-progress TWAP execution."""
    pair: str
    target_pos: float
    start_pos: float
    n_slices: int
    slices_done: int = 0
    cancelled: bool = False
    trace_id: str = ""
    start_time: float = field(default_factory=time.time)

    @property
    def progress(self) -> float:
        if self.n_slices == 0:
            return 1.0
        return self.slices_done / self.n_slices

    @property
    def current_pos(self) -> float:
        if self.n_slices == 0:
            return self.target_pos
        frac = self.slices_done / self.n_slices
        return self.start_pos + frac * (self.target_pos - self.start_pos)


class CerebellumNode:
    """Manages smooth order execution via TWAP slicing.

    Converts SynapticVectors into TWAP schedules.
    Urgency maps to number of slices: urgency=1 -> 1 slice (immediate),
    urgency=0 -> max_slices.
    """

    def __init__(self, max_slices: int = 10):
        self.max_slices = max_slices
        self.active_tasks: dict[str, TWAPState] = {}

    def submit(self, signal, current_pos: float = 0.0) -> TWAPState:
        """Create a TWAP execution from a SynapticVector.

        Cancels any existing task for the same pair.
        """
        # Cancel existing task for this pair
        if signal.pair in self.active_tasks:
            self.active_tasks[signal.pair].cancelled = True
            log.info("Cancelled previous task for %s", signal.pair)

        n_slices = max(1, int(self.max_slices * (1.0 - signal.urgency)))
        state = TWAPState(
            pair=signal.pair,
            target_pos=signal.target_pos,
            start_pos=current_pos,
            n_slices=n_slices,
            trace_id=signal.trace_id,
        )
        self.active_tasks[signal.pair] = state
        log.info("TWAP %s: %.2f -> %.2f in %d slices (trace=%s)",
                 signal.pair, current_pos, signal.target_pos, n_slices, signal.trace_id)
        return state

    def tick(self, pair: str) -> float | None:
        """Advance one TWAP slice for a pair.

        Returns the new target position for this slice, or None if done/cancelled.
        """
        state = self.active_tasks.get(pair)
        if state is None:
            return None
        if state.cancelled:
            del self.active_tasks[pair]
            return None
        if state.slices_done >= state.n_slices:
            del self.active_tasks[pair]
            return None

        state.slices_done += 1
        pos = state.current_pos
        log.debug("TWAP %s slice %d/%d -> pos=%.4f", pair, state.slices_done, state.n_slices, pos)

        if state.slices_done >= state.n_slices:
            del self.active_tasks[pair]

        return pos
