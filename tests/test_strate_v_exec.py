"""Tests for Strate V execution layer — pure Python, no framework deps."""

import time
import pytest

from src.strate_v_exec.synapse import SynapticVector
from src.strate_v_exec.cortex_node import CortexNode
from src.strate_v_exec.cerebellum_node import CerebellumNode, TWAPState
from src.strate_v_exec.reflex_arc import ReflexArc


class TestSynapticVector:
    def test_create(self):
        sv = SynapticVector(pair="BTCUSDT", target_pos=0.8)
        assert sv.pair == "BTCUSDT"
        assert sv.target_pos == 0.8
        assert len(sv.trace_id) == 12

    def test_not_expired(self):
        sv = SynapticVector(pair="ETHUSDT", target_pos=-0.5, ttl=60.0)
        assert not sv.is_expired()

    def test_expired(self):
        sv = SynapticVector(pair="ETHUSDT", target_pos=-0.5, ttl=0.0,
                           timestamp=time.time() - 1.0)
        assert sv.is_expired()

    def test_to_dict(self):
        sv = SynapticVector(pair="BTCUSDT", target_pos=1.0)
        d = sv.to_dict()
        assert d["pair"] == "BTCUSDT"
        assert d["target_pos"] == 1.0
        assert "trace_id" in d


class TestCortexNode:
    def test_dry_run(self):
        node = CortexNode(dry_run=True)
        sv = SynapticVector(pair="BTCUSDT", target_pos=0.5)
        node.publish(sv)  # should not raise
        node.close()


class TestCerebellumNode:
    def test_twap_slicing(self):
        cb = CerebellumNode(max_slices=10)
        sv = SynapticVector(pair="BTCUSDT", target_pos=1.0, urgency=0.0)
        state = cb.submit(sv, current_pos=0.0)
        assert state.n_slices == 10

        positions = []
        for _ in range(10):
            pos = cb.tick("BTCUSDT")
            assert pos is not None
            positions.append(pos)

        # After all slices, should be done
        assert cb.tick("BTCUSDT") is None
        assert positions[-1] == pytest.approx(1.0)

    def test_urgent_immediate(self):
        cb = CerebellumNode(max_slices=10)
        sv = SynapticVector(pair="ETHUSDT", target_pos=-1.0, urgency=1.0)
        state = cb.submit(sv, current_pos=0.0)
        assert state.n_slices == 1  # urgency=1 -> immediate

        pos = cb.tick("ETHUSDT")
        assert pos == pytest.approx(-1.0)
        assert cb.tick("ETHUSDT") is None

    def test_cancellation(self):
        cb = CerebellumNode(max_slices=10)
        sv1 = SynapticVector(pair="BTCUSDT", target_pos=1.0, urgency=0.0)
        cb.submit(sv1, current_pos=0.0)
        cb.tick("BTCUSDT")  # 1 slice done

        # New signal cancels old one
        sv2 = SynapticVector(pair="BTCUSDT", target_pos=-0.5, urgency=0.5)
        state2 = cb.submit(sv2, current_pos=0.1)
        assert state2.target_pos == -0.5


class TestReflexArc:
    def test_no_crash(self):
        ra = ReflexArc(threshold=0.05, window_seconds=60.0)
        t = 1000.0
        for i in range(10):
            assert not ra.update(100.0 + i * 0.1, now=t + i)

    def test_flash_crash(self):
        ra = ReflexArc(threshold=0.05, window_seconds=60.0)
        t = 1000.0
        ra.update(100.0, now=t)
        ra.update(99.0, now=t + 1)
        triggered = ra.update(94.0, now=t + 2)  # 6% drop
        assert triggered
        assert ra.trigger_count == 1

    def test_recovery(self):
        ra = ReflexArc(threshold=0.05, window_seconds=60.0)
        t = 1000.0
        ra.update(100.0, now=t)
        ra.update(94.0, now=t + 1)  # triggers
        assert ra.triggered
        # Price recovers
        assert not ra.update(99.0, now=t + 2)
        assert not ra.triggered
