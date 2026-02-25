"""
Tests for core.state — GlobalState, SpeakerProfile, DashboardRow.
"""

import time
import pytest
from core.state import GlobalState, SpeakerProfile, LatencyRecord, DashboardRow


class TestSpeakerProfile:
    """Unit tests for SpeakerProfile."""

    def test_initial_values(self) -> None:
        p = SpeakerProfile()
        assert p.total_utterances == 0
        assert p.toxic_count == 0
        assert p.toxicity_ratio == 0.0
        assert p.rewrite_strictness == 1.0

    def test_toxicity_ratio(self) -> None:
        p = SpeakerProfile(total_utterances=10, toxic_count=3)
        assert abs(p.toxicity_ratio - 0.3) < 1e-9


class TestLatencyRecord:
    """Unit tests for LatencyRecord."""

    def test_elapsed_zero_when_not_ended(self) -> None:
        r = LatencyRecord(start_time=time.time())
        assert r.elapsed_ms == 0.0

    def test_elapsed_positive(self) -> None:
        r = LatencyRecord(start_time=100.0, end_time=100.25)
        assert abs(r.elapsed_ms - 250.0) < 1e-6


class TestGlobalState:
    """Unit and integration tests for GlobalState."""

    def test_get_speaker_creates_profile(self) -> None:
        state = GlobalState()
        profile = state.get_speaker("alice")
        assert isinstance(profile, SpeakerProfile)
        assert "alice" in state.speakers

    def test_get_speaker_returns_same_profile(self) -> None:
        state = GlobalState()
        p1 = state.get_speaker("bob")
        p2 = state.get_speaker("bob")
        assert p1 is p2

    def test_record_toxic_utterance(self) -> None:
        state = GlobalState()
        state.record_utterance("user1", is_toxic=True)
        p = state.get_speaker("user1")
        assert p.total_utterances == 1
        assert p.toxic_count == 1
        assert p.consecutive_toxic == 1
        assert p.rewrite_strictness == 1.5

    def test_record_nontoxic_resets_consecutive(self) -> None:
        state = GlobalState()
        state.record_utterance("user1", is_toxic=True)
        state.record_utterance("user1", is_toxic=True)
        state.record_utterance("user1", is_toxic=False)
        p = state.get_speaker("user1")
        assert p.consecutive_toxic == 0

    def test_strictness_escalation(self) -> None:
        state = GlobalState()
        for _ in range(5):
            state.record_utterance("toxic_user", is_toxic=True)
        p = state.get_speaker("toxic_user")
        assert p.rewrite_strictness == 3.0  # Capped at 3.0

    def test_strictness_deescalation(self) -> None:
        state = GlobalState()
        for _ in range(4):
            state.record_utterance("user", is_toxic=True)
        state.record_utterance("user", is_toxic=False)
        p = state.get_speaker("user")
        assert p.rewrite_strictness < 3.0

    def test_latency_tracking(self) -> None:
        state = GlobalState()
        rec = state.start_pass()
        assert state.pass_counter == 1
        elapsed = state.end_pass(rec)
        assert elapsed >= 0.0
        assert state.avg_latency_ms >= 0.0

    def test_dashboard_row(self) -> None:
        state = GlobalState()
        row = DashboardRow(
            timestamp=time.time(),
            transcript="test",
            toxic=True,
            confidence=0.95,
            rewritten="polite test",
            speaker="alice",
        )
        state.add_dashboard_row(row)
        assert len(state.dashboard_log) == 1
        assert state.dashboard_log[0].transcript == "test"

    def test_dashboard_snapshot(self) -> None:
        state = GlobalState()
        state.record_utterance("default", is_toxic=True)
        state.add_dashboard_row(DashboardRow(
            timestamp=time.time(),
            transcript="bad word",
            toxic=True,
            confidence=0.9,
            rewritten="good word",
            speaker="default",
        ))
        snap = state.get_dashboard_snapshot()
        assert "pass_counter" in snap
        assert "speakers" in snap
        assert "log" in snap
        assert len(snap["log"]) == 1

    def test_dashboard_log_capped(self) -> None:
        state = GlobalState()
        state._max_dashboard_rows = 5
        for i in range(10):
            state.add_dashboard_row(DashboardRow(
                timestamp=time.time(),
                transcript=f"msg{i}",
                toxic=False,
                confidence=0.0,
                rewritten=f"msg{i}",
                speaker="default",
            ))
        assert len(state.dashboard_log) == 5
