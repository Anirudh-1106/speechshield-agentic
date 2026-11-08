"""Tests for streaming speaker identification (no pyannote needed)."""

import numpy as np

from config.settings import DiarizationSettings
from core.diarization import SpeakerTracker
from core.event_bus import Event, EventBus
from core.state import GlobalState
from agents.diarization_agent import DiarizationAgent


class TestSpeakerTracker:
    def test_same_voice_maps_to_same_speaker(self) -> None:
        t = SpeakerTracker(threshold=0.7)
        rng = np.random.default_rng(0)
        voice = rng.standard_normal(192)
        a, _, new_a = t.assign(voice + rng.normal(0, 0.1, 192))
        b, sim, new_b = t.assign(voice + rng.normal(0, 0.1, 192))
        assert a == b == "speaker_1"
        assert new_a and not new_b
        assert sim > 0.9

    def test_different_voices_get_different_ids(self) -> None:
        t = SpeakerTracker(threshold=0.7)
        rng = np.random.default_rng(1)
        ids = {t.assign(rng.standard_normal(192))[0] for _ in range(3)}
        assert ids == {"speaker_1", "speaker_2", "speaker_3"}
        assert t.count == 3

    def test_max_speakers_caps_new_ids(self) -> None:
        t = SpeakerTracker(threshold=0.99, max_speakers=2)
        rng = np.random.default_rng(2)
        ids = [t.assign(rng.standard_normal(64))[0] for _ in range(5)]
        assert set(ids) <= {"speaker_1", "speaker_2"}

    def test_reset(self) -> None:
        t = SpeakerTracker()
        t.assign(np.ones(8))
        t.reset()
        assert t.count == 0


class FakeEngine:
    def __init__(self, speakers: list[str]) -> None:
        self._speakers = speakers
        self.calls = 0

    def load(self) -> None:
        pass

    def identify(self, pcm: bytes, sample_rate: int = 16000) -> tuple[str, float]:
        self.calls += 1
        return self._speakers.pop(0), 0.9

    def get_speaker_count(self) -> int:
        return 2

    def reset(self) -> None:
        pass


async def test_diarization_agent_tags_trace_and_emits() -> None:
    bus = EventBus()
    state = GlobalState()
    agent = DiarizationAgent(bus, state, engine=FakeEngine(["speaker_1"]))

    await agent._handle_segment(Event("speech_segment", {"audio": b"\x01\x00" * 32000, "sample_rate": 16000}, correlation_id="seg_a"))

    emitted = await bus._queue.get()
    assert emitted.type == "speaker_identified"
    assert emitted.data["speaker_id"] == "speaker_1"
    assert emitted.correlation_id == "seg_a"
    assert state.trace("seg_a").speaker == "speaker_1"
    assert state.current_speaker == "speaker_1"


async def test_short_segments_reuse_previous_speaker() -> None:
    bus = EventBus()
    state = GlobalState()
    engine = FakeEngine(["speaker_2"])
    agent = DiarizationAgent(bus, state, engine=engine)
    await agent._handle_segment(Event("speech_segment", {"audio": b"\x00\x00" * 32000}, correlation_id="long"))
    await agent._handle_segment(Event("speech_segment", {"audio": b"\x00\x00" * 1600}, correlation_id="short"))
    assert engine.calls == 1
    assert state.trace("short").speaker == "speaker_2"


async def test_agent_degrades_when_model_unavailable() -> None:
    class Broken(FakeEngine):
        def load(self) -> None:
            raise RuntimeError("needs HF_TOKEN")

    state = GlobalState()
    state.settings = state.settings.model_copy(update={"diarization": DiarizationSettings(enabled=True)})
    agent = DiarizationAgent(EventBus(), state, engine=Broken([]))
    await agent.start()  # returns instead of raising
    assert state.system_info["diarization"] == "unavailable"
