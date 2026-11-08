"""
ASR and Voice agent logic with fake models (no Whisper / Edge TTS needed).
"""

import asyncio
from types import SimpleNamespace

import pytest

from agents.asr_agent import ASRAgent, resolve_device, resolve_model
from agents.voice_agent import VoiceAgent
from config.settings import Settings
from core.event_bus import Event, EventBus
from core.state import GlobalState


def seg(text, words=(), no_speech=0.01, logprob=-0.2):
    return SimpleNamespace(
        text=text, no_speech_prob=no_speech, avg_logprob=logprob,
        words=[SimpleNamespace(word=w, start=s, end=e, probability=0.9) for w, s, e in words],
    )


class FakeWhisper:
    def __init__(self, segments, delay=0.0):
        self.segments = segments
        self.delay = delay
        self.kwargs = {}

    def transcribe(self, audio, **kwargs):
        import time
        time.sleep(self.delay)
        self.kwargs = kwargs
        return iter(self.segments), None


def make_asr(segments, delay=0.0, **asr):
    s = Settings()
    s = s.model_copy(update={"asr": s.asr.model_copy(update=asr)})
    state = GlobalState(settings=s)
    agent = ASRAgent(EventBus(), state)
    agent._model = FakeWhisper(segments, delay)
    agent._ready.set()
    return agent, state


SEGMENT = Event("speech_segment", {"audio": b"\x00\x01" * 16000, "duration": 1.0}, correlation_id="seg_x")


async def test_transcript_with_word_timings() -> None:
    agent, state = make_asr([seg(" You idiot.", [(" You", 0.0, 0.3), (" idiot.", 0.3, 0.8)])])
    await agent._handle_segment(SEGMENT)
    ev = agent.bus._queue.get_nowait()
    assert ev.type == "text_transcript" and ev.correlation_id == "seg_x"
    assert ev.data["text"] == "You idiot."
    assert ev.data["words"] == [
        {"word": "You", "start": 0.0, "end": 0.3, "probability": 0.9},
        {"word": "idiot.", "start": 0.3, "end": 0.8, "probability": 0.9},
    ]
    assert "asr" in state.trace("seg_x").stages
    assert agent._model.kwargs["word_timestamps"] is True
    assert agent._model.kwargs["condition_on_previous_text"] is False


@pytest.mark.parametrize("segments", [
    [],
    [seg("  ")],
    [seg(" Thank you.", no_speech=0.5)],          # classic silence hallucination
    [seg(" Thanks for watching!", logprob=-1.2)],
])
async def test_no_speech_emits_transcript_empty(segments) -> None:
    agent, _ = make_asr(segments)
    await agent._handle_segment(SEGMENT)
    ev = agent.bus._queue.get_nowait()
    assert ev.type == "transcript_empty"
    assert ev.data["reason"] == "no_speech"


async def test_confident_thank_you_is_kept() -> None:
    agent, _ = make_asr([seg(" Thank you.", no_speech=0.01, logprob=-0.1)])
    await agent._handle_segment(SEGMENT)
    assert agent.bus._queue.get_nowait().type == "text_transcript"


async def test_timeout_emits_transcript_empty_with_reason() -> None:
    agent, _ = make_asr([seg(" hi")], delay=0.5, timeout_seconds=0.1)
    await agent._handle_segment(SEGMENT)
    ev = agent.bus._queue.get_nowait()
    assert ev.type == "transcript_empty" and ev.data["reason"] == "timeout"


def test_resolve_device_explicit_values_pass_through() -> None:
    assert resolve_device("cpu", "auto") == ("cpu", "int8")
    assert resolve_device("cuda", "auto") == ("cuda", "int8_float16")
    assert resolve_model("auto", "cuda") == "small"
    assert resolve_model("auto", "cpu") == "base.en"
    assert resolve_model("tiny.en", "cuda") == "tiny.en"
    assert resolve_device("cpu", "float32") == ("cpu", "float32")


def test_cuda_failure_falls_back_to_cpu() -> None:
    s = Settings()
    s = s.model_copy(update={"asr": s.asr.model_copy(update={"device": "auto"})})
    agent = ASRAgent(EventBus(), GlobalState(settings=s))
    created = []

    def fake_create(cls, device, compute):
        created.append((device, compute))
        if device == "cuda":
            raise RuntimeError("Library cublas64_12.dll is not found")
        return FakeWhisper([])

    import agents.asr_agent as mod
    agent._create_and_warm = lambda cls, d, c: fake_create(cls, d, c)  # type: ignore[method-assign]
    orig = mod.resolve_device
    mod.resolve_device = lambda d, c: ("cuda", "float16")
    try:
        agent._load_model()
    finally:
        mod.resolve_device = orig
    assert created == [("cuda", "float16"), ("cpu", "int8")]
    assert "cpu/int8" in agent.state.system_info["asr"]


# ---------------------------------------------------------------------------
# VoiceAgent
# ---------------------------------------------------------------------------

def make_voice(interceptor=False, censor_mode="bleep", speak_clean=True):
    s = Settings()
    s = s.model_copy(update={
        "interceptor": s.interceptor.model_copy(update={"enabled": interceptor, "censor_mode": censor_mode}),
        "voice": s.voice.model_copy(update={"speak_clean": speak_clean}),
    })
    agent = VoiceAgent(EventBus(), GlobalState(settings=s), play_audio=False)
    return agent


@pytest.mark.parametrize("interceptor,mode,speak_clean,toxic,source,expected", [
    (False, "bleep", True, False, "audio", True),    # standard mode speaks everything
    (False, "bleep", False, False, "audio", False),  # ... unless speak_clean is off
    (False, "bleep", False, True, "audio", True),
    (True, "bleep", True, True, "audio", False),     # bleep keeps the real voice: no TTS
    (True, "tts", True, True, "audio", True),
    (True, "tts", True, False, "audio", False),
    (False, "bleep", True, True, "api", False),      # API requests never speak
])
def test_should_synthesize(interceptor, mode, speak_clean, toxic, source, expected) -> None:
    agent = make_voice(interceptor, mode, speak_clean)
    assert agent._should_synthesize(toxic, source) is expected


async def test_tts_failure_still_reports_so_interceptor_can_mute() -> None:
    agent = make_voice(interceptor=True, censor_mode="tts")

    async def failing(text, cid=""):
        return None

    agent._synthesize = failing  # type: ignore[method-assign]
    await agent._handle_rewrite(Event("rewrite_result", {"rewritten_text": "please stop", "was_toxic": True}, correlation_id="c1"))
    ev = agent.bus._queue.get_nowait()
    assert ev.type == "audio_output_ready" and ev.data["tts_audio_bytes"] is None


async def test_tts_pcm_forwarded() -> None:
    agent = make_voice(interceptor=True, censor_mode="tts")

    async def ok(text, cid=""):
        await asyncio.sleep(0)
        return b"\x01\x00" * 100

    agent._synthesize = ok  # type: ignore[method-assign]
    await agent._handle_rewrite(Event("rewrite_result", {"rewritten_text": "please stop", "was_toxic": True}, correlation_id="c2"))
    ev = agent.bus._queue.get_nowait()
    assert ev.data["tts_audio_bytes"] == b"\x01\x00" * 100
    assert "tts" in agent.state.trace("c2").stages
