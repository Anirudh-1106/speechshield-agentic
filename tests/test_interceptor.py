"""
Tests for the interceptor's censoring engine (no audio hardware).

Drives CensorEngine with synthetic audio and an explicit clock to check
the core product promise: unjudged speech is never played, clean speech
plays untouched, and only the toxic words are censored.
"""

import numpy as np

from agents.audio_interceptor import CensorEngine
from config.settings import Settings

SR = 16000
CHUNK = 512           # 32 ms
DT = CHUNK / SR


def tone(amplitude: float = 8000) -> bytes:
    t = np.arange(CHUNK) / SR
    return (np.sin(2 * np.pi * 220 * t) * amplitude).astype(np.int16).tobytes()


def silence() -> bytes:
    return np.zeros(CHUNK, dtype=np.int16).tobytes()


def make_engine(**interceptor) -> CensorEngine:
    s = Settings()
    s = s.model_copy(update={
        "interceptor": s.interceptor.model_copy(update={"delay_seconds": 0.2, "max_hold_seconds": 1.0, **interceptor}),
        "segmenter": s.segmenter.model_copy(update={"min_silence_ms": 160, "min_speech_ms": 100, "pre_roll_ms": 64}),
    })
    return CensorEngine(s)


def _played(engine: CensorEngine) -> bytearray:
    if not hasattr(engine, "_test_played"):
        engine._test_played = bytearray()  # type: ignore[attr-defined]
    return engine._test_played  # type: ignore[attr-defined]


def capture(engine: CensorEngine, chunks: list[bytes], t0: float = 0.0, realtime: bool = True):
    """Push chunks at real-time spacing, releasing audio each tick like the
    agent's release loop does. Returns (closed segments, end time)."""
    closed = []
    played = _played(engine)
    t = t0
    for i, c in enumerate(chunks):
        seg = engine.push(c, i, t)
        if seg:
            closed.append(seg)
        if realtime:
            audio, _ = engine.release(t, output_buffered=-1e9)
            played.extend(b"".join(audio))
        t += DT
    return closed, t


def drain(engine: CensorEngine, now: float):
    """Release at *now*; returns (everything played so far, release events)."""
    audio, events = engine.release(now, output_buffered=-1e9)  # ignore output pacing
    played = _played(engine)
    played.extend(b"".join(audio))
    return bytes(played), events


UTTERANCE = [silence()] * 4 + [tone()] * 10 + [silence()] * 8


def test_speech_is_held_until_verdict() -> None:
    e = make_engine()
    closed, t = capture(e, UTTERANCE)
    assert len(closed) == 1
    out, events = drain(e, t + 0.5)
    # Only the leading silence (not adopted as pre-roll) may play; speech waits.
    assert len(out) == 2 * CHUNK * 2
    assert events == []
    assert e.pending_segments == 1


def test_clean_utterance_plays_unmodified() -> None:
    e = make_engine()
    closed, t = capture(e, UTTERANCE)
    e.on_toxicity(closed[0].segment_id, toxic=False, spans=[])
    out, events = drain(e, t + 0.5)
    assert out == b"".join(UTTERANCE[: len(out) // (CHUNK * 2)])
    assert events[0].action == "passed"
    assert events[0].hold_seconds >= 0.2


def test_toxic_word_is_bleeped_rest_is_original() -> None:
    e = make_engine(censor_mode="bleep", word_padding_ms=0)
    closed, t = capture(e, UTTERANCE)
    seg = closed[0]
    # Segment audio starts at the 2 pre-roll chunks; censor 0.10-0.20 s into it.
    e.on_toxicity(seg.segment_id, toxic=True, spans=[{"start": 0.10, "end": 0.20, "text": "idiot"}])
    out, events = drain(e, t + 0.5)
    assert events[0].action == "bleeped"
    pcm = np.frombuffer(out, dtype=np.int16)
    seg_start = 2 * CHUNK  # two leading silence chunks played before the segment
    original = np.frombuffer(b"".join(UTTERANCE), dtype=np.int16)
    lo, hi = seg_start + int(0.10 * SR), seg_start + int(0.20 * SR)
    assert not np.array_equal(pcm[lo:hi], original[lo:hi])         # bleeped
    assert np.array_equal(pcm[:lo - 10], original[:lo - 10])       # before: untouched
    assert np.array_equal(pcm[hi + 10:hi + 2000], original[hi + 10:hi + 2000])  # after: untouched


def test_mute_mode_silences_span() -> None:
    e = make_engine(censor_mode="mute", word_padding_ms=0)
    closed, t = capture(e, UTTERANCE)
    e.on_toxicity(closed[0].segment_id, toxic=True, spans=[{"start": 0.10, "end": 0.20, "text": "x"}])
    out, _ = drain(e, t + 0.5)
    pcm = np.frombuffer(out, dtype=np.int16)
    lo = 2 * CHUNK + int(0.10 * SR)
    assert not pcm[lo:lo + int(0.1 * SR)].any()


def test_toxic_without_word_timings_censors_whole_utterance() -> None:
    e = make_engine(censor_mode="mute")
    closed, t = capture(e, UTTERANCE)
    e.on_toxicity(closed[0].segment_id, toxic=True, spans=[])
    out, events = drain(e, t + 0.5)
    pcm = np.frombuffer(out, dtype=np.int16)
    assert events[0].action == "muted"
    assert not pcm[2 * CHUNK:].any()


def test_timeout_fails_closed_by_default() -> None:
    e = make_engine(fail_mode="closed")
    closed, t = capture(e, UTTERANCE)
    out, events = drain(e, t + 2.0)  # well past max_hold
    assert events[0].action == "timeout_muted"
    assert not np.frombuffer(out, dtype=np.int16)[2 * CHUNK:].any()


def test_timeout_fail_open_passes_audio() -> None:
    e = make_engine(fail_mode="open")
    closed, t = capture(e, UTTERANCE)
    _, events = drain(e, t + 2.0)
    assert events[0].action == "timeout_passed"


def test_known_toxic_never_fails_open() -> None:
    e = make_engine(fail_mode="open", censor_mode="tts")
    closed, t = capture(e, UTTERANCE)
    e.on_toxicity(closed[0].segment_id, toxic=True, spans=[])  # waiting for TTS that never comes
    _, events = drain(e, t + 2.0)
    assert events[0].action == "timeout_muted"


def test_tts_replacement_plays_once() -> None:
    e = make_engine(censor_mode="tts")
    closed, t = capture(e, UTTERANCE)
    cid = closed[0].segment_id
    e.on_toxicity(cid, toxic=True, spans=[])
    out, events = drain(e, t + 0.5)
    assert events == []  # still waiting for TTS
    tts = b"\x07\x00" * 4000
    e.on_tts(cid, tts)
    out, events = drain(e, t + 0.6)
    assert events[0].action == "replaced"
    assert out.count(tts) == 1


def test_failed_tts_mutes() -> None:
    e = make_engine(censor_mode="tts")
    closed, t = capture(e, UTTERANCE)
    e.on_toxicity(closed[0].segment_id, toxic=True, spans=[])
    e.on_tts(closed[0].segment_id, None)
    _, events = drain(e, t + 0.5)
    assert events[0].action == "muted"


def test_empty_transcript_releases_immediately() -> None:
    e = make_engine()
    closed, t = capture(e, UTTERANCE)
    e.on_transcript_empty(closed[0].segment_id, "no_speech")
    _, events = drain(e, t + 0.3)
    assert events[0].action == "passed"


def test_silence_is_dropped_to_catch_up() -> None:
    e = make_engine()
    capture(e, [silence()] * 30, realtime=False)
    out, _ = drain(e, 30 * DT + 2.0)  # far behind schedule
    assert len(out) < 30 * CHUNK * 2
    assert e.silence_dropped > 0


def test_state_is_released_after_playback() -> None:
    e = make_engine()
    closed, t = capture(e, UTTERANCE)
    e.on_toxicity(closed[0].segment_id, toxic=False, spans=[])
    drain(e, t + 0.5)
    assert e._segs == {}
