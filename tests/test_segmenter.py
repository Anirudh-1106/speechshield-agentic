"""Tests for the streaming utterance segmenter."""

import numpy as np

from config.settings import SegmenterSettings
from core.segmenter import UtteranceSegmenter, chunk_rms

SR = 16000
CHUNK = 512  # 32 ms


def tone(amplitude: float, n: int = CHUNK) -> bytes:
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 220 * t) * amplitude).astype(np.int16).tobytes()


def silence(n: int = CHUNK) -> bytes:
    return np.zeros(n, dtype=np.int16).tobytes()


def feed(seg: UtteranceSegmenter, chunks: list[bytes], start_seq: int = 0):
    results = []
    for i, c in enumerate(chunks):
        results.append(seg.process(c, start_seq + i, i * CHUNK / SR))
    return results


def make(**kw) -> UtteranceSegmenter:
    cfg = SegmenterSettings(**{"min_silence_ms": 160, "min_speech_ms": 100, "pre_roll_ms": 64, **kw})
    return UtteranceSegmenter(cfg, sample_rate=SR, silence_threshold=300.0)


def test_rms() -> None:
    assert chunk_rms(silence()) == 0.0
    assert 6000 < chunk_rms(tone(10000)) < 8000


def test_utterance_closes_after_silence() -> None:
    seg = make()
    chunks = [silence()] * 5 + [tone(8000)] * 10 + [silence()] * 6
    results = feed(seg, chunks)
    opened = [r for r in results if r.opened]
    closed = [r.closed for r in results if r.closed]
    assert len(opened) == 1 and len(closed) == 1
    utt = closed[0]
    # pre-roll (2 chunks) + speech (10) + trailing silence until the close (5 x 32ms >= 160ms)
    assert len(utt.chunk_seqs) == 2 + 10 + 5
    assert opened[0].adopted_seqs == [3, 4]
    assert utt.voiced_ms >= 300
    assert not utt.forced


def test_every_chunk_in_segment_is_tagged() -> None:
    seg = make()
    results = feed(seg, [tone(8000)] * 4 + [silence()] * 6)
    ids = {r.segment_id for r in results[:9]}
    assert len(ids) == 1 and None not in ids


def test_short_blip_is_discarded() -> None:
    seg = make(min_speech_ms=200)
    results = feed(seg, [tone(8000)] * 2 + [silence()] * 6)
    assert any(r.discarded for r in results)
    assert not any(r.closed for r in results)


def test_max_length_forces_close() -> None:
    seg = make(max_segment_seconds=1.0)
    syllables = ([tone(8000)] * 4 + [tone(200)]) * 12  # ~1.9 s of speech with short gaps, no pause
    results = feed(seg, syllables)
    closed = [r.closed for r in results if r.closed]
    assert closed and closed[0].forced
    assert 0.95 <= closed[0].duration_seconds <= 1.1


def test_adaptive_noise_floor_ignores_steady_hum() -> None:
    seg = make()
    hum = tone(600)  # RMS ~420: above the fixed floor, but constant background
    for i in range(200):
        seg.process(hum, i, i * 0.032)
    seg2_results = [seg.process(hum, 1000 + i, 10 + i * 0.032) for i in range(10)]
    assert not any(r.is_voiced for r in seg2_results)
    assert seg.process(tone(9000), 2000, 20.0).is_voiced


def test_flush_closes_open_segment() -> None:
    seg = make()
    feed(seg, [tone(8000)] * 10)
    assert seg.in_speech
    res = seg.flush()
    assert res.closed is not None
    assert not seg.in_speech
