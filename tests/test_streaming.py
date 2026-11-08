"""
Streaming censoring: audio is released word-by-word while the speaker is
still talking, and toxic words are censored (bleep / cut / cloned-voice
splice) as they stream out.  Pure engine + agent logic, no hardware.
"""

import numpy as np
import pytest

from agents.asr_agent import ASRAgent
from agents.audio_interceptor import CensorEngine
from agents.toxicity_agent import ToxicityAgent
from config.settings import Settings
from core.event_bus import Event, EventBus
from core.state import GlobalState

SR = 16000
CHUNK = 512
DT = CHUNK / SR


def tone(amplitude: float = 8000) -> bytes:
    t = np.arange(CHUNK) / SR
    return (np.sin(2 * np.pi * 220 * t) * amplitude).astype(np.int16).tobytes()


def silence() -> bytes:
    return np.zeros(CHUNK, dtype=np.int16).tobytes()


def settings(**interceptor) -> Settings:
    s = Settings()
    return s.model_copy(update={
        "interceptor": s.interceptor.model_copy(update={
            "delay_seconds": 0.2, "max_hold_seconds": 1.0, "partial_interval_seconds": 0.5,
            "word_padding_ms": 0, "clone_timeout_seconds": 0.5, **interceptor}),
        "segmenter": s.segmenter.model_copy(update={"min_silence_ms": 160, "min_speech_ms": 100, "pre_roll_ms": 64}),
    })


class Harness:
    """Drives the engine like the agent's capture + 50 Hz release loops."""

    def __init__(self, streaming: bool = True, **interceptor) -> None:
        self.e = CensorEngine(settings(**interceptor), streaming=streaming)
        self.t = 0.0
        self.seq = 0
        self.played = bytearray()
        self.partials: list[tuple[str, bytes]] = []
        self.clones = []
        self.events = []

    def push(self, chunks):
        closed = []
        for c in chunks:
            seg = self.e.push(c, self.seq, self.t)
            self.seq += 1
            if seg:
                closed.append(seg)
            self.partials += self.e.take_partial_requests()
            self.clones += self.e.take_clone_requests()
            self.tick()
            self.t += DT
        return closed

    def tick(self, now=None):
        audio, events = self.e.release(self.t if now is None else now, output_buffered=-1e9)
        self.played.extend(b"".join(audio))
        self.events += events
        self.clones += self.e.take_clone_requests()


# 1.6 s of continuous speech: loud syllables with short dips, no pause.
# (A perfectly constant tone would be treated as background noise.)
SPEECH = ([tone()] * 4 + [tone(200)]) * 10


def test_partial_passes_requested_while_speaking() -> None:
    h = Harness()
    h.push(SPEECH)
    assert h.partials, "streaming should ask for passes before the utterance ends"
    cid, audio = h.partials[0]
    assert len(audio) >= 0.5 * SR * 2
    # Only one pass in flight at a time (self-pacing to ASR speed)
    assert len({c for c, _ in h.partials}) == 1 and len(h.partials) == 1


def test_no_partials_when_streaming_off_or_tts() -> None:
    for h in (Harness(streaming=False), Harness(censor_mode="tts")):
        h.push(SPEECH)
        assert h.partials == []


def test_confirmed_words_play_before_utterance_ends() -> None:
    h = Harness()
    h.push(SPEECH[:20])
    cid = h.partials[0][0]
    before = len(h.played)
    h.e.on_partial(cid, safe_until=0.5, spans=[])
    h.push(SPEECH[20:30])  # speaker still talking
    released = len(h.played) - before
    assert released >= int(0.4 * SR) * 2      # ~0.5 s of speech already out
    assert h.e.pending_segments == 1          # utterance not finished yet
    assert h.events == []                     # verdict not reported until final


def test_nothing_plays_past_the_confirmed_point() -> None:
    h = Harness()
    h.push(SPEECH[:20])
    cid = h.partials[0][0]
    h.e.on_partial(cid, safe_until=0.3, spans=[])
    h.push(SPEECH[20:])
    h.tick(h.t + 5)  # long after — still held (utterance never closed)
    assert len(h.played) <= int(0.32 * SR) * 2 + 2 * CHUNK * 2


def test_toxic_word_is_bleeped_mid_stream() -> None:
    h = Harness(censor_mode="bleep")
    h.push(SPEECH[:20])
    cid = h.partials[0][0]
    h.e.on_partial(cid, safe_until=0.6, spans=[{"start": 0.2, "end": 0.4, "text": "idiot"}])
    h.tick(h.t + 0.2)
    pcm = np.frombuffer(bytes(h.played), dtype=np.int16)
    original = np.frombuffer(b"".join(SPEECH), dtype=np.int16)
    lo, hi = int(0.25 * SR), int(0.35 * SR)
    assert not np.array_equal(pcm[lo:hi], original[lo:hi])
    assert np.array_equal(pcm[:int(0.15 * SR)], original[:int(0.15 * SR)])


def test_meaning_toxic_stops_streaming_until_final_verdict() -> None:
    h = Harness(censor_mode="mute")
    h.push(SPEECH[:20])
    cid = h.partials[0][0]
    h.e.on_partial(cid, safe_until=0.6, spans=[], meaning_toxic=True)
    h.push(SPEECH[20:])
    assert len(h.played) == 0
    closed = h.push([silence()] * 8)
    assert closed
    h.e.on_toxicity(cid, toxic=True, spans=[])
    h.tick(h.t + 0.5)
    assert h.events[0].action == "muted"
    assert not np.frombuffer(bytes(h.played), dtype=np.int16).any()


def test_final_verdict_after_streaming_reports_once() -> None:
    h = Harness()
    h.push(SPEECH[:20])
    cid = h.partials[0][0]
    h.e.on_partial(cid, safe_until=1.2, spans=[])
    h.push(SPEECH[20:] + [silence()] * 8)
    h.e.on_toxicity(cid, toxic=False, spans=[])
    h.tick(h.t + 0.5)
    h.tick(h.t + 1.0)
    assert [ev.action for ev in h.events] == ["passed"]
    assert h.e._segs == {}


# ---------------------------------------------------------------------------
# voice_clone: splice the speaker's cloned voice in place of the toxic word
# ---------------------------------------------------------------------------

def _clone_flow(pcm, spans=({"start": 0.2, "end": 0.4, "text": "idiot"},)):
    h = Harness(censor_mode="voice_clone")
    h.push(SPEECH[:34])  # ~1.09 s spoken so far
    cid = h.partials[0][0]
    h.e.on_partial(cid, safe_until=1.0, spans=list(spans))
    h.tick(h.t + 0.1)
    assert len(h.clones) == 1 and h.clones[0].text == "idiot"
    held = len(h.played)
    h.e.on_clone(h.clones[0].key, pcm)
    h.tick(h.t + 0.2)
    return h, held


def test_clone_pending_holds_the_word() -> None:
    h, held = _clone_flow(b"\x10\x00" * 1600)
    assert held <= int(0.2 * SR) * 2  # nothing past the toxic word before the clone arrived


def test_clone_is_spliced_in_and_timeline_adjusts() -> None:
    rep = (np.ones(4800, dtype=np.int16) * 1000).tobytes()  # 0.3 s replacement for a 0.2 s word
    h, _ = _clone_flow(rep)
    pcm = np.frombuffer(bytes(h.played), dtype=np.int16)
    expected = int(0.2 * SR) + 4800 + (len(h.played) // 2 - int(0.2 * SR) - 4800)
    assert len(pcm) == expected
    seg_at = int(0.2 * SR)
    middle = pcm[seg_at + 200: seg_at + 4600]
    assert np.all(middle == middle[0])  # the replacement samples (loudness-matched, constant)


def test_cut_removes_the_word() -> None:
    h, _ = _clone_flow(b"")
    # 1.0 s confirmed minus the 0.2 s cut word
    assert abs(len(h.played) // 2 - int(0.8 * SR)) <= CHUNK


def test_clone_failure_falls_back_to_bleep() -> None:
    h, _ = _clone_flow(None)
    pcm = np.frombuffer(bytes(h.played), dtype=np.int16)
    original = np.frombuffer(b"".join(SPEECH), dtype=np.int16)
    assert len(pcm) >= int(0.9 * SR)
    assert not np.array_equal(pcm[int(0.25 * SR):int(0.35 * SR)], original[int(0.25 * SR):int(0.35 * SR)])


def test_clone_timeout_bleeps_and_moves_on() -> None:
    h = Harness(censor_mode="voice_clone", clone_timeout_seconds=0.3)
    h.push(SPEECH[:34])
    cid = h.partials[0][0]
    h.e.on_partial(cid, safe_until=1.0, spans=[{"start": 0.2, "end": 0.4, "text": "idiot"}])
    h.tick(h.t + 1.0)  # clone never arrives
    assert len(h.played) // 2 >= int(0.9 * SR)
    h.e.on_clone(h.clones[0].key, b"\x01\x00" * 100)  # late reply is ignored
    h.tick(h.t + 1.5)


def test_meaning_toxic_whole_utterance_is_bleeped_not_cloned() -> None:
    h = Harness(censor_mode="voice_clone", streaming=False)
    closed = h.push(SPEECH[:20] + [silence()] * 8)
    h.e.on_toxicity(closed[0].segment_id, toxic=True, spans=[])
    h.tick(h.t + 0.5)
    assert h.clones == []
    assert h.events[0].action == "bleeped"


# ---------------------------------------------------------------------------
# ASR LocalAgreement + streaming verdicts
# ---------------------------------------------------------------------------

def w(text, start, end):
    return {"word": text, "start": start, "end": end, "probability": 0.9}


def test_local_agreement_commits_only_stable_prefix() -> None:
    agent = ASRAgent(EventBus(), GlobalState())
    first = [w("You", 0.0, 0.2), w("are", 0.2, 0.4), w("a", 0.4, 0.5)]
    assert agent._commit("s", first, edge=0.7) == []  # nothing to agree with yet
    second = [w("You", 0.0, 0.2), w("are", 0.2, 0.4), w("an", 0.4, 0.5), w("idiot", 0.5, 0.9)]
    assert [x["word"] for x in agent._commit("s", second, edge=1.0)] == ["You", "are"]
    third = second + [w("again", 0.9, 1.3)]
    # "an"/"idiot" now agree; "idiot" ends before the edge; "again" is too close to it
    assert [x["word"] for x in agent._commit("s", third, edge=1.1)] == ["You", "are", "an", "idiot"]


def test_commit_never_goes_backwards() -> None:
    agent = ASRAgent(EventBus(), GlobalState())
    words = [w("hi", 0.0, 0.2), w("there", 0.2, 0.5)]
    agent._commit("s", words, edge=1.0)
    assert len(agent._commit("s", words, edge=1.0)) == 2
    assert len(agent._commit("s", [w("hi", 0.0, 0.2), w("bear", 0.2, 0.5)], edge=1.0)) == 2


async def test_partial_dropped_after_final_transcription() -> None:
    agent = ASRAgent(EventBus(), GlobalState())
    agent._ready.set()
    agent._finalized.append("done")
    await agent._handle_partial(Event("speech_partial", {"audio": b"\x00\x00" * 100}, correlation_id="done"))
    assert agent.bus._queue.empty()


async def test_partial_verdict_flags_strong_words() -> None:
    bus = EventBus()
    agent = ToxicityAgent(bus, GlobalState())
    agent._keyword_only = True
    agent._ready.set()
    words = [w("you", 0.0, 0.2), w("fucking", 0.2, 0.6), w("idiot", 0.6, 1.0)]
    await agent._handle_partial(Event("partial_transcript", {"words": words, "safe_until": 1.0}, correlation_id="p"))
    ev = bus._queue.get_nowait()
    assert ev.type == "partial_verdict"
    assert ev.data["toxic_spans"][0]["start"] == 0.2
    assert ev.data["meaning_toxic"] is False
    assert ev.data["safe_until"] == 1.0


async def test_empty_partial_still_answers() -> None:
    bus = EventBus()
    agent = ToxicityAgent(bus, GlobalState())
    await agent._handle_partial(Event("partial_transcript", {"words": [], "safe_until": 0.0}, correlation_id="p"))
    assert bus._queue.get_nowait().data["safe_until"] == 0.0


@pytest.mark.parametrize("text,expected", [
    ("fucking idiot,", "goof"), ("Shut up", "hold on"), ("f***ing", ""), ("bullshit.", "nonsense"),
    ("go kill yourself", "take a break"), ("retarded", ""), ("stupid", "silly"),
    ("xyzzy-unknown", None),  # unrecognised: bleep, never synthesise unvetted words
    ("", None),
])
def test_soft_replacements(text, expected) -> None:
    from core.lexicon import soft_replacement
    assert soft_replacement(text) == expected


async def test_partial_never_releases_past_a_mild_word() -> None:
    """Regression: 'Shut up ...' leaked when the first fragment wasn't toxic yet."""
    bus = EventBus()
    agent = ToxicityAgent(bus, GlobalState())
    agent._keyword_only = True
    agent._ready.set()
    words = [w("Shut", 0.1, 0.3), w("up", 0.3, 0.5), w("and", 0.5, 0.6), w("stop", 0.6, 0.9)]
    await agent._handle_partial(Event("partial_transcript", {"words": words, "safe_until": 0.9}, correlation_id="m"))
    ev = bus._queue.get_nowait()
    assert ev.data["toxic_spans"] == []         # not toxic (yet): nothing censored...
    assert ev.data["safe_until"] == 0.1         # ...but nothing from "Shut" on is released


async def test_clean_partial_releases_everything() -> None:
    bus = EventBus()
    agent = ToxicityAgent(bus, GlobalState())
    agent._keyword_only = True
    agent._ready.set()
    words = [w("nice", 0.0, 0.3), w("shot", 0.3, 0.6)]
    await agent._handle_partial(Event("partial_transcript", {"words": words, "safe_until": 0.6}, correlation_id="c"))
    assert bus._queue.get_nowait().data["safe_until"] == 0.6
