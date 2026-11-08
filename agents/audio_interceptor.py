"""
AudioInterceptor — Inline real-time audio censor.

Captures call audio from a virtual audio device (e.g. VB-Cable) and plays
it to the user's real headphones/speakers only once it has been judged:

  • clean speech        → original audio, untouched
  • ``bleep`` / ``mute`` → original voice; only the toxic words are bleeped/silenced
  • ``voice_clone``      → original voice; each toxic word is replaced by a polite
                           one spoken *in the speaker's own cloned voice*
                           (intensifiers like "f***ing" are simply cut out)
  • ``tts``              → the whole toxic utterance replaced by the polite rewrite

Streaming
---------
With ``streaming`` on, the interceptor does not wait for the speaker to
finish.  Every ``partial_interval_seconds`` of new audio it asks for a
partial transcription; words confirmed by two consecutive passes
(LocalAgreement) are checked immediately and audio is released up to the
last confirmed word — so listeners hear speech ~1 s behind real time
instead of a whole sentence behind.  Toxic words found mid-sentence are
censored as they stream out.

Safety: nothing is played before it is judged.  If no verdict arrives
within ``max_hold_seconds`` of an utterance ending, ``fail_mode`` decides
(``closed`` = mute, ``open`` = pass), and known-toxic audio is never passed.

Every censoring decision is an *edit* on the utterance timeline (bleep,
mute, cut, replace, pending-clone).  Partial verdicts, the final verdict
and late-arriving voice clones all add edits, which are applied as audio
streams out.

Requirements:
  - VB-Audio Virtual Cable (free): https://vb-audio.com/Cable/
  - Set your call app (Discord/Zoom/game) to output to "CABLE Input"
  - SpeechShield captures from "CABLE Output" and plays to your real device

Metrics:
  - speechshield_interceptor_segments_total{action} — utterances by outcome
  - speechshield_interceptor_hold_seconds — audio delay per utterance
  - speechshield_interceptor_edits_total{kind} — words censored, by method
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from core.event_bus import Event
from core.health import health_checks
from core.logging import get_logger
from core.metrics import metrics
from core.segmenter import SpeechSegment, UtteranceSegmenter

if TYPE_CHECKING:
    from config.settings import Settings
    from core.audio_io import AsyncCapture, PcmPlayer
    from core.event_bus import EventBus
    from core.state import GlobalState

logger = get_logger("agents.audio_interceptor")

SAMPLE_WIDTH = 2  # 16-bit
MAX_AHEAD_SECONDS = 0.25      # never queue more than this in the output device buffer
CATCHUP_SLACK_SECONDS = 0.15  # drop silence once this far behind the target delay
FADE_SAMPLES = 80             # 5 ms crossfade at cut/replace boundaries (no clicks)

_ACTION_FOR_MODE = {"bleep": "bleeped", "mute": "muted", "voice_clone": "voice_cloned"}


@dataclass
class _Held:
    """One captured chunk waiting in the delay line."""
    seq: int
    audio: bytes
    t: float
    seg: Optional[str] = None
    offset: float = 0.0  # seconds from the start of its utterance audio


@dataclass
class SpanEdit:
    """A censoring operation on an utterance's timeline (seconds from its start)."""
    start: float
    end: float
    text: str
    kind: str                    # bleep | mute | cut | replace | pending
    key: str = ""
    pcm: Optional[bytes] = None  # replacement audio (kind == replace)
    requested_at: float = 0.0
    inserted: bool = False


@dataclass
class CloneRequest:
    key: str
    correlation_id: str
    text: str
    duration: float


@dataclass
class _Seg:
    """Censoring state for one utterance."""
    id: str
    opened_at: float
    status: str = "open"         # open | pending | decided
    closed_at: float = 0.0
    next_offset: float = 0.0
    held_chunks: int = 0
    action: str = ""             # passed | bleeped | muted | voice_cloned | replaced | timeout_*
    edits: list[SpanEdit] = field(default_factory=list)
    tts: Optional[bytes] = None
    tts_written: bool = False
    known_toxic: bool = False
    awaiting_tts: bool = False
    # streaming
    safe_until: float = 0.0      # audio before this offset is cleared for release
    hold_rest: bool = False      # toxic by meaning: stop streaming, wait for the final verdict
    partial_inflight: bool = False
    last_partial_offset: float = 0.0
    first_hold: Optional[float] = None
    reported: bool = False


@dataclass
class Released:
    """Reported once per utterance, when its verdict is known and it has started playing."""
    correlation_id: str
    action: str
    hold_seconds: float


class CensorEngine:
    """Pure delay-line/censoring logic, independent of audio hardware.

    Feed captured chunks with :meth:`push`, verdicts with ``on_*``, pull
    playable audio with :meth:`release`.  Requests for streaming passes and
    voice clones accumulate in :meth:`take_partial_requests` /
    :meth:`take_clone_requests` for the agent to forward.
    """

    def __init__(self, settings: "Settings", *, streaming: bool = False) -> None:
        self.cfg = settings.interceptor
        self.sample_rate = settings.audio.sample_rate
        self.segmenter = UtteranceSegmenter(
            settings.segmenter,
            sample_rate=self.sample_rate,
            silence_threshold=settings.audio.silence_threshold,
        )
        self.streaming = streaming
        self._chunks: deque[_Held] = deque()
        self._segs: dict[str, _Seg] = {}
        self._edits_by_key: dict[str, tuple[str, SpanEdit]] = {}
        self._partial_requests: list[tuple[str, bytes]] = []
        self._clone_requests: list[CloneRequest] = []
        self._now = 0.0
        self.silence_dropped = 0

    @property
    def streaming_active(self) -> bool:
        return self.streaming and self.cfg.censor_mode != "tts"

    # ------------------------------------------------------------------
    # Capture side
    # ------------------------------------------------------------------

    def _dur(self, audio: bytes) -> float:
        return len(audio) / SAMPLE_WIDTH / self.sample_rate

    def push(self, chunk: bytes, seq: int, t: float) -> Optional[SpeechSegment]:
        """Buffer a captured chunk; returns an utterance when one closes."""
        self._now = max(self._now, t)
        held = _Held(seq=seq, audio=chunk, t=t)
        self._chunks.append(held)
        res = self.segmenter.process(chunk, seq, t)

        if res.opened:
            seg = _Seg(id=res.opened, opened_at=t)
            self._segs[seg.id] = seg
            adopted = set(res.adopted_seqs)
            by_seq = {h.seq: h for h in self._chunks if h.seq in adopted}
            for s in res.adopted_seqs:  # pre-roll, in capture order
                h = by_seq.get(s)
                dur = self._dur(h.audio) if h else self._dur(chunk)
                if h is not None:  # may already have been played if delay < pre-roll
                    h.seg, h.offset = seg.id, seg.next_offset
                    seg.held_chunks += 1
                seg.next_offset += dur

        if res.segment_id:
            seg = self._segs[res.segment_id]
            held.seg, held.offset = seg.id, seg.next_offset
            seg.next_offset += self._dur(chunk)
            seg.held_chunks += 1

        if res.discarded:
            self._decide(self._segs[res.discarded], "passed")
        if res.closed:
            seg = self._segs[res.closed.segment_id]
            seg.status, seg.closed_at = "pending", t
            return res.closed

        # Streaming: ask for another transcription pass of the utterance so far.
        if self.streaming_active and res.segment_id:
            seg = self._segs[res.segment_id]
            if (seg.status == "open" and not seg.partial_inflight and not seg.hold_rest
                    and seg.next_offset - seg.last_partial_offset >= self.cfg.partial_interval_seconds):
                seg.partial_inflight = True
                seg.last_partial_offset = seg.next_offset
                self._partial_requests.append((seg.id, self.segmenter.current_audio()))
        return None

    def flush(self, t: float) -> Optional[SpeechSegment]:
        res = self.segmenter.flush()
        if res.discarded:
            self._decide(self._segs[res.discarded], "passed")
        if res.closed:
            seg = self._segs[res.closed.segment_id]
            seg.status, seg.closed_at = "pending", t
            return res.closed
        return None

    def take_partial_requests(self) -> list[tuple[str, bytes]]:
        out, self._partial_requests = self._partial_requests, []
        return out

    def take_clone_requests(self) -> list[CloneRequest]:
        out, self._clone_requests = self._clone_requests, []
        return out

    # ------------------------------------------------------------------
    # Verdicts
    # ------------------------------------------------------------------

    def _decide(self, seg: _Seg, action: str, tts: Optional[bytes] = None) -> None:
        if seg.status == "decided":
            return
        if seg.status == "open":
            # Discarded segments are decided while still "open".
            seg.closed_at = seg.closed_at or seg.opened_at
        if action == "passed" and seg.edits:
            action = _ACTION_FOR_MODE.get(self.cfg.censor_mode, "bleeped")  # streamed words were censored
        seg.status, seg.action, seg.tts = "decided", action, tts

    def _whole_kind(self) -> str:
        return "mute" if self.cfg.censor_mode == "mute" else "bleep"

    def _add_edits(self, seg: _Seg, spans: list[dict]) -> None:
        """Turn toxic word spans into timeline edits (deduplicated)."""
        for span in spans:
            whole = bool(span.get("whole"))
            start = 0.0 if whole else float(span["start"])
            end = math.inf if whole else float(span["end"])
            if not whole and any(e.start < end and start < e.end for e in seg.edits):
                continue  # already being censored (seen in an earlier partial pass)
            if whole:
                kind = self._whole_kind()  # toxic by meaning: no single word to swap
            elif self.cfg.censor_mode == "voice_clone":
                kind = "pending"
            else:
                kind = self._whole_kind()
            edit = SpanEdit(start, end, str(span.get("text", "")), kind, requested_at=self._now)
            seg.edits.append(edit)
            seg.edits.sort(key=lambda e: e.start)
            metrics().interceptor_edits_total.labels(kind="whole" if whole else kind).inc()
            if kind == "pending":
                edit.key = f"{seg.id}:{len(self._edits_by_key)}"
                self._edits_by_key[edit.key] = (seg.id, edit)
                self._clone_requests.append(CloneRequest(edit.key, seg.id, edit.text, end - start))

    def on_partial(self, cid: str, safe_until: float, spans: list[dict], meaning_toxic: bool = False) -> None:
        """A streaming pass confirmed words up to *safe_until* (seconds into the utterance)."""
        seg = self._segs.get(cid)
        if not seg:
            return
        seg.partial_inflight = False
        if seg.status == "decided":
            return
        if spans:
            seg.known_toxic = True
            self._add_edits(seg, spans)
        if meaning_toxic and not spans:
            seg.known_toxic = True
            seg.hold_rest = True  # can't point at a word: let the final verdict handle the rest
            return
        # Stop short of the boundary so a following word's onset can't leak.
        seg.safe_until = max(seg.safe_until, safe_until - self.cfg.word_padding_ms / 1000.0)

    def on_transcript_empty(self, cid: str, reason: str = "no_speech") -> None:
        seg = self._segs.get(cid)
        if not seg:
            return
        if reason == "no_speech":
            self._decide(seg, "passed")
        else:  # ASR failed: we know nothing about this audio
            self._decide(seg, "timeout_passed" if self.cfg.fail_mode == "open" and not seg.known_toxic else "timeout_muted")
            if seg.action == "timeout_muted":
                seg.edits.append(SpanEdit(0.0, math.inf, "", "mute"))

    def on_toxicity(self, cid: str, toxic: bool, spans: list[dict]) -> None:
        seg = self._segs.get(cid)
        if not seg or seg.status == "decided":
            return
        if not toxic:
            self._decide(seg, "passed")
            return
        seg.known_toxic = True
        if self.cfg.censor_mode == "tts":
            seg.awaiting_tts = True
            return
        self._add_edits(seg, spans or [{"start": 0.0, "end": 0.0, "whole": True}])
        if any(math.isinf(e.end) for e in seg.edits):
            self._decide(seg, "muted" if self.cfg.censor_mode == "mute" else "bleeped")
        else:
            self._decide(seg, _ACTION_FOR_MODE.get(self.cfg.censor_mode, "bleeped"))

    def on_tts(self, cid: str, pcm: Optional[bytes]) -> None:
        seg = self._segs.get(cid)
        if not seg or seg.status == "decided" or not seg.awaiting_tts:
            return
        if pcm:
            self._decide(seg, "replaced", tts=pcm)
        else:
            seg.edits.append(SpanEdit(0.0, math.inf, "", "mute"))
            self._decide(seg, "muted")

    def on_clone(self, key: str, pcm: Optional[bytes]) -> None:
        """Replacement audio for a pending edit: bytes = splice, b"" = cut, None = bleep."""
        entry = self._edits_by_key.pop(key, None)
        if not entry:
            return
        seg_id, edit = entry
        if edit.kind != "pending":
            return  # timed out already
        if pcm is None:
            edit.kind = "bleep"
        elif not pcm:
            edit.kind = "cut"
        else:
            edit.kind, edit.pcm = "replace", self._match_loudness(seg_id, edit, pcm)
        metrics().interceptor_edits_total.labels(kind=edit.kind).inc()

    def _match_loudness(self, seg_id: str, edit: SpanEdit, pcm: bytes) -> bytes:
        """Scale the cloned word to the loudness of the word it replaces."""
        parts = []
        for h in self._chunks:
            if h.seg != seg_id:
                continue
            c0, n = h.offset, len(h.audio) // SAMPLE_WIDTH
            lo = max(0, int((edit.start - c0) * self.sample_rate))
            hi = min(n, int((edit.end - c0) * self.sample_rate))
            if hi > lo:
                parts.append(np.frombuffer(h.audio, dtype=np.int16)[lo:hi])
        rep = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        if not parts or not len(rep):
            return pcm
        target = float(np.sqrt(np.mean(np.square(np.concatenate(parts).astype(np.float32)))))
        current = float(np.sqrt(np.mean(np.square(rep))))
        if current < 1.0 or target < 1.0:
            return pcm
        gain = min(4.0, max(0.25, target / current))
        return np.clip(rep * gain, -32768, 32767).astype(np.int16).tobytes()

    def _decide_timeout(self, seg: _Seg) -> None:
        if seg.known_toxic or self.cfg.fail_mode == "closed":
            seg.edits.append(SpanEdit(0.0, math.inf, "", "mute"))
            self._decide(seg, "timeout_muted")
        else:
            self._decide(seg, "timeout_passed")

    # ------------------------------------------------------------------
    # Playback side
    # ------------------------------------------------------------------

    def _streamable(self, seg: _Seg, h: _Held) -> bool:
        return (self.streaming_active and not seg.hold_rest
                and h.offset + self._dur(h.audio) <= seg.safe_until)

    def _waiting_for_clone(self, seg: _Seg, h: _Held, now: float) -> bool:
        """Hold a chunk whose word is still being voice-cloned (bleep it on timeout)."""
        pad = self.cfg.word_padding_ms / 1000.0
        c0, c1 = h.offset, h.offset + self._dur(h.audio)
        for e in seg.edits:
            if e.kind == "pending" and e.start - pad < c1 and c0 < e.end + 2 * pad:
                if now - e.requested_at < self.cfg.clone_timeout_seconds:
                    return True
                e.kind = "bleep"
                self._edits_by_key.pop(e.key, None)
                metrics().interceptor_edits_total.labels(kind="clone_timeout").inc()
        return False

    def release(self, now: float, output_buffered: float = 0.0) -> tuple[list[bytes], list[Released]]:
        """Pop every chunk that may be played now, already censored."""
        self._now = max(self._now, now)
        out: list[bytes] = []
        ahead = output_buffered
        while self._chunks and ahead < MAX_AHEAD_SECONDS:
            h = self._chunks[0]
            age = now - h.t
            if age < self.cfg.delay_seconds:
                break

            if h.seg is None:
                self._chunks.popleft()
                if age > self.cfg.delay_seconds + CATCHUP_SLACK_SECONDS:
                    self.silence_dropped += 1  # behind schedule: skip silence to catch up
                    continue
                out.append(h.audio)
                ahead += self._dur(h.audio)
                metrics().interceptor_chunks_passed.inc()
                continue

            seg = self._segs.get(h.seg)
            if seg is None:
                self._chunks.popleft()
                out.append(h.audio)
                continue
            if seg.status != "decided" and not self._streamable(seg, h):
                if seg.status == "pending" and now - seg.closed_at >= self.cfg.max_hold_seconds:
                    self._decide_timeout(seg)
                else:
                    break  # hold until the verdict (or the next streaming pass) arrives
            if self._waiting_for_clone(seg, h, now):
                break

            self._chunks.popleft()
            if seg.first_hold is None:
                seg.first_hold = now - h.t
            audio = self._render(h, seg)
            if audio:
                out.append(audio)
                ahead += self._dur(audio)
            if seg.edits or seg.action == "replaced":
                metrics().interceptor_chunks_replaced.inc()
            else:
                metrics().interceptor_chunks_passed.inc()
            seg.held_chunks -= 1
        return out, self._collect_released()

    def _collect_released(self) -> list[Released]:
        events = []
        for seg in list(self._segs.values()):
            if seg.status != "decided":
                continue
            if not seg.reported and seg.first_hold is not None:
                seg.reported = True
                events.append(Released(seg.id, seg.action, seg.first_hold))
            if seg.held_chunks <= 0 and (seg.reported or seg.first_hold is None):
                self._segs.pop(seg.id, None)
                for e in seg.edits:
                    self._edits_by_key.pop(e.key, None)
        return events

    def _tone(self, abs_start: int, n: int) -> np.ndarray:
        idx = np.arange(abs_start, abs_start + n)
        tone = np.sin(2 * np.pi * self.cfg.bleep_frequency_hz * idx / self.sample_rate)
        return (tone * self.cfg.bleep_volume * 32767).astype(np.int16)

    def _render(self, h: _Held, seg: _Seg) -> bytes:
        if seg.action == "replaced":
            if seg.tts_written:
                return b""
            seg.tts_written = True
            return seg.tts or b""
        if not seg.edits:
            return h.audio

        samples = np.frombuffer(h.audio, dtype=np.int16)
        n = len(samples)
        c0 = int(round(h.offset * self.sample_rate))
        c1 = c0 + n
        pad = int(self.cfg.word_padding_ms / 1000.0 * self.sample_rate)
        parts: list[np.ndarray] = []
        pos = c0
        splice = False  # a cut/replace boundary needs a crossfade
        for e in seg.edits:
            s0 = int(e.start * self.sample_rate) - pad
            # Whisper's word-end timestamps run early: pad the tail twice as much so
            # no trailing consonant survives ("trash" -> "rough" + "sh").
            s1 = c1 if math.isinf(e.end) else int(e.end * self.sample_rate) + 2 * pad
            lo, hi = max(s0, pos), min(s1, c1)
            if hi <= lo:
                continue
            parts.append(samples[pos - c0:lo - c0].copy())
            if e.kind == "bleep" or e.kind == "pending":  # pending only after a timeout race
                parts.append(self._tone(lo, hi - lo))
            elif e.kind == "mute":
                parts.append(np.zeros(hi - lo, dtype=np.int16))
            elif e.kind == "replace" and not e.inserted:
                e.inserted = True
                parts.append(np.frombuffer(e.pcm or b"", dtype=np.int16).copy())
                splice = True
            else:  # cut, or the rest of an already-inserted replacement
                splice = True
            pos = hi
        parts.append(samples[pos - c0:].copy())
        if splice:
            _crossfade(parts)
        return np.concatenate(parts).astype(np.int16).tobytes()

    def backlog_seconds(self, now: float) -> float:
        return now - self._chunks[0].t if self._chunks else 0.0

    @property
    def pending_segments(self) -> int:
        return sum(1 for s in self._segs.values() if s.status != "decided")


def _crossfade(parts: list[np.ndarray]) -> None:
    """Short fades at every join so spliced audio has no clicks."""
    for i, p in enumerate(parts):
        if len(p) < 2 * FADE_SAMPLES:
            continue
        ramp = np.linspace(0.0, 1.0, FADE_SAMPLES, dtype=np.float32)
        if i > 0:
            p[:FADE_SAMPLES] = (p[:FADE_SAMPLES].astype(np.float32) * ramp).astype(np.int16)
        if i < len(parts) - 1:
            p[-FADE_SAMPLES:] = (p[-FADE_SAMPLES:].astype(np.float32) * ramp[::-1]).astype(np.int16)


class AudioInterceptor:
    """Inline audio censor agent — wires CensorEngine to real audio devices."""

    def __init__(self, event_bus: "EventBus", state: "GlobalState") -> None:
        self.bus = event_bus
        self.state = state
        self.cfg = state.settings.interceptor
        self.engine = CensorEngine(state.settings)
        self._capture: Optional["AsyncCapture"] = None
        self._player: Optional["PcmPlayer"] = None
        self._running = False
        self._seq = 0

    def _streaming_enabled(self) -> bool:
        mode = self.cfg.streaming
        if mode == "auto":
            # Several passes per second need GPU-speed ASR; on CPU they would
            # just delay the final transcription.
            return self.state.system_info.get("asr_device") == "cuda"
        return mode == "on"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Open input/output streams and begin intercepting."""
        from core.audio_io import AsyncCapture, PcmPlayer, _pyaudio, find_device

        audio_cfg = self.state.settings.audio
        pa = _pyaudio().PyAudio()
        try:
            input_idx = find_device(pa, self.cfg.input_device_name, is_input=True)
            output_idx = find_device(pa, self.cfg.output_device_name, is_input=False) if self.cfg.output_device_name else audio_cfg.output_device
        finally:
            pa.terminate()

        if input_idx is None:
            msg = (f"Input device '{self.cfg.input_device_name}' not found. Install VB-Cable and set your call app "
                   "to output to 'CABLE Input' (run `python main.py --list-devices`).")
            logger.error("interceptor_input_device_not_found", device_name=self.cfg.input_device_name)
            health_checks().mark_error("AudioInterceptor", msg)
            raise RuntimeError(msg)

        self._capture = AsyncCapture(
            sample_rate=audio_cfg.sample_rate, chunk_size=audio_cfg.chunk_size,
            device_index=input_idx, queue_size=audio_cfg.queue_size,
        )
        self._player = PcmPlayer(sample_rate=audio_cfg.sample_rate, chunk_size=audio_cfg.chunk_size, device_index=output_idx)
        self._capture.open()
        self._player.open()

        self.bus.subscribe("transcript_empty", self._on_transcript_empty)
        self.bus.subscribe("toxicity_result", self._on_toxicity_result)
        self.bus.subscribe("partial_verdict", self._on_partial_verdict)
        self.bus.subscribe("audio_output_ready", self._on_audio_output_ready)
        self.bus.subscribe("clone_ready", self._on_clone_ready)

        self._running = True
        health_checks().mark_ready("AudioInterceptor")
        logger.info(
            "interceptor_started",
            input_device=self.cfg.input_device_name,
            output_device=self.cfg.output_device_name or "system default",
            delay_seconds=self.cfg.delay_seconds,
            censor_mode=self.cfg.censor_mode,
            streaming=self.cfg.streaming,
            fail_mode=self.cfg.fail_mode,
        )
        try:
            await asyncio.gather(self._capture_loop(), self._release_loop())
        finally:
            self._cleanup()

    def stop(self) -> None:
        self._running = False
        health_checks().mark_starting("AudioInterceptor")

    # ------------------------------------------------------------------
    # Loops
    # ------------------------------------------------------------------

    async def _capture_loop(self) -> None:
        assert self._capture is not None
        while self._running:
            item = await self._capture.read(timeout=0.2)
            if item is None:
                continue
            chunk, t = item
            self._seq += 1
            self.engine.streaming = self._streaming_enabled()
            closed = self.engine.push(chunk, self._seq, t)
            for cid, audio in self.engine.take_partial_requests():
                await self.bus.emit(Event("speech_partial", {
                    "audio": audio, "sample_rate": self.engine.sample_rate,
                }, correlation_id=cid))
            if closed:
                await self._emit_segment(closed)
            await self._forward_clone_requests()

    async def _forward_clone_requests(self) -> None:
        for req in self.engine.take_clone_requests():
            await self.bus.emit(Event("clone_request", {
                "key": req.key,
                "text": req.text,
                "duration": req.duration,
                "speaker": self.state.trace(req.correlation_id).speaker,
            }, correlation_id=req.correlation_id))

    async def _release_loop(self) -> None:
        assert self._player is not None
        m = metrics()
        state_info = self.state.system_info
        while self._running:
            await asyncio.sleep(0.02)  # 50 Hz
            now = time.time()
            audio, released = self.engine.release(now, self._player.buffered_seconds)
            for a in audio:
                self._player.write(a)
            await self._forward_clone_requests()
            for r in released:
                m.interceptor_segments_total.labels(action=r.action).inc()
                m.interceptor_hold_seconds.observe(r.hold_seconds)
                self.state.mark_stage(r.correlation_id, "released")
                self.state.trace(r.correlation_id).action = r.action
                if r.action != "passed":
                    logger.info("interceptor_censored", correlation_id=r.correlation_id, action=r.action, hold_s=round(r.hold_seconds, 2))
                await self.bus.emit(Event("segment_released", {
                    "action": r.action, "hold_seconds": r.hold_seconds, "timestamp": now,
                }, correlation_id=r.correlation_id))
            m.interceptor_backlog_seconds.set(self.engine.backlog_seconds(now))
            state_info["streaming"] = self.engine.streaming_active

    async def _emit_segment(self, seg: SpeechSegment) -> None:
        trace = self.state.trace(seg.segment_id)
        trace.speech_end = time.time()
        trace.speech_duration = seg.duration_seconds
        await self.bus.emit(Event("speech_segment", {
            "audio": seg.audio,
            "sample_rate": self.engine.sample_rate,
            "start_time": seg.start_time,
            "end_time": seg.end_time,
            "duration": seg.duration_seconds,
            "forced": seg.forced,
        }, correlation_id=seg.segment_id))

    # ------------------------------------------------------------------
    # Pipeline event handlers
    # ------------------------------------------------------------------

    async def _on_transcript_empty(self, event: Event) -> None:
        self.engine.on_transcript_empty(event.correlation_id, event.data.get("reason", "no_speech"))

    async def _on_toxicity_result(self, event: Event) -> None:
        self.engine.on_toxicity(event.correlation_id, bool(event.data.get("toxic")), event.data.get("toxic_spans") or [])

    async def _on_partial_verdict(self, event: Event) -> None:
        d = event.data
        self.engine.on_partial(event.correlation_id, float(d.get("safe_until", 0.0)), d.get("toxic_spans") or [],
                               bool(d.get("meaning_toxic")))

    async def _on_audio_output_ready(self, event: Event) -> None:
        self.engine.on_tts(event.correlation_id, event.data.get("tts_audio_bytes"))

    async def _on_clone_ready(self, event: Event) -> None:
        self.engine.on_clone(event.data["key"], event.data.get("pcm"))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        for dev in (self._capture, self._player):
            if dev is not None:
                dev.close()
        self._capture = self._player = None
        logger.info("interceptor_cleaned_up")
