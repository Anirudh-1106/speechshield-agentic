"""
UtteranceSegmenter — streaming voice-activity segmentation.

Turns a continuous stream of fixed-size PCM chunks into utterances that
start at speech onset (plus a little pre-roll) and end at a natural
pause, or when they hit a maximum length.

Why not fixed windows?  A 2-second window cuts words in half, which hurts
both ASR accuracy and toxicity detection, and it forces a full window of
latency even for a one-word utterance.  Ending on silence means the
pipeline starts working the moment the speaker stops.

The detector is an energy VAD with an adaptive noise floor: cheap enough
to run inside the audio path, with no model to load.  Whisper's own VAD
filter cleans up whatever borderline audio slips through.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from config.settings import SegmenterSettings

SAMPLE_WIDTH = 2  # int16


def chunk_rms(chunk: bytes) -> float:
    """Root-mean-square level of an int16 PCM chunk."""
    if not chunk:
        return 0.0
    samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(np.square(samples))))


@dataclass
class SpeechSegment:
    """A finished utterance, ready for ASR."""

    segment_id: str
    audio: bytes
    chunk_seqs: list[int]
    start_time: float          # capture time of the first chunk
    end_time: float            # capture time of the last chunk
    voiced_ms: float
    forced: bool = False       # closed by max length rather than a pause
    sample_rate: int = 16000

    @property
    def duration_seconds(self) -> float:
        return len(self.audio) / (SAMPLE_WIDTH * self.sample_rate)


@dataclass
class SegmenterResult:
    """What happened when one chunk was fed in."""

    segment_id: Optional[str] = None             # segment this chunk belongs to
    opened: Optional[str] = None                 # a segment opened on this chunk
    adopted_seqs: list[int] = field(default_factory=list)  # pre-roll chunks pulled into it
    closed: Optional[SpeechSegment] = None       # a segment finished on this chunk
    discarded: Optional[str] = None              # a segment dropped as too short
    is_voiced: bool = False


class UtteranceSegmenter:
    """Streaming energy-VAD segmenter with adaptive noise floor and pre-roll."""

    def __init__(
        self,
        settings: "SegmenterSettings",
        *,
        sample_rate: int = 16000,
        silence_threshold: float = 300.0,
    ) -> None:
        self.settings = settings
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.noise_floor = silence_threshold / settings.noise_multiplier
        # Background level = 10th percentile of recent chunk levels.  Speech
        # has gaps between syllables, so this tracks the background even while
        # someone is talking — and adapts to a steady hum above the floor.
        self._levels: deque[float] = deque(maxlen=150)

        self._pre_roll: deque[tuple[int, bytes, float]] = deque()
        self._pre_roll_ms = 0.0

        self._seg_id: Optional[str] = None
        self._seg_audio = bytearray()
        self._seg_seqs: list[int] = []
        self._seg_start = 0.0
        self._seg_last = 0.0
        self._seg_ms = 0.0
        self._voiced_ms = 0.0
        self._silence_run_ms = 0.0

    # ------------------------------------------------------------------

    @property
    def in_speech(self) -> bool:
        return self._seg_id is not None

    @property
    def current_id(self) -> Optional[str]:
        return self._seg_id

    def current_audio(self) -> bytes:
        """Audio of the utterance still in progress (for streaming ASR passes)."""
        return bytes(self._seg_audio)

    @property
    def threshold(self) -> float:
        return max(self.silence_threshold, self.noise_floor * self.settings.noise_multiplier)

    def _chunk_ms(self, chunk: bytes) -> float:
        return len(chunk) / SAMPLE_WIDTH / self.sample_rate * 1000.0

    def process(self, chunk: bytes, seq: int, capture_time: float) -> SegmenterResult:
        """Feed one chunk; returns how it was classified and any segment boundary."""
        result = SegmenterResult()
        ms = self._chunk_ms(chunk)
        rms = chunk_rms(chunk)
        self._levels.append(rms)
        if len(self._levels) >= 30:  # ~1 s of history before trusting the estimate
            self.noise_floor = float(np.percentile(self._levels, 10))
        voiced = rms >= self.threshold
        result.is_voiced = voiced

        if not self.in_speech:
            if voiced:
                self._open(seq, chunk, capture_time, ms, result)
            else:
                self._push_pre_roll(seq, chunk, capture_time, ms)
            return result

        # Inside an utterance
        result.segment_id = self._seg_id
        self._append(seq, chunk, capture_time, ms)
        if voiced:
            self._voiced_ms += ms
            self._silence_run_ms = 0.0
        else:
            self._silence_run_ms += ms

        if self._silence_run_ms >= self.settings.min_silence_ms:
            self._close(result, forced=False)
        elif self._seg_ms >= self.settings.max_segment_seconds * 1000.0:
            self._close(result, forced=True)
        return result

    def flush(self) -> SegmenterResult:
        """Close any open utterance (call on shutdown or when input stops)."""
        result = SegmenterResult()
        if self.in_speech:
            result.segment_id = self._seg_id
            self._close(result, forced=False)
        return result

    # ------------------------------------------------------------------

    def _push_pre_roll(self, seq: int, chunk: bytes, t: float, ms: float) -> None:
        self._pre_roll.append((seq, chunk, t))
        self._pre_roll_ms += ms
        while self._pre_roll and self._pre_roll_ms - self._chunk_ms(self._pre_roll[0][1]) >= self.settings.pre_roll_ms:
            _, old, _ = self._pre_roll.popleft()
            self._pre_roll_ms -= self._chunk_ms(old)

    def _open(self, seq: int, chunk: bytes, t: float, ms: float, result: SegmenterResult) -> None:
        self._seg_id = f"seg_{uuid.uuid4().hex[:12]}"
        self._seg_audio = bytearray()
        self._seg_seqs = []
        self._seg_ms = 0.0
        self._voiced_ms = ms
        self._silence_run_ms = 0.0
        self._seg_start = self._pre_roll[0][2] if self._pre_roll else t
        for p_seq, p_chunk, p_t in self._pre_roll:
            self._append(p_seq, p_chunk, p_t, self._chunk_ms(p_chunk))
            result.adopted_seqs.append(p_seq)
        self._pre_roll.clear()
        self._pre_roll_ms = 0.0
        self._append(seq, chunk, t, ms)
        result.opened = self._seg_id
        result.segment_id = self._seg_id

    def _append(self, seq: int, chunk: bytes, t: float, ms: float) -> None:
        self._seg_audio.extend(chunk)
        self._seg_seqs.append(seq)
        self._seg_last = t
        self._seg_ms += ms

    def _close(self, result: SegmenterResult, *, forced: bool) -> None:
        seg_id = self._seg_id
        assert seg_id is not None
        if self._voiced_ms < self.settings.min_speech_ms:
            result.discarded = seg_id
        else:
            result.closed = SpeechSegment(
                segment_id=seg_id,
                audio=bytes(self._seg_audio),
                chunk_seqs=list(self._seg_seqs),
                start_time=self._seg_start,
                end_time=self._seg_last,
                voiced_ms=self._voiced_ms,
                forced=forced,
                sample_rate=self.sample_rate,
            )
        self._seg_id = None
        self._seg_audio = bytearray()
        self._seg_seqs = []
        self._seg_ms = 0.0
        self._voiced_ms = 0.0
        self._silence_run_ms = 0.0
