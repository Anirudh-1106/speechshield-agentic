"""
AudioAgent — Captures live microphone audio and emits speech_segment events.

Uses PyAudio in non-blocking callback mode at 16 kHz mono.  Chunks are
fed through the ``UtteranceSegmenter`` so downstream agents receive whole
utterances that end at a natural pause — one correlation ID per utterance.
"""

import asyncio
import time
from typing import TYPE_CHECKING

from core.audio_io import AsyncCapture, find_device, _pyaudio
from core.event_bus import Event
from core.health import health_checks
from core.logging import get_logger
from core.metrics import metrics
from core.segmenter import SpeechSegment, UtteranceSegmenter

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.state import GlobalState

logger = get_logger("agents.audio_agent")


class AudioAgent:
    """Microphone capture agent — streams utterances onto the bus."""

    def __init__(self, event_bus: "EventBus", state: "GlobalState") -> None:
        self.bus = event_bus
        self.state = state
        cfg = state.settings
        self.sample_rate = cfg.audio.sample_rate
        self._segmenter = UtteranceSegmenter(
            cfg.segmenter,
            sample_rate=self.sample_rate,
            silence_threshold=cfg.audio.silence_threshold,
        )
        self._capture: AsyncCapture | None = None
        self._running = False
        self._seq = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Open the mic stream and continuously emit speech_segment events."""
        self._running = True
        cfg = self.state.settings.audio

        pyaudio = _pyaudio()
        pa = pyaudio.PyAudio()
        try:
            device = find_device(pa, None, is_input=True, index=cfg.input_device) if cfg.input_device is not None else None
            info = pa.get_device_info_by_index(device) if device is not None else pa.get_default_input_device_info()
        finally:
            pa.terminate()
        logger.info("audio_input_device", name=info.get("name"), index=info.get("index"))

        self._capture = AsyncCapture(
            sample_rate=self.sample_rate,
            chunk_size=cfg.chunk_size,
            device_index=device,
            queue_size=cfg.queue_size,
        )
        self._capture.open()
        health_checks().mark_ready("AudioAgent")
        logger.info("audio_agent_started", sample_rate=self.sample_rate, chunk_size=cfg.chunk_size)

        try:
            while self._running:
                item = await self._capture.read(timeout=0.2)
                if item is None:
                    continue
                chunk, t = item
                self._seq += 1
                result = self._segmenter.process(chunk, self._seq, t)
                if result.discarded:
                    metrics().audio_chunks_dropped_total.labels(reason="too_short").inc()
                if result.closed:
                    await self._emit_segment(result.closed)
                metrics().audio_queue_depth_samples.set(self._capture.backlog)
        except asyncio.CancelledError:
            logger.info("audio_agent_cancelled")
            raise
        finally:
            flushed = self._segmenter.flush()
            if flushed.closed:
                await self._emit_segment(flushed.closed)
            self._cleanup()

    def stop(self) -> None:
        self._running = False

    async def _emit_segment(self, seg: SpeechSegment) -> None:
        trace = self.state.trace(seg.segment_id)
        trace.speech_end = time.time()
        trace.speech_duration = seg.duration_seconds
        logger.debug("speech_segment", correlation_id=seg.segment_id, seconds=round(seg.duration_seconds, 2), forced=seg.forced)
        await self.bus.emit(Event("speech_segment", {
            "audio": seg.audio,
            "sample_rate": self.sample_rate,
            "start_time": seg.start_time,
            "end_time": seg.end_time,
            "duration": seg.duration_seconds,
            "forced": seg.forced,
        }, correlation_id=seg.segment_id))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        if self._capture is not None:
            self._capture.close()
            self._capture = None
        logger.info("audio_agent_cleaned_up")
