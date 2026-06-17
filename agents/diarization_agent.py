"""
DiarizationAgent — Real-time speaker identification.

Receives audio_chunk events, runs speaker diarization, and emits
speaker_segment events with identified speakers and timestamps.

Metrics:
  - speechshield_speaker_identification_latency_seconds — Diarization latency
  - speechshield_speaker_changes_total — Speaker change count
  - speechshield_speakers_active_total — Active speaker count
"""

import asyncio
import time
from typing import TYPE_CHECKING

from core.event_bus import Event
from core.logging import get_logger
from core.metrics import metrics
from core.health import health_checks
from core.diarization import DiarizationEngine

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.state import GlobalState

logger = get_logger("agents.diarization_agent")

# Collect audio chunks for diarization window (1 second at 16kHz = 32000 bytes)
DIARIZATION_WINDOW_SECONDS = 2.0
SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2  # 16-bit
DIARIZATION_BUFFER_BYTES = int(SAMPLE_RATE * SAMPLE_WIDTH * DIARIZATION_WINDOW_SECONDS)


class DiarizationAgent:
    """Speaker identification agent — identifies speaker changes in real-time."""

    def __init__(self, event_bus: "EventBus", state: "GlobalState") -> None:
        self.bus = event_bus
        self.state = state
        self._engine: DiarizationEngine | None = None
        self._audio_buffer = bytearray()
        self._running = False
        self._last_speaker: str | None = None
        self._speaker_change_count = 0

    # ------------------------------------------------------------------
    # Model loading (once)
    # ------------------------------------------------------------------

    def _load_engine(self) -> None:
        if self._engine is None:
            logger.info("diarization_engine_loading", model="pyannote/speaker-diarization-3.1")
            try:
                self._engine = DiarizationEngine(
                    model="pyannote/speaker-diarization-3.1",
                    device="cpu",
                )
                logger.info("diarization_engine_loaded")
            except Exception as e:
                error_msg = f"Engine load failed: {str(e)}"
                logger.error("diarization_engine_load_failed", error=str(e))
                health_checks().mark_error("DiarizationAgent", error_msg)
                raise

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to audio_chunk events."""
        self._load_engine()
        self._running = True
        self.bus.subscribe("audio_chunk", self._handle_audio_chunk)
        health_checks().mark_ready("DiarizationAgent")
        logger.info(
            "diarization_agent_started",
            subscribed_to="audio_chunk",
            window_seconds=DIARIZATION_WINDOW_SECONDS,
        )

        # Keep alive until stopped
        while self._running:
            await asyncio.sleep(0.1)

    def stop(self) -> None:
        self._running = False
        health_checks().mark_starting("DiarizationAgent")
        if self._engine:
            self._engine.reset()

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    async def _handle_audio_chunk(self, event: Event) -> None:
        audio_bytes: bytes = event.data.get("audio", b"")
        if not audio_bytes:
            return

        self._audio_buffer.extend(audio_bytes)

        # Once we have enough audio, run diarization
        if len(self._audio_buffer) >= DIARIZATION_BUFFER_BYTES:
            raw = bytes(self._audio_buffer)
            self._audio_buffer.clear()

            # Run diarization in executor
            loop = asyncio.get_running_loop()
            segments = await loop.run_in_executor(
                None,
                self._diarize_buffer,
                raw,
                event.correlation_id,
            )

            # Emit speaker_segment events for each identified segment
            for segment in segments:
                speaker = segment["speaker"]
                
                # Track speaker changes
                if speaker != self._last_speaker:
                    self._speaker_change_count += 1
                    metrics().speaker_changes_total.inc()
                    logger.info(
                        "speaker_change_detected",
                        correlation_id=event.correlation_id,
                        speaker=speaker,
                        previous_speaker=self._last_speaker,
                    )
                    self._last_speaker = speaker

                # Update global state with the latest speaker
                self.state.current_speaker = speaker

                # Emit speaker_identified event
                await self.bus.emit(
                    Event(
                        "speaker_identified",
                        {
                            "speaker_id": speaker,
                            "start": segment["start"],
                            "end": segment["end"],
                            "confidence": segment["confidence"],
                            "duration": segment["duration"],
                            "timestamp": time.time(),
                        },
                        correlation_id=event.correlation_id,
                    )
                )

            # Update active speakers gauge
            if self._engine:
                speaker_count = self._engine.get_speaker_count()
                metrics().active_speakers.set(speaker_count)

    # ------------------------------------------------------------------
    # Diarization helper
    # ------------------------------------------------------------------

    def _diarize_buffer(self, raw_pcm: bytes, correlation_id: str) -> list[dict]:
        """Synchronous diarization wrapper for executor."""
        if not self._engine:
            return []

        start_time = time.time()
        m = metrics()

        try:
            segments = asyncio.run(
                self._engine.diarize(raw_pcm, sample_rate=SAMPLE_RATE, min_duration=0.3)
            )

            elapsed = time.time() - start_time
            m.speaker_identification_latency_seconds.observe(elapsed)

            logger.debug(
                "diarization_segments",
                correlation_id=correlation_id,
                segments_count=len(segments),
                duration_ms=int(elapsed * 1000),
            )

            return segments

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                "diarization_error",
                correlation_id=correlation_id,
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=int(elapsed * 1000),
            )
            health_checks().mark_error("DiarizationAgent", f"Diarization error: {str(e)}")
            return []
