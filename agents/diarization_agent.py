"""
DiarizationAgent — Real-time speaker identification.

Receives speech_segment events, embeds each utterance and matches it to a
known speaker (or registers a new one), then emits speaker_identified and
tags the utterance's trace so transcripts, toxicity profiles and rewrite
strictness are tracked per speaker.

If the embedding model cannot be loaded (missing pyannote, no HF token)
the agent marks itself degraded and the pipeline continues with a single
"default" speaker.

Metrics:
  - speechshield_speaker_identification_latency_seconds — Diarization latency
  - speechshield_speaker_changes_total — Speaker change count
  - speechshield_active_speakers — Active speaker count
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional

from core.diarization import DiarizationEngine
from core.event_bus import Event
from core.health import health_checks
from core.logging import get_logger
from core.metrics import metrics

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.state import GlobalState

logger = get_logger("agents.diarization_agent")


class DiarizationAgent:
    """Speaker identification agent — one speaker label per utterance."""

    def __init__(self, event_bus: "EventBus", state: "GlobalState", engine: Optional[DiarizationEngine] = None) -> None:
        self.bus = event_bus
        self.state = state
        self.settings = state.settings.diarization
        self._engine = engine or DiarizationEngine(self.settings)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="diarization")
        self._running = False
        self._stopped = asyncio.Event()
        self._last_speaker: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._stopped.clear()
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self._executor, self._engine.load)
        except Exception as e:
            logger.error("diarization_unavailable", error=str(e))
            health_checks().mark_degraded("DiarizationAgent", f"Disabled: {e}")
            self.state.system_info["diarization"] = "unavailable"
            return  # pipeline continues with the default speaker
        self._running = True
        self.bus.subscribe("speech_segment", self._handle_segment)
        health_checks().mark_ready("DiarizationAgent")
        self.state.system_info["diarization"] = self.settings.embedding_model
        logger.info("diarization_agent_started", subscribed_to="speech_segment")

        await self._stopped.wait()

    def stop(self) -> None:
        self._running = False
        self._stopped.set()
        health_checks().mark_starting("DiarizationAgent")
        self._engine.reset()

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    async def _handle_segment(self, event: Event) -> None:
        audio: bytes = event.data.get("audio", b"")
        if not audio:
            return
        cid = event.correlation_id
        sample_rate = event.data.get("sample_rate", 16000)
        duration = len(audio) / 2 / sample_rate

        if duration < self.settings.min_segment_seconds and self._last_speaker:
            # Too short for a reliable voiceprint: assume the same speaker continues.
            speaker, similarity = self._last_speaker, 0.0
        else:
            loop = asyncio.get_running_loop()
            start = time.time()
            try:
                speaker, similarity = await loop.run_in_executor(self._executor, self._engine.identify, audio, sample_rate)
            except Exception as e:
                logger.error("diarization_error", correlation_id=cid, error=str(e), error_type=type(e).__name__)
                health_checks().mark_degraded("DiarizationAgent", f"Diarization error: {e}")
                return
            metrics().speaker_identification_latency_seconds.observe(time.time() - start)

        if speaker != self._last_speaker:
            metrics().speaker_changes_total.inc()
            logger.info("speaker_change_detected", correlation_id=cid, speaker=speaker, previous_speaker=self._last_speaker)
            self._last_speaker = speaker

        self.state.trace(cid).speaker = speaker
        self.state.current_speaker = speaker
        stage_ms = self.state.mark_stage(cid, "diarization")
        metrics().stage_latency_seconds.labels(stage="diarization").observe(stage_ms / 1000.0)
        metrics().active_speakers.set(self._engine.get_speaker_count())

        await self.bus.emit(Event("speaker_identified", {
            "speaker_id": speaker,
            "confidence": round(float(similarity), 3),
            "duration": duration,
            "timestamp": time.time(),
        }, correlation_id=cid))
