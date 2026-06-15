"""
ASRAgent — Automatic Speech Recognition agent.

Buffers audio_chunk events, writes temporary WAV segments, and
transcribes with Faster-Whisper (small model, CPU).
Emits text_transcript events.

Metrics:
  - speechshield_asr_latency_seconds — Inference latency
  - speechshield_asr_errors_total — Error count by reason
  - speechshield_utterances_processed_total — Processed utterance count
"""

import asyncio
import io
import time
import wave
from typing import TYPE_CHECKING

from faster_whisper import WhisperModel

from core.event_bus import Event
from core.logging import get_logger
from core.metrics import metrics
from core.health import health_checks

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.state import GlobalState

logger = get_logger("agents.asr_agent")

# Buffer ~2 seconds of audio at 16 kHz before transcribing
SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit
BUFFER_SECONDS = 2.0
BUFFER_BYTES = int(SAMPLE_RATE * SAMPLE_WIDTH * BUFFER_SECONDS)


class ASRAgent:
    """Transcription agent — converts audio chunks into text."""

    def __init__(self, event_bus: "EventBus", state: "GlobalState") -> None:
        self.bus = event_bus
        self.state = state
        self._model: WhisperModel | None = None
        self._audio_buffer = bytearray()
        self._running = False

    # ------------------------------------------------------------------
    # Model loading (once)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is None:
            logger.info("asr_model_loading", model="faster-whisper-small")
            try:
                self._model = WhisperModel(
                    "small",
                    device="cpu",
                    compute_type="int8",
                    download_root="models",
                )
                logger.info("asr_model_loaded", model="faster-whisper-small")
            except Exception as e:
                error_msg = f"Model load failed: {str(e)}"
                logger.error("asr_model_load_failed", error=str(e))
                health_checks().mark_error("ASRAgent", error_msg)
                raise

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to audio_chunk events."""
        self._load_model()
        self._running = True
        self.bus.subscribe("audio_chunk", self._handle_audio_chunk)
        health_checks().mark_ready("ASRAgent")
        logger.info("asr_agent_started", subscribed_to="audio_chunk")

        # Keep alive until stopped
        while self._running:
            await asyncio.sleep(0.1)

    def stop(self) -> None:
        self._running = False
        health_checks().mark_starting("ASRAgent")

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    async def _handle_audio_chunk(self, event: Event) -> None:
        """Process audio chunk and emit transcript when buffer full."""
        audio_bytes: bytes = event.data.get("audio", b"")
        if not audio_bytes:
            return

        self._audio_buffer.extend(audio_bytes)

        if len(self._audio_buffer) >= BUFFER_BYTES:
            raw = bytes(self._audio_buffer)
            self._audio_buffer.clear()

            # Run transcription in executor to avoid blocking the loop
            loop = asyncio.get_running_loop()
            
            try:
                start_time = time.time()
                m = metrics()
                
                # Time the transcription
                with m.asr_latency_seconds.time():
                    text = await loop.run_in_executor(None, self._transcribe, raw)
                
                elapsed = time.time() - start_time
                
                if text and text.strip():
                    logger.info(
                        "asr_transcript_emitted",
                        correlation_id=event.correlation_id,
                        text_length=len(text.strip()),
                        duration_ms=int(elapsed * 1000),
                    )
                    m.utterances_processed_total.labels(status="success").inc()
                    
                    await self.bus.emit(
                        Event(
                            "text_transcript",
                            {
                                "text": text.strip(),
                                "speaker": "default",
                                "timestamp": time.time(),
                            },
                            correlation_id=event.correlation_id,
                        )
                    )
                else:
                    logger.info(
                        "asr_no_speech_detected",
                        correlation_id=event.correlation_id,
                        duration_ms=int(elapsed * 1000),
                    )
                    m.asr_errors_total.labels(reason="no_speech").inc()
                    
            except asyncio.TimeoutError:
                logger.error(
                    "asr_timeout",
                    correlation_id=event.correlation_id,
                    error="Transcription timed out",
                )
                metrics().asr_errors_total.labels(reason="timeout").inc()
                metrics().utterances_processed_total.labels(status="error").inc()
                health_checks().mark_error("ASRAgent", "Inference timeout")
                
            except Exception as e:
                logger.error(
                    "asr_error",
                    correlation_id=event.correlation_id,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                metrics().asr_errors_total.labels(reason="decode_error").inc()
                metrics().utterances_processed_total.labels(status="error").inc()
                health_checks().mark_error("ASRAgent", f"Inference error: {str(e)}")

    # ------------------------------------------------------------------
    # Transcription helper
    # ------------------------------------------------------------------

    def _transcribe(self, raw_pcm: bytes) -> str:
        """Write raw PCM to in-memory WAV and run Faster-Whisper."""
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(raw_pcm)

        wav_buffer.seek(0)

        segments, _info = self._model.transcribe(
            wav_buffer,
            beam_size=1,
            best_of=1,
            language="en",
            vad_filter=True,
        )

        texts: list[str] = []
        for seg in segments:
            texts.append(seg.text)
        return " ".join(texts)
