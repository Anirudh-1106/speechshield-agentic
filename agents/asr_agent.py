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
        
        # Audio buffer state
        self._audio_buffer = bytearray()
        self._current_speaker = "default"
        self._last_audio_time = time.time()
        self._buffer_lock = asyncio.Lock()
        
        self._running = False
        self._flush_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Model loading (once)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is None:
            model_size = getattr(self.state.settings.asr, "model_size", "small")
            logger.info("asr_model_loading", model=f"faster-whisper-{model_size}")
            try:
                self._model = WhisperModel(
                    model_size,
                    device="cpu",
                    compute_type="int8",
                    download_root="models",
                )
                logger.info("asr_model_loaded", model=f"faster-whisper-{model_size}")
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
        
        # Start idle flush loop
        self._flush_task = asyncio.create_task(self._idle_flush_loop())
        
        health_checks().mark_ready("ASRAgent")
        logger.info("asr_agent_started", subscribed_to="audio_chunk")

        # Keep alive until stopped
        while self._running:
            await asyncio.sleep(0.1)

    def stop(self) -> None:
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
        health_checks().mark_starting("ASRAgent")

    # ------------------------------------------------------------------
    # Idle Flush Loop
    # ------------------------------------------------------------------

    async def _idle_flush_loop(self) -> None:
        """Periodically checks if the speaker has paused, and transcribes early."""
        try:
            while self._running:
                await asyncio.sleep(0.1)
                
                idle_flush_sec = self.state.settings.asr.idle_flush_ms / 1000.0
                time_since_last_audio = time.time() - self._last_audio_time
                
                if time_since_last_audio > idle_flush_sec:
                    await self._attempt_transcribe(is_idle_flush=True)
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    async def _handle_audio_chunk(self, event: Event) -> None:
        """Process audio chunk and emit transcript when buffer full."""
        audio_bytes: bytes = event.data.get("audio", b"")
        if not audio_bytes:
            return

        async with self._buffer_lock:
            self._audio_buffer.extend(audio_bytes)
            self._current_speaker = event.data.get("speaker", "default")
            self._last_audio_time = time.time()

        # If buffer exceeds fixed threshold, transcribe immediately
        await self._attempt_transcribe(is_idle_flush=False, correlation_id=event.correlation_id)

    async def _attempt_transcribe(self, is_idle_flush: bool = False, correlation_id: str | None = None) -> None:
        """Extracts the buffer and kicks off transcription if it has enough data."""
        raw: bytes | None = None
        speaker: str = "default"
        
        async with self._buffer_lock:
            buffer_len = len(self._audio_buffer)
            # Only trigger if we meet the fixed threshold OR it's an idle flush with *some* data
            if (not is_idle_flush and buffer_len >= BUFFER_BYTES) or (is_idle_flush and buffer_len > 0):
                raw = bytes(self._audio_buffer)
                speaker = self._current_speaker
                self._audio_buffer.clear()
        
        if not raw:
            return

        # Ensure correlation_id is present
        if correlation_id is None:
            import uuid
            correlation_id = f"flush_{str(uuid.uuid4())[:8]}"

        # Run transcription in executor to avoid blocking the loop
        loop = asyncio.get_running_loop()
        
        try:
            start_time = time.time()
            m = metrics()
            
            # Fetch VAD settings
            vad_filter = self.state.settings.asr.vad_filter
            vad_min_silence_ms = self.state.settings.asr.vad_min_silence_ms
            
            # Time the transcription
            with m.asr_latency_seconds.time():
                text = await loop.run_in_executor(None, self._transcribe, raw, vad_filter, vad_min_silence_ms)
            
            elapsed = time.time() - start_time
            
            if text and text.strip():
                logger.info(
                    "asr_transcript_emitted",
                    correlation_id=correlation_id,
                    text_length=len(text.strip()),
                    duration_ms=int(elapsed * 1000),
                    is_idle_flush=is_idle_flush
                )
                m.utterances_processed_total.labels(status="success").inc()
                
                await self.bus.emit(
                    Event(
                        "text_transcript",
                        {
                            "text": text.strip(),
                            "speaker": speaker,
                            "timestamp": time.time(),
                        },
                        correlation_id=correlation_id,
                    )
                )
            else:
                logger.debug(
                    "asr_no_speech_detected",
                    correlation_id=correlation_id,
                    duration_ms=int(elapsed * 1000),
                    is_idle_flush=is_idle_flush
                )
                m.asr_errors_total.labels(reason="no_speech").inc()
                
        except asyncio.TimeoutError:
            logger.error(
                "asr_timeout",
                correlation_id=correlation_id,
                error="Transcription timed out",
            )
            metrics().asr_errors_total.labels(reason="timeout").inc()
            metrics().utterances_processed_total.labels(status="error").inc()
            health_checks().mark_error("ASRAgent", "Inference timeout")
            
        except Exception as e:
            logger.error(
                "asr_error",
                correlation_id=correlation_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            metrics().asr_errors_total.labels(reason="decode_error").inc()
            metrics().utterances_processed_total.labels(status="error").inc()
            health_checks().mark_error("ASRAgent", f"Inference error: {str(e)}")


    # ------------------------------------------------------------------
    # Transcription helper
    # ------------------------------------------------------------------

    def _transcribe(self, raw_pcm: bytes, vad_filter: bool, vad_min_silence_ms: int) -> str:
        """Write raw PCM to in-memory WAV and run Faster-Whisper."""
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(raw_pcm)

        wav_buffer.seek(0)

        # Parse vad_parameters if needed
        vad_kwargs = {}
        if vad_filter:
            vad_kwargs["min_silence_duration_ms"] = vad_min_silence_ms

        segments, _info = self._model.transcribe(
            wav_buffer,
            beam_size=1,
            best_of=1,
            language="en",
            vad_filter=vad_filter,
            vad_parameters=vad_kwargs if vad_kwargs else None,
        )

        texts: list[str] = []
        for seg in segments:
            texts.append(seg.text)
        return " ".join(texts)
