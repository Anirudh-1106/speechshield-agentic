"""
ASRAgent — Automatic Speech Recognition agent.

Buffers audio_chunk events, writes temporary WAV segments, and
transcribes with Faster-Whisper (small model, CPU).
Emits text_transcript events.
"""

import asyncio
import io
import logging
import struct
import time
import wave
from typing import TYPE_CHECKING

from faster_whisper import WhisperModel

from core.event_bus import Event

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.state import GlobalState

logger = logging.getLogger("ASRAgent")

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
            logger.info("Loading Faster-Whisper 'small' model (CPU) …")
            self._model = WhisperModel(
                "small",
                device="cpu",
                compute_type="int8",
                download_root="models",
            )
            logger.info("Faster-Whisper model loaded.")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to audio_chunk events."""
        self._load_model()
        self._running = True
        self.bus.subscribe("audio_chunk", self._handle_audio_chunk)
        logger.info("ASRAgent started — listening for audio_chunk events.")

        # Keep alive until stopped
        while self._running:
            await asyncio.sleep(0.1)

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    async def _handle_audio_chunk(self, event: Event) -> None:
        audio_bytes: bytes = event.data.get("audio", b"")
        if not audio_bytes:
            return

        self._audio_buffer.extend(audio_bytes)

        if len(self._audio_buffer) >= BUFFER_BYTES:
            raw = bytes(self._audio_buffer)
            self._audio_buffer.clear()

            # Run transcription in executor to avoid blocking the loop
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(None, self._transcribe, raw)

            if text and text.strip():
                logger.info("Transcript: %s", text.strip())
                await self.bus.emit(
                    Event(
                        "text_transcript",
                        {
                            "text": text.strip(),
                            "speaker": "default",
                            "timestamp": time.time(),
                        },
                    )
                )

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
