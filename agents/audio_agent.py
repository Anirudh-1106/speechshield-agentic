"""
AudioAgent — Captures live microphone audio and emits audio_chunk events.

Uses PyAudio in a non-blocking callback mode at 16 kHz mono.
"""

import asyncio
import logging
from typing import TYPE_CHECKING

import pyaudio

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.state import GlobalState

from core.event_bus import Event

logger = logging.getLogger("AudioAgent")

SAMPLE_RATE = 16_000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024


class AudioAgent:
    """Microphone capture agent — streams audio_chunk events onto the bus."""

    def __init__(self, event_bus: "EventBus", state: "GlobalState") -> None:
        self.bus = event_bus
        self.state = state
        self._pa: pyaudio.PyAudio | None = None
        self._stream: pyaudio.Stream | None = None
        self._buffer: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Open the mic stream and continuously emit audio_chunk events."""
        self._running = True
        self._pa = pyaudio.PyAudio()

        # Choose default input device
        dev_info = self._pa.get_default_input_device_info()
        logger.info(
            "Using input device: %s (index %s)",
            dev_info.get("name"),
            dev_info.get("index"),
        )

        self._stream = self._pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=self._audio_callback,
        )
        self._stream.start_stream()
        logger.info("AudioAgent started — streaming mic audio.")

        try:
            while self._running:
                try:
                    chunk = await asyncio.wait_for(self._buffer.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue

                await self.bus.emit(
                    Event("audio_chunk", {"audio": chunk, "sample_rate": SAMPLE_RATE})
                )
        except asyncio.CancelledError:
            logger.info("AudioAgent cancelled.")
        finally:
            self._cleanup()

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # PyAudio callback (runs in a separate thread)
    # ------------------------------------------------------------------

    def _audio_callback(
        self,
        in_data: bytes | None,
        frame_count: int,
        time_info: dict,
        status_flags: int,
    ) -> tuple[None, int]:
        if in_data is not None:
            try:
                self._buffer.put_nowait(in_data)
            except asyncio.QueueFull:
                # Drop oldest chunk to keep real-time
                try:
                    self._buffer.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    self._buffer.put_nowait(in_data)
                except asyncio.QueueFull:
                    pass
        return (None, pyaudio.paContinue)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None
        logger.info("AudioAgent cleaned up.")
