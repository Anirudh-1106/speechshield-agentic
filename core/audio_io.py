"""
Audio I/O helpers shared by the capture and interceptor agents.

  • Device discovery by name substring or index
  • ``AsyncCapture`` — PyAudio input stream whose callback thread hands
    chunks to asyncio *thread-safely* (``call_soon_threadsafe``)
  • ``PcmPlayer`` — callback-driven output stream fed from a byte buffer,
    so writing audio never blocks the event loop

PyAudio is imported lazily so headless deployments (Docker, API-only)
can import the rest of the system without PortAudio installed.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Any, Optional

from core.logging import get_logger

logger = get_logger("core.audio_io")

SAMPLE_WIDTH = 2  # int16


def _pyaudio() -> Any:
    import pyaudio  # noqa: PLC0415 — optional dependency
    return pyaudio


def list_devices() -> list[dict]:
    """Return every audio device PortAudio can see."""
    pyaudio = _pyaudio()
    pa = pyaudio.PyAudio()
    try:
        devices = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            devices.append({
                "index": i,
                "name": info.get("name", ""),
                "inputs": int(info.get("maxInputChannels", 0)),
                "outputs": int(info.get("maxOutputChannels", 0)),
                "default_rate": int(info.get("defaultSampleRate", 0)),
                "host_api": pa.get_host_api_info_by_index(info.get("hostApi", 0)).get("name", ""),
            })
        return devices
    finally:
        pa.terminate()


def find_device(pa: Any, name: Optional[str], *, is_input: bool, index: Optional[int] = None) -> Optional[int]:
    """Resolve a device by explicit index, else by case-insensitive name substring.

    Prefers WASAPI/MME matches in host-API order as PortAudio lists them,
    returning the first device with channels in the requested direction.
    """
    key = "maxInputChannels" if is_input else "maxOutputChannels"
    if index is not None:
        info = pa.get_device_info_by_index(index)
        if info.get(key, 0) > 0:
            return index
        logger.warning("device_index_wrong_direction", index=index, is_input=is_input)
        return None
    if not name:
        return None
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if name.lower() in str(info.get("name", "")).lower() and info.get(key, 0) > 0:
            logger.info("device_found", name=info.get("name"), index=i, is_input=is_input)
            return i
    logger.warning("device_not_found", name=name, is_input=is_input)
    return None


class AsyncCapture:
    """Microphone/virtual-cable capture delivering chunks into asyncio."""

    def __init__(
        self,
        *,
        sample_rate: int,
        chunk_size: int,
        device_index: Optional[int] = None,
        queue_size: int = 200,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device_index = device_index
        self._queue: asyncio.Queue[tuple[bytes, float]] = asyncio.Queue(maxsize=queue_size)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._pa: Any = None
        self._stream: Any = None
        self.dropped = 0

    def open(self) -> None:
        pyaudio = _pyaudio()
        self._loop = asyncio.get_running_loop()
        self._pa = pyaudio.PyAudio()
        kwargs: dict[str, Any] = dict(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._callback,
        )
        if self.device_index is not None:
            kwargs["input_device_index"] = self.device_index
        self._stream = self._pa.open(**kwargs)
        self._stream.start_stream()

    def _callback(self, in_data: Optional[bytes], frame_count: int, time_info: dict, status: int) -> tuple[None, int]:
        pyaudio = _pyaudio()
        if in_data is not None and self._loop is not None:
            # Capture timestamp taken on the audio thread, not when asyncio gets to it.
            self._loop.call_soon_threadsafe(self._enqueue, in_data, time.time())
        return (None, pyaudio.paContinue)

    def _enqueue(self, data: bytes, t: float) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self.dropped += 1
            from core.metrics import metrics
            metrics().audio_chunks_dropped_total.labels(reason="queue_full").inc()
        self._queue.put_nowait((data, t))

    async def read(self, timeout: float = 0.2) -> Optional[tuple[bytes, float]]:
        """Next ``(chunk, capture_time)``, or None on timeout."""
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    @property
    def backlog(self) -> int:
        return self._queue.qsize()

    def close(self) -> None:
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


class PcmPlayer:
    """Non-blocking PCM playback.

    ``write()`` appends to an internal buffer and returns immediately; the
    PortAudio callback thread drains it, emitting silence on underrun.
    """

    def __init__(self, *, sample_rate: int, chunk_size: int, device_index: Optional[int] = None) -> None:
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device_index = device_index
        self._buf: deque[bytes] = deque()
        self._buffered = 0
        self._lock = threading.Lock()
        self._pa: Any = None
        self._stream: Any = None

    def open(self) -> None:
        pyaudio = _pyaudio()
        self._pa = pyaudio.PyAudio()
        kwargs: dict[str, Any] = dict(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._callback,
        )
        if self.device_index is not None:
            kwargs["output_device_index"] = self.device_index
        self._stream = self._pa.open(**kwargs)
        self._stream.start_stream()

    def write(self, pcm: bytes) -> None:
        if not pcm:
            return
        if len(pcm) % SAMPLE_WIDTH:
            pcm = pcm[:-1]
        with self._lock:
            self._buf.append(pcm)
            self._buffered += len(pcm)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()
            self._buffered = 0

    @property
    def buffered_seconds(self) -> float:
        return self._buffered / SAMPLE_WIDTH / self.sample_rate

    def _read(self, nbytes: int) -> bytes:
        out = bytearray()
        with self._lock:
            while self._buf and len(out) < nbytes:
                head = self._buf[0]
                need = nbytes - len(out)
                if len(head) <= need:
                    out.extend(self._buf.popleft())
                else:
                    out.extend(head[:need])
                    self._buf[0] = head[need:]
            self._buffered -= len(out)
        if len(out) < nbytes:
            out.extend(b"\x00" * (nbytes - len(out)))
        return bytes(out)

    def _callback(self, in_data: Any, frame_count: int, time_info: dict, status: int) -> tuple[bytes, int]:
        pyaudio = _pyaudio()
        return (self._read(frame_count * SAMPLE_WIDTH), pyaudio.paContinue)

    def close(self) -> None:
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


def play_pcm_blocking(pcm: bytes, *, sample_rate: int, device_index: Optional[int] = None) -> None:
    """Play PCM to completion (run in an executor thread)."""
    pyaudio = _pyaudio()
    pa = pyaudio.PyAudio()
    try:
        kwargs: dict[str, Any] = dict(format=pyaudio.paInt16, channels=1, rate=sample_rate, output=True)
        if device_index is not None:
            kwargs["output_device_index"] = device_index
        stream = pa.open(**kwargs)
        try:
            stream.write(pcm)
        finally:
            stream.stop_stream()
            stream.close()
    finally:
        pa.terminate()
