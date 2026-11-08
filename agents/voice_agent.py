"""
VoiceAgent — Text-to-Speech synthesis using Edge TTS.

Receives rewrite_result events, synthesises speech with Microsoft Edge TTS
entirely in memory (nothing is written to disk), and emits
audio_output_ready events carrying 16-bit mono PCM.

  • Standard mode    — plays the rewrite (and, if ``voice.speak_clean``,
                       clean utterances too) on the default output device.
  • Interceptor mode — only synthesises toxic utterances when
                       ``interceptor.censor_mode == "tts"``; the
                       AudioInterceptor splices the PCM into the call audio.

Metrics:
  - speechshield_tts_latency_seconds — TTS synthesis latency
"""

import asyncio
import io
import time
from typing import TYPE_CHECKING, Optional

from core.event_bus import Event
from core.logging import get_logger
from core.metrics import metrics
from core.health import health_checks

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.state import GlobalState

logger = get_logger("agents.voice_agent")


class VoiceAgent:
    """TTS agent — converts rewritten text into audible speech."""

    def __init__(
        self,
        event_bus: "EventBus",
        state: "GlobalState",
        *,
        play_audio: bool = True,
    ) -> None:
        self.bus = event_bus
        self.state = state
        self._play_audio = play_audio
        self._running = False
        self._stopped = asyncio.Event()
        self.voice = state.settings.voice.voice
        self.sample_rate = state.settings.audio.sample_rate

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._stopped.clear()
        self._running = True
        self.bus.subscribe("rewrite_result", self._handle_rewrite)
        health_checks().mark_ready("VoiceAgent")
        logger.info(
            "voice_agent_started",
            tts_provider=self.state.settings.voice.provider,
            voice=self.voice,
            subscribed_to="rewrite_result",
        )

        await self._stopped.wait()

    def stop(self) -> None:
        self._running = False
        self._stopped.set()
        health_checks().mark_starting("VoiceAgent")

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    def _should_synthesize(self, was_toxic: bool, source: str) -> bool:
        if source == "api":
            return False
        icfg = self.state.settings.interceptor
        if icfg.enabled:
            return was_toxic and icfg.censor_mode == "tts"
        return was_toxic or self.state.settings.voice.speak_clean

    async def _handle_rewrite(self, event: Event) -> None:
        text: str = event.data.get("rewritten_text", "")
        was_toxic: bool = event.data.get("was_toxic", False)
        cid = event.correlation_id
        if not text or not self._should_synthesize(was_toxic, event.data.get("source", "audio")):
            return

        pcm = await self._synthesize(text, cid)
        stage_ms = self.state.mark_stage(cid, "tts")
        metrics().stage_latency_seconds.labels(stage="tts").observe(stage_ms / 1000.0)

        # Emitted even on failure so the interceptor can fall back to muting.
        await self.bus.emit(Event("audio_output_ready", {
            "text": text,
            "was_toxic": was_toxic,
            "tts_audio_bytes": pcm,
            "sample_rate": self.sample_rate,
            "timestamp": time.time(),
        }, correlation_id=cid))

        if pcm and self._play_audio and not self.state.settings.interceptor.enabled:
            from core.audio_io import play_pcm_blocking

            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(
                    None,
                    lambda: play_pcm_blocking(pcm, sample_rate=self.sample_rate, device_index=self.state.settings.audio.output_device),
                )
            except Exception as exc:
                logger.error("voice_playback_error", correlation_id=cid, error=str(exc))

    # ------------------------------------------------------------------
    # TTS synthesis (async — edge-tts is natively async)
    # ------------------------------------------------------------------

    async def _synthesize(self, text: str, correlation_id: str = "") -> Optional[bytes]:
        """Synthesise *text* to 16-bit mono PCM at the pipeline sample rate."""
        import edge_tts

        m = metrics()
        start_time = time.time()
        try:
            with m.tts_latency_seconds.time():
                mp3 = bytearray()
                communicate = edge_tts.Communicate(text, self.voice)

                async def _collect() -> None:
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            mp3.extend(chunk["data"])

                await asyncio.wait_for(_collect(), timeout=self.state.settings.voice.timeout_seconds)
                loop = asyncio.get_running_loop()
                pcm = await loop.run_in_executor(None, self._mp3_to_pcm, bytes(mp3), self.sample_rate)
            logger.debug("tts_synthesis_complete", correlation_id=correlation_id, duration_ms=int((time.time() - start_time) * 1000))
            return pcm
        except Exception as exc:
            logger.error(
                "tts_synthesis_failed",
                correlation_id=correlation_id,
                error=str(exc) or type(exc).__name__,
                error_type=type(exc).__name__,
                duration_ms=int((time.time() - start_time) * 1000),
            )
            health_checks().mark_degraded("VoiceAgent", f"TTS failed: {type(exc).__name__}")
            return None

    # ------------------------------------------------------------------
    # PCM conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _mp3_to_pcm(mp3: bytes, sample_rate: int = 16000) -> bytes:
        """Decode MP3 bytes to raw 16-bit mono PCM (needs ffmpeg on PATH)."""
        if not mp3:
            raise ValueError("TTS returned no audio")
        from pydub import AudioSegment  # type: ignore[import-untyped]

        seg = AudioSegment.from_file(io.BytesIO(mp3), format="mp3")
        seg = seg.set_frame_rate(sample_rate).set_channels(1).set_sample_width(2)
        return seg.raw_data
