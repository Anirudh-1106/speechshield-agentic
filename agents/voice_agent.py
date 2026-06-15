"""
VoiceAgent — Text-to-Speech synthesis using Edge TTS.

Receives rewrite_result events, synthesises speech with Microsoft Edge TTS,
and emits audio_output_ready events.  Optionally plays the audio via PyAudio.

Metrics:
  - speechshield_tts_latency_seconds — TTS synthesis latency
"""

import asyncio
import os
import time
from typing import TYPE_CHECKING

import edge_tts
import pyaudio

from core.event_bus import Event
from core.logging import get_logger
from core.metrics import metrics
from core.health import health_checks

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.state import GlobalState

logger = get_logger("agents.voice_agent")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "audio")

# Natural-sounding English voice
VOICE = "en-US-GuyNeural"


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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        self.bus.subscribe("rewrite_result", self._handle_rewrite)
        health_checks().mark_ready("VoiceAgent")
        logger.info(
            "voice_agent_started",
            tts_provider="edge-tts",
            voice=VOICE,
            subscribed_to="rewrite_result",
        )

        while self._running:
            await asyncio.sleep(0.1)

    def stop(self) -> None:
        self._running = False
        health_checks().mark_starting("VoiceAgent")

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    async def _handle_rewrite(self, event: Event) -> None:
        text: str = event.data.get("rewritten_text", "")
        if not text:
            return

        try:
            wav_path = await self._synthesize(text, event.correlation_id)

            if wav_path:
                logger.info(
                    "voice_audio_synthesized",
                    correlation_id=event.correlation_id,
                    wav_path=wav_path,
                    text_length=len(text),
                )
                await self.bus.emit(
                    Event(
                        "audio_output_ready",
                        {
                            "wav_path": wav_path,
                            "text": text,
                            "timestamp": time.time(),
                        },
                        correlation_id=event.correlation_id,
                    )
                )

                if self._play_audio:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, self._play, wav_path)
        except Exception as e:
            logger.error(
                "voice_synthesis_error",
                correlation_id=event.correlation_id,
                error=str(e),
                error_type=type(e).__name__,
                text_length=len(text),
            )
            health_checks().mark_error("VoiceAgent", f"Synthesis error: {str(e)}")

    # ------------------------------------------------------------------
    # TTS synthesis (async — edge-tts is natively async)
    # ------------------------------------------------------------------

    async def _synthesize(self, text: str, correlation_id: str = "") -> str | None:
        """Generate audio file from text using Edge TTS."""
        m = metrics()
        start_time = time.time()
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filename = f"output_{int(time.time() * 1000)}.mp3"
        out_path = os.path.join(OUTPUT_DIR, filename)
        
        try:
            with m.tts_latency_seconds.time():
                communicate = edge_tts.Communicate(text, VOICE)
                await communicate.save(out_path)
            
            elapsed = time.time() - start_time
            logger.debug(
                "tts_synthesis_complete",
                correlation_id=correlation_id,
                duration_ms=int(elapsed * 1000),
            )
            return out_path
        except Exception as exc:
            elapsed = time.time() - start_time
            logger.error(
                "tts_synthesis_failed",
                correlation_id=correlation_id,
                error=str(exc),
                error_type=type(exc).__name__,
                duration_ms=int(elapsed * 1000),
            )
            health_checks().mark_error("VoiceAgent", f"TTS failed: {str(exc)}")
            return None

    # ------------------------------------------------------------------
    # Playback (runs in executor thread)
    # ------------------------------------------------------------------

    @staticmethod
    def _play(audio_path: str) -> None:
        """Play an audio file through the default output device.

        edge-tts outputs MP3.  We decode via pydub if available,
        otherwise fall back to the OS default player.
        """
        try:
            from pydub import AudioSegment  # type: ignore[import-untyped]
            from pydub.playback import play  # type: ignore[import-untyped]

            seg = AudioSegment.from_file(audio_path)
            play(seg)
        except ImportError:
            logger.warning("pydub not installed — falling back to OS player.")
            if os.name == "nt":
                os.startfile(audio_path)  # type: ignore[attr-defined]
            else:
                import subprocess

                subprocess.Popen(["xdg-open", audio_path])
        except Exception as exc:
            logger.error("Playback error: %s", exc)
