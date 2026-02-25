"""
VoiceAgent — Text-to-Speech synthesis using Edge TTS.

Receives rewrite_result events, synthesises speech with Microsoft Edge TTS,
and emits audio_output_ready events.  Optionally plays the audio via PyAudio.
"""

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING

import edge_tts
import pyaudio

from core.event_bus import Event

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.state import GlobalState

logger = logging.getLogger("VoiceAgent")

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
        logger.info(
            "VoiceAgent started (edge-tts, voice=%s) — listening for rewrite_result events.",
            VOICE,
        )

        while self._running:
            await asyncio.sleep(0.1)

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    async def _handle_rewrite(self, event: Event) -> None:
        text: str = event.data.get("rewritten_text", "")
        if not text:
            return

        wav_path = await self._synthesize(text)

        if wav_path:
            logger.info("Synthesized audio → %s", wav_path)
            await self.bus.emit(
                Event(
                    "audio_output_ready",
                    {
                        "wav_path": wav_path,
                        "text": text,
                        "timestamp": time.time(),
                    },
                )
            )

            if self._play_audio:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._play, wav_path)

    # ------------------------------------------------------------------
    # TTS synthesis (async — edge-tts is natively async)
    # ------------------------------------------------------------------

    async def _synthesize(self, text: str) -> str | None:
        """Generate audio file from text using Edge TTS."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filename = f"output_{int(time.time() * 1000)}.mp3"
        out_path = os.path.join(OUTPUT_DIR, filename)
        try:
            communicate = edge_tts.Communicate(text, VOICE)
            await communicate.save(out_path)
            return out_path
        except Exception as exc:
            logger.error("TTS synthesis failed: %s", exc)
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
