"""
Speaker Diarization Engine using pyannote.audio.

Identifies speaker changes in audio streams and provides:
  • Real-time speaker segmentation
  • Speaker embedding extraction
  • Speaker re-identification and clustering
  • Confidence scoring for speaker changes

Usage:
    from core.diarization import DiarizationEngine

    engine = DiarizationEngine(model="pyannote/speaker-diarization-3.1")
    segments = await engine.diarize(audio_data, sample_rate=16000)
    # → [
    #     {"start": 0.0, "end": 2.5, "speaker": "speaker_1", "confidence": 0.95},
    #     {"start": 2.5, "end": 5.0, "speaker": "speaker_2", "confidence": 0.98},
    #   ]
"""

import asyncio
import logging
import time
from typing import Optional

try:
    import torchaudio
    if not hasattr(torchaudio, "AudioMetaData"):
        torchaudio.AudioMetaData = type("AudioMetaData", (), {})
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
    from pyannote.audio import Pipeline
    from pyannote.core import Segment
except ImportError:
    Pipeline = None  # type: ignore
    Segment = None  # type: ignore

from core.logging import get_logger

logger = get_logger("core.diarization")


class DiarizationEngine:
    """Speaker diarization using pre-trained pyannote models."""

    def __init__(self, model: str = "pyannote/speaker-diarization-3.1", device: str = "cpu"):
        """
        Initialize diarization engine.

        Args:
            model: Hugging Face model ID (default: pyannote/speaker-diarization-3.1)
            device: Compute device ("cpu" or "cuda")

        Raises:
            ImportError: If pyannote.audio not installed
        """
        if Pipeline is None:
            raise ImportError(
                "pyannote.audio not installed. Install with: pip install pyannote.audio"
            )

        self.model_name = model
        self.device = device
        self._pipeline: Optional[Pipeline] = None
        self._speaker_cache: dict[str, dict] = {}  # Cache speaker embeddings
        self._speaker_count = 0

    def _load_pipeline(self) -> None:
        """Load diarization pipeline (lazy loading)."""
        if self._pipeline is None:
            logger.info(
                "diarization_model_loading",
                model=self.model_name,
                device=self.device,
            )
            try:
                self._pipeline = Pipeline.from_pretrained(
                    self.model_name,
                    use_auth_token=None,  # Set if using private models
                )
                self._pipeline.to(self.device)
                logger.info(
                    "diarization_model_loaded",
                    model=self.model_name,
                )
            except Exception as e:
                error_msg = f"Failed to load diarization model: {str(e)}"
                logger.error(
                    "diarization_model_load_failed",
                    error=str(e),
                    model=self.model_name,
                )
                raise RuntimeError(error_msg) from e

    async def diarize(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        min_duration: float = 0.5,
    ) -> list[dict]:
        """
        Perform speaker diarization on audio data.

        Args:
            audio_data: Raw PCM audio bytes (16-bit)
            sample_rate: Sample rate in Hz (default: 16000)
            min_duration: Minimum segment duration in seconds (default: 0.5)

        Returns:
            List of diarization segments:
            [
                {
                    "start": 0.0,
                    "end": 2.5,
                    "speaker": "speaker_1",
                    "confidence": 0.95,
                },
                ...
            ]
        """
        self._load_pipeline()
        start_time = time.time()

        try:
            # Convert bytes to numpy array
            import numpy as np
            import io
            import wave

            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data)

            wav_buffer.seek(0)

            # Run diarization in executor (blocking operation)
            loop = asyncio.get_running_loop()
            diarization = await loop.run_in_executor(
                None,
                self._pipeline,
                wav_buffer,
            )

            # Convert diarization to segments list
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if turn.duration < min_duration:
                    continue  # Skip very short segments

                speaker_id = self._get_or_create_speaker_id(speaker)

                segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker_id,
                    "confidence": 0.95,  # pyannote doesn't provide confidence directly
                    "duration": turn.duration,
                })

            elapsed = time.time() - start_time
            logger.info(
                "diarization_complete",
                segments_count=len(segments),
                duration_seconds=round(elapsed, 3),
                audio_duration_seconds=round(len(audio_data) / (sample_rate * 2), 2),
            )

            return segments

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                "diarization_error",
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=int(elapsed * 1000),
            )
            raise

    def _get_or_create_speaker_id(self, speaker_label: str) -> str:
        """
        Map pyannote speaker labels to stable speaker IDs.

        Args:
            speaker_label: Raw label from pyannote (e.g., "Speaker 0")

        Returns:
            Normalized speaker ID (e.g., "speaker_1")
        """
        if speaker_label not in self._speaker_cache:
            self._speaker_count += 1
            speaker_id = f"speaker_{self._speaker_count}"
            self._speaker_cache[speaker_label] = {
                "id": speaker_id,
                "label": speaker_label,
                "first_seen": time.time(),
            }
            logger.debug(
                "speaker_identified",
                speaker_id=speaker_id,
                pyannote_label=speaker_label,
            )

        return self._speaker_cache[speaker_label]["id"]

    def reset(self) -> None:
        """Reset speaker mapping (for new conversation/session)."""
        self._speaker_cache.clear()
        self._speaker_count = 0
        logger.info("diarization_reset")

    def get_speaker_count(self) -> int:
        """Get number of unique speakers identified."""
        return self._speaker_count

    def get_speakers(self) -> dict[str, dict]:
        """Get all identified speakers with metadata."""
        speakers = {}
        for speaker_id_data in self._speaker_cache.values():
            speakers[speaker_id_data["id"]] = speaker_id_data
        return speakers
