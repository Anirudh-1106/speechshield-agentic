"""
Streaming speaker identification.

Running a full batch diarization pipeline on short windows does not work
for live audio: each run labels speakers from scratch, so "SPEAKER_00" in
one window has nothing to do with "SPEAKER_00" in the next.  Instead we:

  1. embed each utterance with a speaker-embedding model (pyannote), and
  2. cluster embeddings *online* — each utterance joins the most similar
     known speaker (cosine similarity ≥ threshold) or starts a new one.

``SpeakerTracker`` is pure numpy and model-agnostic; ``DiarizationEngine``
adds the pyannote embedding backend.

Usage:
    engine = DiarizationEngine(settings.diarization)
    speaker_id, similarity = engine.identify(pcm_bytes, sample_rate=16000)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from core.logging import get_logger

if TYPE_CHECKING:
    from config.settings import DiarizationSettings

logger = get_logger("core.diarization")


@dataclass
class _Speaker:
    id: str
    centroid: np.ndarray
    count: int = 1
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


class SpeakerTracker:
    """Online clustering of speaker embeddings by cosine similarity."""

    def __init__(self, threshold: float = 0.5, max_speakers: int = 8) -> None:
        self.threshold = threshold
        self.max_speakers = max_speakers
        self._speakers: list[_Speaker] = []

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def assign(self, embedding: np.ndarray) -> tuple[str, float, bool]:
        """Return ``(speaker_id, similarity, is_new)`` for an utterance embedding."""
        e = self._unit(embedding)
        if self._speakers:
            sims = [float(np.dot(e, s.centroid)) for s in self._speakers]
            best = int(np.argmax(sims))
            if sims[best] >= self.threshold or len(self._speakers) >= self.max_speakers:
                spk = self._speakers[best]
                # Running mean keeps the centroid stable as evidence accumulates.
                spk.centroid = self._unit(spk.centroid * spk.count + e)
                spk.count += 1
                spk.last_seen = time.time()
                return spk.id, sims[best], False
        spk = _Speaker(id=f"speaker_{len(self._speakers) + 1}", centroid=e)
        self._speakers.append(spk)
        return spk.id, 1.0, True

    def reset(self) -> None:
        self._speakers.clear()

    @property
    def count(self) -> int:
        return len(self._speakers)

    def speakers(self) -> dict[str, dict]:
        return {
            s.id: {"utterances": s.count, "first_seen": s.first_seen, "last_seen": s.last_seen}
            for s in self._speakers
        }


class DiarizationEngine:
    """Speaker-embedding model + online tracker."""

    def __init__(self, settings: "DiarizationSettings") -> None:
        self.settings = settings
        self.tracker = SpeakerTracker(settings.similarity_threshold, settings.max_speakers)
        self._inference: Any = None

    def load(self) -> None:
        """Load the embedding model (raises with an actionable message on failure)."""
        if self._inference is not None:
            return
        try:
            import torchaudio
            # pyannote 3.x references torchaudio APIs removed in torchaudio 2.9+
            if not hasattr(torchaudio, "AudioMetaData"):
                torchaudio.AudioMetaData = type("AudioMetaData", (), {})  # type: ignore[attr-defined]
            if not hasattr(torchaudio, "list_audio_backends"):
                torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]
            from pyannote.audio import Inference, Model
        except ImportError as e:
            raise RuntimeError("pyannote.audio is not installed: pip install -r requirements-diarization.txt") from e

        token = self.settings.hf_token or os.getenv("HF_TOKEN")
        logger.info("diarization_model_loading", model=self.settings.embedding_model, token=bool(token))
        try:
            model = Model.from_pretrained(self.settings.embedding_model, use_auth_token=token)
        except Exception as e:
            raise RuntimeError(
                f"Could not load '{self.settings.embedding_model}': {e}. If the model is gated, accept its terms "
                "on huggingface.co and set HF_TOKEN."
            ) from e
        if model is None:
            raise RuntimeError(f"'{self.settings.embedding_model}' needs a HuggingFace token (set HF_TOKEN).")
        self._inference = Inference(model, window="whole")
        logger.info("diarization_model_loaded", model=self.settings.embedding_model)

    def embed(self, pcm: bytes, sample_rate: int = 16000) -> np.ndarray:
        import torch

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        waveform = torch.from_numpy(audio).unsqueeze(0)
        return np.asarray(self._inference({"waveform": waveform, "sample_rate": sample_rate}))

    def identify(self, pcm: bytes, sample_rate: int = 16000) -> tuple[str, float]:
        self.load()
        speaker_id, similarity, is_new = self.tracker.assign(self.embed(pcm, sample_rate))
        if is_new:
            logger.info("new_speaker", speaker=speaker_id)
        return speaker_id, similarity

    def reset(self) -> None:
        self.tracker.reset()

    def get_speaker_count(self) -> int:
        return self.tracker.count

    def get_speakers(self) -> dict[str, dict]:
        return self.tracker.speakers()


def make_engine(settings: "DiarizationSettings") -> Optional[DiarizationEngine]:
    return DiarizationEngine(settings) if settings.enabled else None
