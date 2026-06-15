"""
GlobalState — Shared state accessible by all agents.

Stores speaker toxicity trends, latency tracking, system configuration,
and a dashboard log for the Streamlit UI.
Thread-safe through asyncio (single-threaded concurrency).
"""

import time
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.settings import Settings

logger = logging.getLogger("GlobalState")


@dataclass
class SpeakerProfile:
    """Per-speaker toxicity trend information."""
    total_utterances: int = 0
    toxic_count: int = 0
    last_toxic_time: float = 0.0
    consecutive_toxic: int = 0
    rewrite_strictness: float = 1.0  # 1.0 = normal, higher = stricter

    @property
    def toxicity_ratio(self) -> float:
        if self.total_utterances == 0:
            return 0.0
        return self.toxic_count / self.total_utterances


@dataclass
class LatencyRecord:
    """Single pipeline pass latency."""
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def elapsed_ms(self) -> float:
        if self.end_time <= 0:
            return 0.0
        return (self.end_time - self.start_time) * 1000.0


@dataclass
class DashboardRow:
    """One row in the live dashboard feed."""
    timestamp: float
    transcript: str
    toxic: bool
    confidence: float
    rewritten: str
    speaker: str
    latency_ms: float = 0.0


class GlobalState:
    """Centralized shared state for the multi-agent system."""

    def __init__(self, settings: "Settings | None" = None) -> None:
        """
        Initialize global state.

        Args:
            settings: Configuration settings object. If None, loads from environment.
        """
        # Load settings if not provided
        if settings is None:
            from config.settings import settings as load_settings
            settings = load_settings()
        self.settings = settings

        # Speaker toxicity profiles (keyed by speaker id / "default")
        self.speakers: dict[str, SpeakerProfile] = {}

        # Latency tracking — rolling window of last N pipeline passes
        self._latency_window: list[LatencyRecord] = []
        self._max_latency_records: int = settings.pipeline.latency_window_size

        # System configuration (legacy dict access, populated from settings)
        self.config: dict = {
            "latency_budget_ms": settings.pipeline.latency_budget_ms,
            "default_rewrite_strictness": settings.pipeline.default_rewrite_strictness,
            "toxicity_threshold": settings.toxicity.threshold,
            "sample_rate": settings.audio.sample_rate,
            "channels": settings.audio.channels,
            "chunk_size": settings.audio.chunk_size,
        }

        # Pipeline pass counter
        self.pass_counter: int = 0

        # Dashboard feed — recent pipeline results (capped)
        self.dashboard_log: list[DashboardRow] = []
        self._max_dashboard_rows: int = settings.dashboard.max_rows

        logger.info(
            "GlobalState initialized with settings env=%s latency_budget_ms=%.0f threshold=%.2f",
            settings.env,
            settings.pipeline.latency_budget_ms,
            settings.toxicity.threshold,
        )

    # ------------------------------------------------------------------
    # Speaker helpers
    # ------------------------------------------------------------------

    def get_speaker(self, speaker_id: str = "default") -> SpeakerProfile:
        if speaker_id not in self.speakers:
            self.speakers[speaker_id] = SpeakerProfile()
        return self.speakers[speaker_id]

    def record_utterance(self, speaker_id: str, is_toxic: bool) -> None:
        profile = self.get_speaker(speaker_id)
        profile.total_utterances += 1
        if is_toxic:
            profile.toxic_count += 1
            profile.consecutive_toxic += 1
            profile.last_toxic_time = time.time()
            # Escalate rewrite strictness on repeated toxicity
            profile.rewrite_strictness = min(3.0, 1.0 + 0.5 * profile.consecutive_toxic)
        else:
            profile.consecutive_toxic = 0
            profile.rewrite_strictness = max(1.0, profile.rewrite_strictness - 0.25)
        logger.debug(
            "Speaker '%s': utterances=%d toxic=%d consecutive=%d strictness=%.2f",
            speaker_id,
            profile.total_utterances,
            profile.toxic_count,
            profile.consecutive_toxic,
            profile.rewrite_strictness,
        )

    # ------------------------------------------------------------------
    # Latency helpers
    # ------------------------------------------------------------------

    def start_pass(self) -> LatencyRecord:
        self.pass_counter += 1
        rec = LatencyRecord(start_time=time.time())
        self._latency_window.append(rec)
        if len(self._latency_window) > self._max_latency_records:
            self._latency_window.pop(0)
        return rec

    def end_pass(self, record: LatencyRecord) -> float:
        record.end_time = time.time()
        elapsed = record.elapsed_ms
        logger.debug("Pipeline pass #%d latency: %.1f ms", self.pass_counter, elapsed)
        return elapsed

    @property
    def avg_latency_ms(self) -> float:
        completed = [r for r in self._latency_window if r.end_time > 0]
        if not completed:
            return 0.0
        return sum(r.elapsed_ms for r in completed) / len(completed)

    # ------------------------------------------------------------------
    # Dashboard helpers
    # ------------------------------------------------------------------

    def add_dashboard_row(self, row: DashboardRow) -> None:
        self.dashboard_log.append(row)
        if len(self.dashboard_log) > self._max_dashboard_rows:
            self.dashboard_log = self.dashboard_log[-self._max_dashboard_rows:]

    def get_dashboard_snapshot(self) -> dict:
        """Return a serializable snapshot for the Streamlit UI."""
        speakers_data = {}
        for sid, p in self.speakers.items():
            speakers_data[sid] = {
                "total_utterances": p.total_utterances,
                "toxic_count": p.toxic_count,
                "toxicity_ratio": round(p.toxicity_ratio, 3),
                "consecutive_toxic": p.consecutive_toxic,
                "rewrite_strictness": round(p.rewrite_strictness, 2),
            }
        return {
            "pass_counter": self.pass_counter,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "speakers": speakers_data,
            "log": [
                {
                    "time": time.strftime("%H:%M:%S", time.localtime(r.timestamp)),
                    "transcript": r.transcript,
                    "toxic": r.toxic,
                    "confidence": round(r.confidence, 3),
                    "rewritten": r.rewritten,
                    "speaker": r.speaker,
                    "latency_ms": round(r.latency_ms, 1),
                }
                for r in self.dashboard_log[-100:]  # last 100 for display
            ],
        }
