"""
GlobalState — Shared state accessible by all agents.

Stores speaker toxicity trends, per-utterance pipeline traces, latency
tracking, conversation history, and a dashboard log for the Streamlit UI.
Thread-safe through asyncio (single-threaded concurrency).
"""

import time
import logging
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from config.settings import Settings

logger = logging.getLogger("GlobalState")

# Pipeline stages in order, used for latency breakdowns.
STAGES = ("asr", "diarization", "toxicity", "rewrite", "tts", "released")


@dataclass
class SpeakerProfile:
    """Per-speaker toxicity trend information."""
    total_utterances: int = 0
    toxic_count: int = 0
    last_toxic_time: float = 0.0
    consecutive_toxic: int = 0
    rewrite_strictness: float = 1.0  # 1.0 = normal, higher = stricter
    last_seen: float = 0.0

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
class PipelineTrace:
    """Everything known about one utterance as it moves through the pipeline."""
    correlation_id: str
    speech_end: float = 0.0          # when the speaker stopped (latency origin)
    speech_duration: float = 0.0
    stages: dict[str, float] = field(default_factory=dict)  # stage -> ms after speech_end
    speaker: str = "default"
    transcript: str = ""
    words: list[dict] = field(default_factory=list)
    toxic: bool = False
    confidence: float = 0.0
    labels: dict[str, float] = field(default_factory=dict)
    toxic_spans: list[dict] = field(default_factory=list)
    rewritten: str = ""
    rewrite_method: str = ""
    action: str = ""                 # interceptor action: passed / bleeped / muted / replaced
    source: str = "audio"            # audio | api
    completed: bool = False


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
    correlation_id: str = ""
    rewrite_method: str = ""
    action: str = ""
    source: str = "audio"
    stages: dict[str, float] = field(default_factory=dict)
    top_label: str = ""


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

        # Per-utterance traces, bounded so memory never grows without limit
        self.traces: "OrderedDict[str, PipelineTrace]" = OrderedDict()
        self._max_traces: int = 1000
        self._stage_window: dict[str, deque[float]] = {
            s: deque(maxlen=settings.pipeline.latency_window_size) for s in STAGES
        }

        # Recent utterances per speaker — context for the LLM rewriter
        self._history: dict[str, deque[str]] = {}

        # Dashboard feed — recent pipeline results (capped)
        self.dashboard_log: list[DashboardRow] = []
        self._max_dashboard_rows: int = settings.dashboard.max_rows

        # Active tracking
        self.current_speaker: str = "default"

        # Free-form system info published to the dashboard (mode, LLM backend …)
        self.system_info: dict[str, Any] = {"started_at": time.time()}

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
            self.speakers[speaker_id] = SpeakerProfile(
                rewrite_strictness=self.settings.pipeline.default_rewrite_strictness,
            )
        return self.speakers[speaker_id]

    def record_utterance(self, speaker_id: str, is_toxic: bool) -> None:
        profile = self.get_speaker(speaker_id)
        profile.total_utterances += 1
        profile.last_seen = time.time()
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

    def add_history(self, speaker_id: str, text: str) -> None:
        turns = max(1, self.settings.rewrite.context_turns)
        self._history.setdefault(speaker_id, deque(maxlen=turns)).append(text)

    def get_history(self, speaker_id: str) -> list[str]:
        return list(self._history.get(speaker_id, ()))

    # ------------------------------------------------------------------
    # Trace helpers
    # ------------------------------------------------------------------

    def trace(self, correlation_id: str) -> PipelineTrace:
        """Get or create the trace for *correlation_id*."""
        tr = self.traces.get(correlation_id)
        if tr is None:
            tr = PipelineTrace(correlation_id=correlation_id, speech_end=time.time())
            self.traces[correlation_id] = tr
            while len(self.traces) > self._max_traces:
                self.traces.popitem(last=False)
        return tr

    def mark_stage(self, correlation_id: str, stage: str, at: Optional[float] = None) -> float:
        """Record that *stage* finished; returns ms since end of speech."""
        tr = self.trace(correlation_id)
        ms = max(0.0, ((at or time.time()) - tr.speech_end) * 1000.0)
        tr.stages[stage] = ms
        if stage in self._stage_window:
            self._stage_window[stage].append(ms)
        return ms

    def stage_stats(self) -> dict[str, dict[str, float]]:
        """p50/p95/mean latency (ms) per stage over the rolling window."""
        out: dict[str, dict[str, float]] = {}
        for stage, values in self._stage_window.items():
            if not values:
                continue
            ordered = sorted(values)
            out[stage] = {
                "p50": round(ordered[len(ordered) // 2], 1),
                "p95": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 1),
                "mean": round(sum(ordered) / len(ordered), 1),
                "count": len(ordered),
            }
        return out

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

    def record_pass(self, latency_ms: float) -> None:
        """Record a completed pipeline pass whose latency was measured elsewhere."""
        now = time.time()
        rec = self.start_pass()
        rec.start_time = now - latency_ms / 1000.0
        rec.end_time = now

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
                "last_seen": p.last_seen,
            }
        return {
            "generated_at": time.time(),
            "pass_counter": self.pass_counter,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "latency_budget_ms": self.settings.pipeline.latency_budget_ms,
            "stage_latency": self.stage_stats(),
            "system": self.system_info,
            "speakers": speakers_data,
            "log": [
                {
                    "ts": r.timestamp,
                    "time": time.strftime("%H:%M:%S", time.localtime(r.timestamp)),
                    "transcript": r.transcript,
                    "toxic": r.toxic,
                    "confidence": round(r.confidence, 3),
                    "rewritten": r.rewritten,
                    "speaker": r.speaker,
                    "latency_ms": round(r.latency_ms, 1),
                    "correlation_id": r.correlation_id,
                    "rewrite_method": r.rewrite_method,
                    "action": r.action,
                    "source": r.source,
                    "stages": {k: round(v, 1) for k, v in r.stages.items()},
                    "top_label": r.top_label,
                }
                for r in self.dashboard_log[-self._max_dashboard_rows:]
            ],
        }
