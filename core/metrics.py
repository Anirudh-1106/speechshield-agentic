"""
Prometheus metrics module for SpeechShield.

Provides:
  • Latency histograms (ASR, rewrite, toxicity, TTS, per-stage, end-to-end)
  • Counter metrics (detections, fallbacks, dropped chunks/events, restarts)
  • Gauge metrics (queue depths, active speakers, LLM circuit state)
  • Health status tracking

Usage:
    from core.metrics import metrics

    with metrics().asr_latency_seconds.time():
        # Do ASR work
        pass

    metrics().toxicity_detections_total.labels(severity="high").inc()
"""

from typing import Any, Optional

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    CollectorRegistry,
    REGISTRY,
)

# Stage latency buckets: fine resolution below 1 s where the budget lives.
_STAGE_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)


class Metrics:
    """Prometheus metrics for SpeechShield system."""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics.

        Args:
            registry: Prometheus registry (None = default global registry)
        """
        self.registry = registry or REGISTRY
        r = self._get_or_create

        # =====================================================================
        # Latency Histograms (in seconds)
        # =====================================================================

        self.asr_latency_seconds = r(
            Histogram, "speechshield_asr_latency_seconds",
            "ASR (speech-to-text) inference latency",
            buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
        )
        self.toxicity_latency_seconds = r(
            Histogram, "speechshield_toxicity_latency_seconds",
            "Toxicity detection inference latency",
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0, 5.0),
        )
        self.rewrite_latency_seconds = r(
            Histogram, "speechshield_rewrite_latency_seconds",
            "LLM rewrite inference latency",
            buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
        )
        self.tts_latency_seconds = r(
            Histogram, "speechshield_tts_latency_seconds",
            "Text-to-speech synthesis latency",
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 15.0),
        )
        self.pipeline_latency_seconds = r(
            Histogram, "speechshield_pipeline_latency_seconds",
            "End-to-end latency from end of speech to moderation verdict",
            buckets=_STAGE_BUCKETS,
        )
        self.stage_latency_seconds = r(
            Histogram, "speechshield_stage_latency_seconds",
            "Latency of each pipeline stage, measured from end of speech",
            labelnames=["stage"],
            buckets=_STAGE_BUCKETS,
        )
        self.speaker_identification_latency_seconds = r(
            Histogram, "speechshield_speaker_identification_latency_seconds",
            "Diarization latency",
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
        )
        self.interceptor_hold_seconds = r(
            Histogram, "speechshield_interceptor_hold_seconds",
            "How long the interceptor held audio before releasing it",
            buckets=_STAGE_BUCKETS,
        )

        # =====================================================================
        # Counters
        # =====================================================================

        self.toxicity_detections_total = r(
            Counter, "speechshield_toxicity_detections_total",
            "Total number of toxic utterances detected",
            labelnames=["severity"],  # low, medium, high
        )
        self.rewrite_attempts_total = r(
            Counter, "speechshield_rewrite_attempts_total",
            "Total rewrite attempts",
            labelnames=["status"],  # success, timeout, error
        )
        self.rewrite_fallback_total = r(
            Counter, "speechshield_rewrite_fallback_total",
            "Total times fallback rewriting was used",
            labelnames=["reason"],  # llm_unavailable, llm_failed, guardrail, circuit_open
        )
        self.rewrite_guardrail_rejections_total = r(
            Counter, "speechshield_rewrite_guardrail_rejections_total",
            "LLM rewrites rejected by safety guardrails",
            labelnames=["reason"],
        )
        self.utterances_processed_total = r(
            Counter, "speechshield_utterances_processed_total",
            "Total utterances processed",
            labelnames=["status"],  # success, error
        )
        self.asr_errors_total = r(
            Counter, "speechshield_asr_errors_total",
            "Total ASR errors",
            labelnames=["reason"],  # timeout, no_speech, decode_error, hallucination
        )
        self.speaker_changes_total = r(
            Counter, "speechshield_speaker_changes_total",
            "Total speaker changes",
        )
        self.audio_chunks_dropped_total = r(
            Counter, "speechshield_audio_chunks_dropped_total",
            "Total audio chunks dropped",
            labelnames=["reason"],
        )
        self.event_bus_dropped_total = r(
            Counter, "speechshield_event_bus_dropped_total",
            "Events shed because a subscriber's inbox was full",
            labelnames=["event_type", "subscriber"],
        )
        self.handler_errors_total = r(
            Counter, "speechshield_handler_errors_total",
            "Exceptions raised by event handlers",
            labelnames=["subscriber"],
        )
        self.agent_restarts_total = r(
            Counter, "speechshield_agent_restarts_total",
            "Times the supervisor restarted a crashed agent",
            labelnames=["agent"],
        )
        self.api_requests_total = r(
            Counter, "speechshield_api_requests_total",
            "Moderation API requests",
            labelnames=["endpoint", "status"],
        )

        # =====================================================================
        # Gauges (current state)
        # =====================================================================

        self.audio_queue_depth_samples = r(
            Gauge, "speechshield_audio_queue_depth_samples",
            "Current audio chunk queue depth",
        )
        self.asr_queue_depth_samples = r(
            Gauge, "speechshield_asr_queue_depth_samples",
            "Current ASR work queue depth",
        )
        self.toxicity_queue_depth_samples = r(
            Gauge, "speechshield_toxicity_queue_depth_samples",
            "Current toxicity detection queue depth",
        )
        self.rewrite_queue_depth_samples = r(
            Gauge, "speechshield_rewrite_queue_depth_samples",
            "Current rewrite work queue depth",
        )
        self.active_speakers = r(
            Gauge, "speechshield_active_speakers",
            "Number of active speakers",
        )
        self.pipeline_pass_counter = r(
            Gauge, "speechshield_pipeline_pass_counter",
            "Total pipeline passes completed",
        )
        self.avg_latency_ms = r(
            Gauge, "speechshield_avg_latency_ms",
            "Rolling average pipeline latency in milliseconds",
        )
        self.llm_circuit_open = r(
            Gauge, "speechshield_llm_circuit_open",
            "1 while the LLM circuit breaker is open (rules-only rewriting)",
        )
        self.interceptor_backlog_seconds = r(
            Gauge, "speechshield_interceptor_backlog_seconds",
            "Audio currently held by the interceptor, in seconds",
        )

        # =====================================================================
        # Summaries (quantile tracking)
        # =====================================================================

        self.pipeline_latency_quantiles = r(
            Summary, "speechshield_pipeline_latency_quantiles_seconds",
            "Pipeline latency quantiles (p50, p95, p99)",
        )

        # =====================================================================
        # Interceptor metrics
        # =====================================================================

        self.interceptor_chunks_passed = r(
            Counter, "speechshield_interceptor_chunks_passed_total",
            "Audio chunks passed through (clean)",
        )
        self.interceptor_chunks_replaced = r(
            Counter, "speechshield_interceptor_chunks_replaced_total",
            "Audio chunks censored (bleeped, muted or replaced with TTS)",
        )
        self.interceptor_edits_total = r(
            Counter, "speechshield_interceptor_edits_total",
            "Toxic words censored by the interceptor, by method",
            labelnames=["kind"],  # bleep, mute, pending, replace, cut, whole, clone_timeout
        )
        self.voice_clone_latency_seconds = r(
            Histogram, "speechshield_voice_clone_latency_seconds",
            "Time to synthesise a replacement word in the speaker's voice",
            labelnames=["cache"],  # hit, miss
            buckets=_STAGE_BUCKETS,
        )
        self.interceptor_segments_total = r(
            Counter, "speechshield_interceptor_segments_total",
            "Utterances released by the interceptor, by action taken",
            labelnames=["action"],  # passed, bleeped, muted, replaced, timeout_muted, timeout_passed
        )

    def _get_or_create(self, cls: type, name: str, documentation: str, **kwargs: Any) -> Any:
        """Create a collector, or return the already-registered one.

        Re-initialisation happens in tests and when modules are reloaded;
        prometheus_client refuses duplicate names, so reuse the existing one.
        """
        try:
            return cls(name=name, documentation=documentation, registry=self.registry, **kwargs)
        except ValueError:
            existing = self.registry._names_to_collectors.get(name)  # type: ignore[attr-defined]
            if existing is None:
                raise
            return existing


# Global metrics instance
_metrics_instance: Optional[Metrics] = None


def metrics() -> Metrics:
    """Get global metrics instance (lazy-loaded)."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = Metrics()
    return _metrics_instance


def reset_metrics() -> None:
    """Reset metrics instance (for testing)."""
    global _metrics_instance
    _metrics_instance = None
