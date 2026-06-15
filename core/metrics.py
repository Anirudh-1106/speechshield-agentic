"""
Prometheus metrics module for SpeechShield.

Provides:
  • Latency histograms (ASR, rewrite, toxicity, TTS)
  • Counter metrics (detections, fallbacks, dropped chunks)
  • Gauge metrics (queue depths, active speakers)
  • Health status tracking

Usage:
    from core.metrics import metrics
    
    with metrics.asr_latency_seconds.time():
        # Do ASR work
        pass
    
    metrics.toxicity_detections_total.inc()
    metrics.queue_depth_samples.set(queue.qsize())
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    CollectorRegistry,
    REGISTRY,
)
from typing import Optional
import time


class Metrics:
    """Prometheus metrics for SpeechShield system."""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics.

        Args:
            registry: Prometheus registry (None = default global registry)
        """
        self.registry = registry or REGISTRY

        # Check if metrics already registered to avoid duplication
        try:
            # =====================================================================
            # Latency Histograms (in seconds)
            # =====================================================================

            self.asr_latency_seconds = Histogram(
                name="speechshield_asr_latency_seconds",
                documentation="ASR (speech-to-text) inference latency",
                buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
                registry=self.registry,
            )

            self.toxicity_latency_seconds = Histogram(
                name="speechshield_toxicity_latency_seconds",
                documentation="Toxicity detection inference latency",
                buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
                registry=self.registry,
            )

            self.rewrite_latency_seconds = Histogram(
                name="speechshield_rewrite_latency_seconds",
                documentation="LLM rewrite inference latency",
                buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
                registry=self.registry,
            )

            self.tts_latency_seconds = Histogram(
                name="speechshield_tts_latency_seconds",
                documentation="Text-to-speech synthesis latency",
                buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 15.0),
                registry=self.registry,
            )

            self.pipeline_latency_seconds = Histogram(
                name="speechshield_pipeline_latency_seconds",
                documentation="End-to-end pipeline latency",
                buckets=(0.1, 0.3, 0.5, 1.0, 2.0, 5.0),
                registry=self.registry,
            )

            # =====================================================================
            # Counters
            # =====================================================================

            self.toxicity_detections_total = Counter(
                name="speechshield_toxicity_detections_total",
                documentation="Total number of toxic utterances detected",
                labelnames=["severity"],  # low, medium, high
                registry=self.registry,
            )

            self.rewrite_attempts_total = Counter(
                name="speechshield_rewrite_attempts_total",
                documentation="Total rewrite attempts",
                labelnames=["status"],  # success, timeout, error
                registry=self.registry,
            )

            self.rewrite_fallback_total = Counter(
                name="speechshield_rewrite_fallback_total",
                documentation="Total times fallback rewriting was used",
                labelnames=["reason"],  # llm_timeout, llm_error, confidence_low
                registry=self.registry,
            )

            self.audio_chunks_dropped_total = Counter(
                name="speechshield_audio_chunks_dropped_total",
                documentation="Total audio chunks dropped due to queue overflow",
                registry=self.registry,
            )

            self.utterances_processed_total = Counter(
                name="speechshield_utterances_processed_total",
                documentation="Total utterances processed",
                labelnames=["status"],  # success, error
                registry=self.registry,
            )

            self.asr_errors_total = Counter(
                name="speechshield_asr_errors_total",
                documentation="Total ASR errors",
                labelnames=["reason"],  # timeout, no_speech, decode_error
                registry=self.registry,
            )

            # =====================================================================
            # Gauges (current state)
            # =====================================================================

            self.audio_queue_depth_samples = Gauge(
                name="speechshield_audio_queue_depth_samples",
                documentation="Current audio chunk queue depth",
                registry=self.registry,
            )

            self.asr_queue_depth_samples = Gauge(
                name="speechshield_asr_queue_depth_samples",
                documentation="Current ASR work queue depth",
                registry=self.registry,
            )

            self.toxicity_queue_depth_samples = Gauge(
                name="speechshield_toxicity_queue_depth_samples",
                documentation="Current toxicity detection queue depth",
                registry=self.registry,
            )

            self.rewrite_queue_depth_samples = Gauge(
                name="speechshield_rewrite_queue_depth_samples",
                documentation="Current rewrite work queue depth",
                registry=self.registry,
            )

            self.active_speakers = Gauge(
                name="speechshield_active_speakers",
                documentation="Number of active speakers",
                registry=self.registry,
            )

            self.pipeline_pass_counter = Gauge(
                name="speechshield_pipeline_pass_counter",
                documentation="Total pipeline passes completed",
                registry=self.registry,
            )

            self.avg_latency_ms = Gauge(
                name="speechshield_avg_latency_ms",
                documentation="Rolling average pipeline latency in milliseconds",
                registry=self.registry,
            )

            # =====================================================================
            # Summaries (quantile tracking)
            # =====================================================================

            self.pipeline_latency_quantiles = Summary(
                name="speechshield_pipeline_latency_quantiles_seconds",
                documentation="Pipeline latency quantiles (p50, p95, p99)",
                registry=self.registry,
            )
        except ValueError:
            # Metrics already registered, get them from registry
            # This happens in tests or when re-initializing
            self.asr_latency_seconds = self.registry._names_to_collectors.get(
                "speechshield_asr_latency_seconds"
            )
            self.toxicity_latency_seconds = self.registry._names_to_collectors.get(
                "speechshield_toxicity_latency_seconds"
            )
            self.rewrite_latency_seconds = self.registry._names_to_collectors.get(
                "speechshield_rewrite_latency_seconds"
            )
            self.tts_latency_seconds = self.registry._names_to_collectors.get(
                "speechshield_tts_latency_seconds"
            )
            self.pipeline_latency_seconds = self.registry._names_to_collectors.get(
                "speechshield_pipeline_latency_seconds"
            )
            self.toxicity_detections_total = self.registry._names_to_collectors.get(
                "speechshield_toxicity_detections_total"
            )
            self.rewrite_attempts_total = self.registry._names_to_collectors.get(
                "speechshield_rewrite_attempts_total"
            )
            self.rewrite_fallback_total = self.registry._names_to_collectors.get(
                "speechshield_rewrite_fallback_total"
            )
            self.audio_chunks_dropped_total = self.registry._names_to_collectors.get(
                "speechshield_audio_chunks_dropped_total"
            )
            self.utterances_processed_total = self.registry._names_to_collectors.get(
                "speechshield_utterances_processed_total"
            )
            self.asr_errors_total = self.registry._names_to_collectors.get(
                "speechshield_asr_errors_total"
            )
            self.audio_queue_depth_samples = self.registry._names_to_collectors.get(
                "speechshield_audio_queue_depth_samples"
            )
            self.asr_queue_depth_samples = self.registry._names_to_collectors.get(
                "speechshield_asr_queue_depth_samples"
            )
            self.toxicity_queue_depth_samples = self.registry._names_to_collectors.get(
                "speechshield_toxicity_queue_depth_samples"
            )
            self.rewrite_queue_depth_samples = self.registry._names_to_collectors.get(
                "speechshield_rewrite_queue_depth_samples"
            )
            self.active_speakers = self.registry._names_to_collectors.get(
                "speechshield_active_speakers"
            )
            self.pipeline_pass_counter = self.registry._names_to_collectors.get(
                "speechshield_pipeline_pass_counter"
            )
            self.avg_latency_ms = self.registry._names_to_collectors.get(
                "speechshield_avg_latency_ms"
            )
            self.pipeline_latency_quantiles = self.registry._names_to_collectors.get(
                "speechshield_pipeline_latency_quantiles_seconds"
            )


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
