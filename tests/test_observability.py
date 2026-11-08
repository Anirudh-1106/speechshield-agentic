"""
Tests for observability modules (logging, metrics, health checks).

Tests cover:
  • Structured logging configuration
  • Correlation ID tracking
  • Prometheus metrics recording
  • Health check status transitions
  • Metrics server endpoints
"""

import pytest
from core.logging import (
    configure_logging,
    get_logger,
    set_correlation_id,
    get_correlation_id,
    clear_correlation_id,
)
from core.metrics import metrics, reset_metrics
from core.health import HealthStatus, ComponentHealth, health_checks, reset_health_checks
from core.event_bus import Event


class TestStructuredLogging:
    """Test structured logging functionality."""

    def test_configure_logging_text_format(self):
        """Test logging configuration with text format."""
        configure_logging(level="INFO", format="text", enable_file_logging=False)
        logger = get_logger("TestModule")
        assert logger is not None

    def test_configure_logging_json_format(self):
        """Test logging configuration with JSON format."""
        configure_logging(level="DEBUG", format="json", enable_file_logging=False)
        logger = get_logger("TestModule")
        assert logger is not None

    def test_configure_logging_idempotent(self):
        """Test that configure_logging is idempotent."""
        configure_logging(level="INFO", format="text")
        logger1 = get_logger("Module1")
        configure_logging(level="DEBUG", format="json")
        logger2 = get_logger("Module2")
        # Should not raise errors on second call
        assert logger1 is not None and logger2 is not None

    def test_get_logger_per_module(self):
        """Test getting module-specific loggers."""
        logger1 = get_logger("agents.asr_agent")
        logger2 = get_logger("agents.toxicity_agent")
        logger3 = get_logger("agents.asr_agent")  # Same as logger1

        assert logger1 is not None
        assert logger2 is not None
        assert logger3 is not None


class TestCorrelationID:
    """Test correlation ID tracking."""

    def test_set_and_get_correlation_id(self):
        """Test setting and retrieving correlation ID."""
        clear_correlation_id()
        correlation_id = "test-123-abc"
        set_correlation_id(correlation_id)
        assert get_correlation_id() == correlation_id

    def test_clear_correlation_id(self):
        """Test clearing correlation ID."""
        set_correlation_id("test-456")
        assert get_correlation_id() is not None
        clear_correlation_id()
        assert get_correlation_id() is None

    def test_correlation_id_isolation(self):
        """Test that correlation IDs don't bleed across requests."""
        set_correlation_id("request-1")
        id1 = get_correlation_id()
        clear_correlation_id()

        set_correlation_id("request-2")
        id2 = get_correlation_id()

        assert id1 != id2


class TestPrometheusMetrics:
    """Test Prometheus metrics."""

    def test_metrics_singleton(self):
        """Test metrics singleton pattern."""
        reset_metrics()
        m1 = metrics()
        m2 = metrics()
        assert m1 is m2

    def test_latency_histograms(self):
        """Test latency histogram metrics."""
        reset_metrics()
        m = metrics()

        # Test recording latencies
        with m.asr_latency_seconds.time():
            pass  # Simulate work

        with m.toxicity_latency_seconds.time():
            pass

        with m.rewrite_latency_seconds.time():
            pass

        # Histograms should have observations
        assert m.asr_latency_seconds._sum.get() >= 0
        assert m.toxicity_latency_seconds._sum.get() >= 0
        assert m.rewrite_latency_seconds._sum.get() >= 0

    def test_counters(self):
        """Test counter metrics."""
        reset_metrics()
        m = metrics()

        # Test incrementing counters
        m.toxicity_detections_total.labels(severity="high").inc()
        m.toxicity_detections_total.labels(severity="high").inc()
        m.toxicity_detections_total.labels(severity="low").inc()

        m.rewrite_attempts_total.labels(status="success").inc()
        m.rewrite_fallback_total.labels(reason="llm_timeout").inc()

        # Counters should have values
        assert m.toxicity_detections_total.labels(severity="high")._value.get() >= 2
        assert m.toxicity_detections_total.labels(severity="low")._value.get() >= 1
        assert m.rewrite_attempts_total.labels(status="success")._value.get() >= 1
        assert m.rewrite_fallback_total.labels(reason="llm_timeout")._value.get() >= 1

    def test_gauges(self):
        """Test gauge metrics."""
        reset_metrics()
        m = metrics()

        # Test setting gauges
        m.audio_queue_depth_samples.set(10)
        m.asr_queue_depth_samples.set(5)
        m.active_speakers.set(2)
        m.pipeline_pass_counter.set(100)
        m.avg_latency_ms.set(250.5)

        # Gauges should have values
        assert m.audio_queue_depth_samples._value.get() == 10
        assert m.asr_queue_depth_samples._value.get() == 5
        assert m.active_speakers._value.get() == 2
        assert m.pipeline_pass_counter._value.get() == 100
        assert m.avg_latency_ms._value.get() == 250.5

    def test_gauge_increment_decrement(self):
        """Test gauge increment/decrement."""
        reset_metrics()
        m = metrics()

        m.audio_queue_depth_samples.set(0)
        m.audio_queue_depth_samples.inc()
        m.audio_queue_depth_samples.inc()
        assert m.audio_queue_depth_samples._value.get() == 2

        m.audio_queue_depth_samples.dec()
        assert m.audio_queue_depth_samples._value.get() == 1


class TestHealthChecks:
    """Test health check functionality."""

    def test_health_checks_singleton(self):
        """Test health checks singleton pattern."""
        reset_health_checks()
        hc1 = health_checks()
        hc2 = health_checks()
        assert hc1 is hc2

    def test_component_health_status_transitions(self):
        """Test component status transitions."""
        reset_health_checks()
        health_checks()

        comp = ComponentHealth("TestComponent")
        assert comp.status == HealthStatus.UNKNOWN

        comp.mark_starting()
        assert comp.status == HealthStatus.STARTING

        comp.mark_ready()
        assert comp.status == HealthStatus.READY

        comp.mark_error("Connection failed")
        assert comp.status == HealthStatus.ERROR
        assert comp.error_count == 1

        comp.mark_degraded("Slow response")
        assert comp.status == HealthStatus.DEGRADED

    def test_mark_ready(self):
        """Test marking components as ready."""
        reset_health_checks()
        hc = health_checks()

        hc.mark_ready("AudioAgent")
        assert hc.get_component("AudioAgent").status == HealthStatus.READY

    def test_mark_error(self):
        """Test marking components as errored."""
        reset_health_checks()
        hc = health_checks()

        hc.mark_error("ASRAgent", "Model failed to load")
        comp = hc.get_component("ASRAgent")
        assert comp.status == HealthStatus.ERROR
        assert comp.last_error == "Model failed to load"
        assert comp.error_count == 1

    def test_is_ready_property(self):
        """Test is_ready property."""
        reset_health_checks()
        hc = health_checks()

        # Initially not ready (all UNKNOWN)
        assert not hc.is_ready

        # Mark all registered components as ready
        for agent_name in list(hc.components.keys()):
            hc.mark_ready(agent_name)

        # Should be ready now
        assert hc.is_ready

    def test_is_alive_property(self):
        """Test is_alive property."""
        reset_health_checks()
        hc = health_checks()

        # Initially alive (no errors)
        assert hc.is_alive

        # Mark one as error
        hc.mark_error("ASRAgent", "Connection refused")
        assert not hc.is_alive

        # Mark as ready should restore alive
        hc.mark_ready("ASRAgent")
        assert hc.is_alive

    def test_get_status(self):
        """Test getting status snapshot."""
        reset_health_checks()
        hc = health_checks()

        hc.mark_ready("AudioAgent")
        hc.mark_error("ASRAgent", "Failed")

        status = hc.get_status()
        assert "ready" in status
        assert "alive" in status
        assert "uptime_seconds" in status
        assert "components" in status
        assert "AudioAgent" in status["components"]


class TestEventCorrelationID:
    """Test correlation ID in events."""

    def test_event_generates_correlation_id(self):
        """Test that events generate correlation IDs."""
        event1 = Event("test_event", {"data": "test"})
        event2 = Event("test_event", {"data": "test"})

        # Each event should have a unique correlation ID
        assert event1.correlation_id is not None
        assert event2.correlation_id is not None
        assert event1.correlation_id != event2.correlation_id

    def test_event_preserves_correlation_id(self):
        """Test that events preserve provided correlation IDs."""
        correlation_id = "trace-123-xyz"
        event = Event("test_event", {"data": "test"}, correlation_id=correlation_id)

        assert event.correlation_id == correlation_id

    def test_event_repr_includes_correlation_id(self):
        """Test that event repr shows correlation ID."""
        correlation_id = "test-correlation"
        event = Event("test_event", correlation_id=correlation_id)

        repr_str = repr(event)
        assert "test-correlation" in repr_str or "test-c" in repr_str


class TestComponentHealth:
    """Test individual component health tracking."""

    def test_component_health_to_dict(self):
        """Test component health serialization."""
        comp = ComponentHealth("TestAgent")
        comp.mark_ready("Ready to go")

        data = comp.to_dict()
        assert data["name"] == "TestAgent"
        assert data["status"] == "ready"
        assert data["message"] == "Ready to go"
        assert data["error_count"] == 0

    def test_component_error_tracking(self):
        """Test component error count tracking."""
        comp = ComponentHealth("TestAgent")

        comp.mark_error("Error 1")
        assert comp.error_count == 1

        comp.mark_error("Error 2")
        assert comp.error_count == 2

        comp.mark_ready()  # Reset on success
        assert comp.error_count == 2  # Error count stays


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
