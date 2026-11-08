"""
Tests for metrics server HTTP endpoints.

Tests cover:
  • /metrics endpoint (Prometheus format)
  • /health endpoint (component status)
  • /status endpoint (system status)
  • /ready endpoint (Kubernetes readiness)
  • /live endpoint (Kubernetes liveness)
"""

import pytest
from fastapi.testclient import TestClient
from core.metrics_server import create_metrics_app
from core.health import health_checks, reset_health_checks
from core.metrics import metrics, reset_metrics


@pytest.fixture
def app():
    """Create test FastAPI app."""
    return create_metrics_app()


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def setup_health():
    """Setup health checks for testing."""
    reset_health_checks()
    hc = health_checks()
    for agent in list(hc.components.keys()):
        hc.mark_ready(agent)
    yield hc
    reset_health_checks()


class TestMetricsEndpoint:
    """Test /metrics endpoint."""

    def test_metrics_endpoint_status(self, client):
        """Test /metrics endpoint returns 200."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_endpoint_content_type(self, client):
        """Test /metrics returns correct content type."""
        response = client.get("/metrics")
        assert "text/plain" in response.headers.get("content-type", "")

    def test_metrics_endpoint_has_prometheus_data(self, client, setup_health):
        """Test /metrics contains Prometheus data."""
        reset_metrics()
        m = metrics()

        # Record some metrics
        m.asr_latency_seconds.observe(1.5)
        m.toxicity_detections_total.labels(severity="high").inc()

        response = client.get("/metrics")
        text = response.text

        # Check for Prometheus format indicators
        assert "# HELP" in text or "# TYPE" in text
        assert "speechshield" in text

    def test_metrics_endpoint_has_latency_metrics(self, client):
        """Test /metrics includes latency metrics."""
        response = client.get("/metrics")
        text = response.text

        # Check for latency histogram names
        assert "speechshield_asr_latency_seconds" in text

    def test_metrics_endpoint_has_counter_metrics(self, client):
        """Test /metrics includes counter metrics."""
        response = client.get("/metrics")
        text = response.text

        # Check for counter names
        assert "speechshield_toxicity_detections_total" in text


class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_endpoint_status(self, client, setup_health):
        """Test /health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_endpoint_content_type(self, client, setup_health):
        """Test /health returns JSON."""
        response = client.get("/health")
        assert "application/json" in response.headers.get("content-type", "")

    def test_health_endpoint_has_required_fields(self, client, setup_health):
        """Test /health response has required fields."""
        response = client.get("/health")
        data = response.json()

        assert "ready" in data
        assert "alive" in data
        assert "uptime_seconds" in data
        assert "components" in data

    def test_health_endpoint_ready_true_when_all_ready(self, client, setup_health):
        """Test /health ready=true when all components ready."""
        response = client.get("/health")
        data = response.json()

        assert data["ready"] is True

    def test_health_endpoint_alive_true_when_no_errors(self, client, setup_health):
        """Test /health alive=true when no errors."""
        response = client.get("/health")
        data = response.json()

        assert data["alive"] is True

    def test_health_endpoint_components_structure(self, client, setup_health):
        """Test /health components have required fields."""
        response = client.get("/health")
        data = response.json()

        components = data["components"]
        assert len(components) > 0

        # Check first component structure
        first_component = next(iter(components.values()))
        assert "status" in first_component
        assert "message" in first_component
        assert "error_count" in first_component

    def test_health_endpoint_with_errors(self, client):
        """Test /health when component has errors."""
        reset_health_checks()
        hc = health_checks()
        hc.mark_ready("AudioAgent")
        hc.mark_error("ASRAgent", "Model failed")

        response = client.get("/health")
        data = response.json()

        assert data["alive"] is False
        assert data["components"]["ASRAgent"]["status"] == "error"


class TestStatusEndpoint:
    """Test /status endpoint."""

    def test_status_endpoint_status(self, client, setup_health):
        """Test /status endpoint returns 200."""
        response = client.get("/status")
        assert response.status_code == 200

    def test_status_endpoint_content_type(self, client, setup_health):
        """Test /status returns JSON."""
        response = client.get("/status")
        assert "application/json" in response.headers.get("content-type", "")

    def test_status_endpoint_has_required_fields(self, client, setup_health):
        """Test /status response has required fields."""
        response = client.get("/status")
        data = response.json()

        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert "version" in data

    def test_status_endpoint_has_metrics_snapshot(self, client, setup_health):
        """Test /status includes metrics snapshot."""
        reset_metrics()
        m = metrics()

        # Set some metrics
        m.active_speakers.set(5)
        m.avg_latency_ms.set(250.0)
        m.audio_queue_depth_samples.set(10)

        response = client.get("/status")
        data = response.json()

        # Should have current metric values
        assert "active_speakers" in data or "metrics" in data


class TestReadinessEndpoint:
    """Test /ready endpoint (Kubernetes readiness probe)."""

    def test_ready_endpoint_200_when_ready(self, client, setup_health):
        """Test /ready returns 200 when system ready."""
        response = client.get("/ready")
        assert response.status_code == 200

    def test_ready_endpoint_503_when_not_ready(self, client):
        """Test /ready returns 503 when system not ready."""
        reset_health_checks()
        health_checks()
        # Leave all as UNKNOWN (not ready)

        response = client.get("/ready")
        assert response.status_code == 503

    def test_ready_endpoint_200_when_degraded(self, client):
        """Test /ready returns 200 when degraded (but not down)."""
        reset_health_checks()
        hc = health_checks()

        for agent in list(hc.components.keys()):
            hc.mark_ready(agent)

        hc.mark_degraded("AudioAgent", "High latency")

        response = client.get("/ready")
        # Degraded should still return 200 (system operational)
        assert response.status_code == 200

    def test_ready_endpoint_503_when_errors(self, client):
        """Test /ready returns 503 when component has errors."""
        reset_health_checks()
        hc = health_checks()

        hc.mark_ready("AudioAgent")
        hc.mark_error("ASRAgent", "Failed")

        response = client.get("/ready")
        assert response.status_code == 503


class TestLivenessEndpoint:
    """Test /live endpoint (Kubernetes liveness probe)."""

    def test_live_endpoint_200_when_alive(self, client, setup_health):
        """Test /live returns 200 when system alive."""
        response = client.get("/live")
        assert response.status_code == 200

    def test_live_endpoint_503_when_dead(self, client):
        """Test /live returns 503 when system has errors."""
        reset_health_checks()
        hc = health_checks()

        hc.mark_error("ASRAgent", "Critical failure")

        response = client.get("/live")
        assert response.status_code == 503

    def test_live_endpoint_200_when_degraded(self, client):
        """Test /live returns 200 when degraded (still alive)."""
        reset_health_checks()
        hc = health_checks()

        for agent in list(hc.components.keys()):
            hc.mark_ready(agent)

        hc.mark_degraded("AudioAgent", "High queue")

        response = client.get("/live")
        # Degraded = still alive
        assert response.status_code == 200


class TestEndpointIntegration:
    """Integration tests across endpoints."""

    def test_all_endpoints_accessible(self, client, setup_health):
        """Test all endpoints are accessible."""
        endpoints = ["/metrics", "/health", "/status", "/ready", "/live"]

        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code in [200, 503], f"{endpoint} failed"

    def test_health_status_consistency(self, client):
        """Test /health, /ready, /live are consistent."""
        reset_health_checks()
        hc = health_checks()

        # All ready
        for agent in list(hc.components.keys()):
            hc.mark_ready(agent)

        # All should indicate ready/alive
        assert client.get("/ready").status_code == 200
        assert client.get("/live").status_code == 200
        health_data = client.get("/health").json()
        assert health_data["ready"] is True
        assert health_data["alive"] is True

    def test_metrics_consistency_across_requests(self, client):
        """Test metrics remain consistent across requests."""
        reset_metrics()
        m = metrics()
        m.active_speakers.set(5)

        response1 = client.get("/status")
        response2 = client.get("/status")

        # Both requests should work
        assert response1.status_code == 200
        assert response2.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
