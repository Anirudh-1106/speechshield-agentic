"""
HTTP server for SpeechShield.

Provides:
  • Prometheus metrics endpoint (/metrics)
  • Health check endpoints (/health, /status, /ready, /live)
  • Moderation API (POST /v1/moderate) and live state (GET /v1/state)
  • Runs on a separate thread for non-blocking operation

The /v1/* endpoints are protected by an optional bearer API key and a
per-client rate limit; the server binds to localhost unless configured
otherwise.

Usage:
    from core.metrics_server import start_metrics_server

    stop_server = start_metrics_server(port=9501)
    ...
    stop_server()
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import threading
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any, Callable, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from core import __version__
from core.health import health_checks
from core.metrics import metrics

if TYPE_CHECKING:
    from config.settings import ServerSettings
    from core.api import ModerationGateway

logger = logging.getLogger("MetricsServer")


class ModerateRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to moderate")
    speaker: str = Field(default="api", max_length=128, description="Stable ID of the author (enables per-user strictness)")


class _RateLimiter:
    """Sliding one-minute window per client."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            hits = self._hits[client]
            while hits and now - hits[0] > 60.0:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return False
            hits.append(now)
            if len(self._hits) > 10_000:  # bound memory under many distinct clients
                for k in [k for k, v in self._hits.items() if not v][:5000]:
                    del self._hits[k]
            return True


def create_metrics_app(
    gateway: Optional["ModerationGateway"] = None,
    server_settings: Optional["ServerSettings"] = None,
) -> FastAPI:
    """Create the FastAPI application."""
    if server_settings is None:
        from config.settings import ServerSettings
        server_settings = ServerSettings()
    cfg = server_settings
    limiter = _RateLimiter(cfg.rate_limit_per_minute)

    app = FastAPI(title="SpeechShield", version=__version__)

    def guard(request: Request) -> None:
        """API key + rate limit for /v1/* endpoints."""
        if cfg.api_key:
            supplied = request.headers.get("authorization", "")
            expected = f"Bearer {cfg.api_key}"
            if not hmac.compare_digest(supplied.encode(), expected.encode()):
                metrics().api_requests_total.labels(endpoint=request.url.path, status="401").inc()
                raise HTTPException(status_code=401, detail="Invalid or missing API key")
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client):
            metrics().api_requests_total.labels(endpoint=request.url.path, status="429").inc()
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

    @app.get("/metrics")
    async def get_metrics() -> Response:
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    @app.get("/health")
    async def get_health() -> dict:
        """
        Health check endpoint.

        Returns health status with full component details.
        """
        status = health_checks().get_status()
        return {
            "ready": status["ready"],
            "alive": status["alive"],
            "uptime_seconds": status["uptime_seconds"],
            "components": status["components"],  # Full component data
        }

    @app.get("/status")
    async def get_status() -> dict:
        """Detailed system status endpoint."""
        from datetime import datetime, timezone

        m = metrics()
        status = health_checks().get_status()
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": __version__,
            "uptime_seconds": status["uptime_seconds"],
            "active_speakers": int(m.active_speakers._value.get()),
            "audio_queue_depth": int(m.audio_queue_depth_samples._value.get()),
            "asr_queue_depth": int(m.asr_queue_depth_samples._value.get()),
            "pipeline_latency_ms": float(m.avg_latency_ms._value.get()),
            "avg_latency_ms": float(m.avg_latency_ms._value.get()),
            "llm_circuit_open": bool(m.llm_circuit_open._value.get()),
            "moderation_api": gateway is not None,
        }

    @app.get("/ready")
    async def get_ready() -> Response:
        """Kubernetes-style readiness probe."""
        if health_checks().is_ready:
            return Response(content="Ready", status_code=200)
        return Response(content="Not Ready", status_code=503)

    @app.get("/live")
    async def get_live() -> Response:
        """Kubernetes-style liveness probe."""
        if health_checks().is_alive:
            return Response(content="Alive", status_code=200)
        return Response(content="Dead", status_code=503)

    @app.post("/v1/moderate", dependencies=[Depends(guard)])
    async def moderate(req: ModerateRequest) -> dict[str, Any]:
        """Classify text and return a masked and a polite rewritten version."""
        if gateway is None:
            raise HTTPException(status_code=503, detail="Pipeline not attached")
        if len(req.text) > cfg.max_text_length:
            metrics().api_requests_total.labels(endpoint="/v1/moderate", status="413").inc()
            raise HTTPException(status_code=413, detail=f"Text longer than {cfg.max_text_length} characters")
        try:
            result = await gateway.moderate(req.text, req.speaker)
        except asyncio.TimeoutError:
            metrics().api_requests_total.labels(endpoint="/v1/moderate", status="504").inc()
            raise HTTPException(status_code=504, detail="Moderation timed out") from None
        metrics().api_requests_total.labels(endpoint="/v1/moderate", status="200").inc()
        return result

    @app.get("/v1/state", dependencies=[Depends(guard)])
    async def state() -> dict[str, Any]:
        """Live pipeline snapshot (feeds the dashboard)."""
        if gateway is None:
            raise HTTPException(status_code=503, detail="Pipeline not attached")
        snap = await asyncio.get_running_loop().run_in_executor(None, gateway.snapshot)
        snap["health"] = health_checks().get_status()
        return snap

    return app


def start_metrics_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    gateway: Optional["ModerationGateway"] = None,
    server_settings: Optional["ServerSettings"] = None,
) -> Callable[[], None]:
    """
    Start the HTTP server in a background thread.

    Returns:
        Callable to stop the server
    """
    app = create_metrics_app(gateway, server_settings)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)

    def run_server() -> None:
        try:
            asyncio.run(server.serve())
        except Exception as exc:  # e.g. port already in use
            logger.error("HTTP server failed: %s", exc)

    server_thread = threading.Thread(target=run_server, daemon=True, name="MetricsServer")
    server_thread.start()

    logger.info("HTTP server on http://%s:%d  (/metrics /health /status /ready /live /v1/moderate /v1/state)", host, port)

    def stop_server() -> None:
        server.should_exit = True
        server_thread.join(timeout=5.0)
        logger.info("HTTP server stopped")

    return stop_server


# Convenience function for typical use
async def run_metrics_server_async(
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Run the server on the current event loop (blocking)."""
    config = uvicorn.Config(create_metrics_app(), host=host, port=port, log_level="warning", access_log=False)
    await uvicorn.Server(config).serve()
