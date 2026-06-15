"""
Metrics HTTP server for SpeechShield.

Provides:
  • Prometheus metrics endpoint (/metrics)
  • Health check endpoint (/health)
  • System status endpoint (/status)
  • Runs on separate thread for non-blocking operation

Usage:
    from core.metrics_server import start_metrics_server
    
    # Start server (non-blocking)
    stop_server = start_metrics_server(port=8000)
    
    # Later, stop server
    stop_server()
"""

import asyncio
import logging
import threading
from typing import Callable, Optional
from fastapi import FastAPI, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import uvicorn

from core.metrics import metrics
from core.health import health_checks

logger = logging.getLogger("MetricsServer")


def create_metrics_app() -> FastAPI:
    """Create FastAPI application for metrics."""
    app = FastAPI(title="SpeechShield Metrics")

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
        hc = health_checks()
        status = hc.get_status()

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
        
        hc = health_checks()
        m = metrics()
        status = hc.get_status()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "2.0.0",
            "uptime_seconds": status["uptime_seconds"],
            "active_speakers": int(m.active_speakers._value.get()),
            "audio_queue_depth": int(m.audio_queue_depth_samples._value.get()),
            "asr_queue_depth": int(m.asr_queue_depth_samples._value.get()),
            "pipeline_latency_ms": float(m.avg_latency_ms._value.get()),
            "avg_latency_ms": float(m.avg_latency_ms._value.get()),
        }

    @app.get("/ready")
    async def get_ready() -> Response:
        """Kubernetes-style readiness probe."""
        hc = health_checks()
        if hc.is_ready:
            return Response(content="Ready", status_code=200)
        else:
            return Response(
                content="Not Ready",
                status_code=503,
            )

    @app.get("/live")
    async def get_live() -> Response:
        """Kubernetes-style liveness probe."""
        hc = health_checks()
        if hc.is_alive:
            return Response(content="Alive", status_code=200)
        else:
            return Response(
                content="Dead",
                status_code=503,
            )

    return app


def start_metrics_server(
    host: str = "127.0.0.1",
    port: int = 8000,
) -> Callable[[], None]:
    """
    Start metrics HTTP server in background thread.

    Args:
        host: Server host (default: localhost only)
        port: Server port (default: 8000)

    Returns:
        Callable to stop the server

    Example:
        stop = start_metrics_server(port=8000)
        # ... do work ...
        stop()  # Stop server
    """
    app = create_metrics_app()

    # Create event loop for server thread
    server_loop = asyncio.new_event_loop()

    def run_server():
        """Run server in background thread."""
        asyncio.set_event_loop(server_loop)
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)
        try:
            server_loop.run_until_complete(server.serve())
        except asyncio.CancelledError:
            pass

    # Start server thread
    server_thread = threading.Thread(target=run_server, daemon=True, name="MetricsServer")
    server_thread.start()

    logger.info(f"Metrics server started on http://{host}:{port}")
    logger.info(f"  - Prometheus metrics: http://{host}:{port}/metrics")
    logger.info(f"  - Health check: http://{host}:{port}/health")
    logger.info(f"  - Status: http://{host}:{port}/status")
    logger.info(f"  - Readiness probe: http://{host}:{port}/ready")
    logger.info(f"  - Liveness probe: http://{host}:{port}/live")

    def stop_server():
        """Stop the server."""
        server_loop.call_soon_threadsafe(server_loop.stop)
        server_thread.join(timeout=5.0)
        logger.info("Metrics server stopped")

    return stop_server


# Convenience function for typical use
async def run_metrics_server_async(
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """
    Run metrics server asynchronously (blocking).

    Use this if you want to await the server in your main async function.

    Args:
        host: Server host
        port: Server port
    """
    app = create_metrics_app()
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    await server.serve()
