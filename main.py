"""
SpeechShield — Multi-Agent Real-Time Voice Moderation System

Entry point: initialises the EventBus, GlobalState, all agents,
and runs them concurrently until Ctrl-C.

Features:
  • Structured logging with correlation IDs
  • Prometheus metrics collection
  • Health checks and monitoring
  • Full async/await architecture
"""

import asyncio
import json
import logging
import os
import signal
import sys
import subprocess

# ---------------------------------------------------------------------------
# Load configuration early
# ---------------------------------------------------------------------------
from config.settings import settings

config = settings()

# ---------------------------------------------------------------------------
# Structured logging setup (using structlog + config)
# ---------------------------------------------------------------------------
from core.logging import configure_logging, get_logger

configure_logging(
    level=config.logging.level,
    format=config.logging.format,
    log_dir=config.logging.log_dir,
    enable_file_logging=config.logging.enable_file_logging,
)
logger = get_logger("Main")

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from core.event_bus import EventBus
from core.state import GlobalState
from core.health import health_checks
from core.metrics_server import start_metrics_server

from agents.audio_agent import AudioAgent
from agents.asr_agent import ASRAgent
from agents.toxicity_agent import ToxicityAgent
from agents.rewrite_agent import RewriteAgent
from agents.voice_agent import VoiceAgent
from agents.diarization_agent import DiarizationAgent
from agents.orchestrator import OrchestratorAgent


async def main() -> None:
    logger.info("=" * 70)
    logger.info("  SpeechShield Agentic v2 — starting up")
    logger.info("=" * 70)
    logger.info("Configuration: env=%s debug=%s", config.env, config.debug)

    # --- Initialize health checks ---
    hc = health_checks()
    hc.mark_starting("EventBus")
    hc.mark_starting("AudioAgent")
    hc.mark_starting("ASRAgent")
    hc.mark_starting("DiarizationAgent")
    hc.mark_starting("ToxicityAgent")
    hc.mark_starting("RewriteAgent")
    hc.mark_starting("VoiceAgent")
    hc.mark_starting("OrchestratorAgent")

    # --- Start metrics server (non-blocking, background thread) ---
    metrics_port = config.dashboard.port + 1000  # e.g., 9501 if dashboard is 8501
    stop_metrics_server = start_metrics_server(host="127.0.0.1", port=metrics_port)
    logger.info("Metrics server started on port %d", metrics_port)

    # --- Core infrastructure ---
    bus = EventBus()
    state = GlobalState(settings=config)

    # --- Instantiate agents ---
    audio_agent = AudioAgent(bus, state)
    asr_agent = ASRAgent(bus, state)
    toxicity_agent = ToxicityAgent(bus, state)
    diarization_agent = DiarizationAgent(bus, state)
    rewrite_agent = RewriteAgent(bus, state)
    voice_agent = VoiceAgent(bus, state, play_audio=config.voice.play_audio)
    orchestrator = OrchestratorAgent(bus, state)

    agents = [audio_agent, asr_agent, diarization_agent, toxicity_agent, rewrite_agent, voice_agent, orchestrator]

    # --- Graceful shutdown ---
    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received.")
        shutdown_event.set()
        bus.stop()
        for agent in agents:
            agent.stop()
        hc.mark_degraded("EventBus", "shutting down")
        stop_metrics_server()

    # On Windows, loop.add_signal_handler is not supported.
    # Use a background thread to watch for KeyboardInterrupt instead.
    if sys.platform == "win32":
        # We rely on KeyboardInterrupt being raised in the main thread.
        pass
    else:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_handler)

    # --- Setup dashboard writer ---
    log_dir = os.path.join(os.path.dirname(__file__), config.logging.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    dashboard_process = None
    if config.dashboard.enabled:
        logger.info(f"Starting Streamlit dashboard on port {config.dashboard.port}...")
        dashboard_process = subprocess.Popen(
            [
                sys.executable, "-m", "streamlit", "run", "dashboard.py",
                "--server.port", str(config.dashboard.port),
                "--server.headless", "true"
            ]
        )

    async def _dashboard_writer() -> None:
        """Periodically write GlobalState snapshot to JSON for the Streamlit dashboard."""
        dash_path = os.path.join(log_dir, "dashboard_state.json")
        refresh_interval = config.dashboard.refresh_interval
        while not shutdown_event.is_set():
            try:
                snapshot = state.get_dashboard_snapshot()
                with open(dash_path, "w", encoding="utf-8") as f:
                    json.dump(snapshot, f, ensure_ascii=False)
            except Exception as exc:
                logger.debug("Dashboard write error", error=exc)
            await asyncio.sleep(refresh_interval)

    # --- Launch all agents + event bus concurrently ---
    tasks = [
        asyncio.create_task(bus.start(), name="EventBus"),
        asyncio.create_task(audio_agent.start(), name="AudioAgent"),
        asyncio.create_task(asr_agent.start(), name="ASRAgent"),
        asyncio.create_task(diarization_agent.start(), name="DiarizationAgent"),
        asyncio.create_task(toxicity_agent.start(), name="ToxicityAgent"),
        asyncio.create_task(rewrite_agent.start(), name="RewriteAgent"),
        asyncio.create_task(voice_agent.start(), name="VoiceAgent"),
        asyncio.create_task(orchestrator.start(), name="Orchestrator"),
        asyncio.create_task(_dashboard_writer(), name="DashboardWriter"),
    ]

    logger.info("All agents launched", metric="startup_complete")

    # Mark all agents as ready
    for agent_name in ["AudioAgent", "ASRAgent", "DiarizationAgent", "ToxicityAgent", "RewriteAgent", "VoiceAgent", "OrchestratorAgent"]:
        hc.mark_ready(agent_name)
    hc.mark_ready("EventBus")

    logger.info("System ready. Press Ctrl+C to stop.")

    try:
        await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Interrupted — shutting down …")
        _signal_handler()
    finally:
        if dashboard_process is not None:
            logger.info("Terminating Streamlit dashboard...")
            dashboard_process.terminate()
            try:
                dashboard_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                dashboard_process.kill()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("SpeechShield shut down cleanly.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bye.")
