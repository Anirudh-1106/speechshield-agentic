"""
SpeechShield — Multi-Agent Real-Time Voice Moderation System

Entry point: initialises the EventBus, GlobalState, all agents,
and runs them concurrently until Ctrl-C.
"""

import asyncio
import json
import logging
import os
import signal
import sys

# ---------------------------------------------------------------------------
# Load configuration early
# ---------------------------------------------------------------------------
from config.settings import settings

config = settings()

# ---------------------------------------------------------------------------
# Logging setup — file + console (configurable)
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.dirname(__file__), config.logging.log_dir)
os.makedirs(LOG_DIR, exist_ok=True)

handlers = [logging.StreamHandler(sys.stdout)]
if config.logging.enable_file_logging:
    handlers.append(
        logging.FileHandler(os.path.join(LOG_DIR, "speechshield.log"), encoding="utf-8")
    )

logging.basicConfig(
    level=getattr(logging, config.logging.level),
    format="%(asctime)s [%(name)-16s] %(levelname)-7s %(message)s",
    handlers=handlers,
)
logger = logging.getLogger("Main")

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from core.event_bus import EventBus
from core.state import GlobalState

from agents.audio_agent import AudioAgent
from agents.asr_agent import ASRAgent
from agents.toxicity_agent import ToxicityAgent
from agents.rewrite_agent import RewriteAgent
from agents.voice_agent import VoiceAgent
from agents.orchestrator import OrchestratorAgent


async def main() -> None:
    logger.info("=" * 70)
    logger.info("  SpeechShield Agentic v2 — starting up")
    logger.info("=" * 70)
    logger.info("Configuration: env=%s debug=%s", config.env, config.debug)

    # --- Core infrastructure ---
    bus = EventBus()
    state = GlobalState(settings=config)

    # --- Instantiate agents ---
    audio_agent = AudioAgent(bus, state)
    asr_agent = ASRAgent(bus, state)
    toxicity_agent = ToxicityAgent(bus, state)
    rewrite_agent = RewriteAgent(bus, state)
    voice_agent = VoiceAgent(bus, state, play_audio=config.voice.play_audio)
    orchestrator = OrchestratorAgent(bus, state)

    agents = [audio_agent, asr_agent, toxicity_agent, rewrite_agent, voice_agent, orchestrator]

    # --- Graceful shutdown ---
    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received.")
        shutdown_event.set()
        bus.stop()
        for agent in agents:
            agent.stop()

    # On Windows, loop.add_signal_handler is not supported.
    # Use a background thread to watch for KeyboardInterrupt instead.
    if sys.platform == "win32":
        # We rely on KeyboardInterrupt being raised in the main thread.
        pass
    else:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_handler)

    # --- Launch all agents + event bus concurrently ---
    async def _dashboard_writer() -> None:
        """Periodically write GlobalState snapshot to JSON for the Streamlit dashboard."""
        dash_path = os.path.join(LOG_DIR, "dashboard_state.json")
        refresh_interval = config.dashboard.refresh_interval
        while not shutdown_event.is_set():
            try:
                snapshot = state.get_dashboard_snapshot()
                with open(dash_path, "w", encoding="utf-8") as f:
                    json.dump(snapshot, f, ensure_ascii=False)
            except Exception as exc:
                logger.debug("Dashboard write error: %s", exc)
            await asyncio.sleep(refresh_interval)

    tasks = [
        asyncio.create_task(bus.start(), name="EventBus"),
        asyncio.create_task(audio_agent.start(), name="AudioAgent"),
        asyncio.create_task(asr_agent.start(), name="ASRAgent"),
        asyncio.create_task(toxicity_agent.start(), name="ToxicityAgent"),
        asyncio.create_task(rewrite_agent.start(), name="RewriteAgent"),
        asyncio.create_task(voice_agent.start(), name="VoiceAgent"),
        asyncio.create_task(orchestrator.start(), name="Orchestrator"),
        asyncio.create_task(_dashboard_writer(), name="DashboardWriter"),
    ]

    logger.info("All agents launched. Press Ctrl+C to stop.")

    try:
        await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Interrupted — shutting down …")
        _signal_handler()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("SpeechShield shut down cleanly.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bye.")
