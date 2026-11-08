"""
SpeechShield — Multi-Agent Real-Time Voice Moderation System

Modes:
  standard     Listen on the microphone, speak back a cleaned version (demo)
  interceptor  Sit inline in a call (via VB-Cable) and censor toxic speech
               before it reaches your headphones
  api          Headless: text moderation HTTP API only (Docker / bots)

Examples:
  python main.py                                # standard mode
  python main.py --mode interceptor             # censor call audio
  python main.py --mode interceptor --censor-mode voice_clone
  python main.py --mode api                     # POST /v1/moderate
  python main.py --text "you are a stupid idiot"
  python main.py --doctor                       # check your setup
  python main.py --list-devices

Features:
  • Structured logging with correlation IDs
  • Prometheus metrics, health probes, moderation API
  • Supervised agents with restart-on-crash
  • Full async/await architecture
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any, Optional

ROOT = os.path.dirname(os.path.abspath(__file__))

if TYPE_CHECKING:
    from config.settings import Settings

MODES = ("standard", "interceptor", "api")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="speechshield",
        description="Real-time AI voice moderation — filters toxic speech before you hear it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples:")[1].split("Features:")[0] if __doc__ else None,
    )
    p.add_argument("--mode", choices=MODES, help="Run mode (default: interceptor if enabled in config, else standard)")
    p.add_argument("--env", choices=("dev", "test", "prod"), help="Config profile (default: $SPEECHSHIELD_ENV or dev)")
    p.add_argument("--censor-mode", choices=("bleep", "mute", "voice_clone", "tts"), help="Interceptor: how toxic speech is censored")
    p.add_argument("--input-device", help="Interceptor: capture device name (default 'CABLE Output')")
    p.add_argument("--output-device", help="Interceptor: playback device name (default: system default)")
    p.add_argument("--no-dashboard", action="store_true", help="Do not launch the Streamlit dashboard")
    p.add_argument("--port", type=int, help="HTTP server port (metrics, health, API)")
    p.add_argument("--text", help="Moderate one piece of text, print JSON and exit")
    p.add_argument("--doctor", action="store_true", help="Check dependencies and devices, then exit")
    p.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    return p.parse_args(argv)


def apply_cli_overrides(args: argparse.Namespace) -> None:
    """CLI flags win over YAML and env: express them as env overrides."""
    env = os.environ
    if args.env:
        env["SPEECHSHIELD_ENV"] = args.env
    if args.mode == "interceptor":
        env["SPEECHSHIELD_INTERCEPTOR_ENABLED"] = "true"
    elif args.mode in ("standard", "api"):
        env["SPEECHSHIELD_INTERCEPTOR_ENABLED"] = "false"
    if args.mode == "api" or args.text:
        env["SPEECHSHIELD_CAPTURE_ENABLED"] = "false"
    if args.censor_mode:
        env["SPEECHSHIELD_CENSOR_MODE"] = args.censor_mode
    if args.input_device:
        env["SPEECHSHIELD_INTERCEPTOR_INPUT_DEVICE"] = args.input_device
    if args.output_device:
        env["SPEECHSHIELD_INTERCEPTOR_OUTPUT_DEVICE"] = args.output_device
    if args.no_dashboard or args.text:
        env["SPEECHSHIELD_DASHBOARD_ENABLED"] = "false"
    if args.port:
        env["SPEECHSHIELD_SERVER_PORT"] = str(args.port)
    if args.text:
        env["SPEECHSHIELD_SERVER_ENABLED"] = "false"
        env.setdefault("SPEECHSHIELD_LOG_LEVEL", "WARNING")


def resolve_mode(cfg: "Settings", args: argparse.Namespace) -> str:
    if args.mode:
        return args.mode
    if not cfg.audio.capture_enabled:
        return "api"
    return "interceptor" if cfg.interceptor.enabled else "standard"


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------

def build_agents(cfg: "Settings", mode: str, bus: Any, state: Any) -> list[tuple[str, Any, bool]]:
    """(name, agent, critical) for the selected mode. Audio imports stay lazy."""
    from agents.orchestrator import OrchestratorAgent
    from agents.rewrite_agent import RewriteAgent
    from agents.toxicity_agent import ToxicityAgent

    agents: list[tuple[str, Any, bool]] = []
    if mode in ("standard", "interceptor"):
        from agents.asr_agent import ASRAgent
        from agents.voice_agent import VoiceAgent

        if mode == "interceptor":
            from agents.audio_interceptor import AudioInterceptor
            agents.append(("AudioInterceptor", AudioInterceptor(bus, state), True))
            if cfg.interceptor.censor_mode == "voice_clone":
                from agents.voice_clone_agent import VoiceCloneAgent
                agents.append(("VoiceCloneAgent", VoiceCloneAgent(bus, state), False))
        else:
            from agents.audio_agent import AudioAgent
            agents.append(("AudioAgent", AudioAgent(bus, state), True))
        agents.append(("ASRAgent", ASRAgent(bus, state), True))
        if cfg.diarization.enabled:
            from agents.diarization_agent import DiarizationAgent
            agents.append(("DiarizationAgent", DiarizationAgent(bus, state), False))
        agents.append(("VoiceAgent", VoiceAgent(bus, state, play_audio=cfg.voice.play_audio), False))

    toxicity = ToxicityAgent(bus, state)
    agents.append(("ToxicityAgent", toxicity, True))
    agents.append(("RewriteAgent", RewriteAgent(bus, state, verifier=toxicity.score), False))
    agents.append(("OrchestratorAgent", OrchestratorAgent(bus, state), False))
    return agents


def _preload_native_libs(mode: str) -> None:
    from core.gpu import register_cuda_libraries

    register_cuda_libraries()
    import torch  # noqa: F401
    import transformers  # noqa: F401
    if mode != "api":
        import faster_whisper  # noqa: F401


def _write_json_atomic(path: str, payload: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    try:
        os.replace(tmp, path)
    except PermissionError:
        pass  # reader has the file open (Windows) — try again next tick


async def run(cfg: "Settings", mode: str) -> int:
    from core.api import ModerationGateway
    from core.event_bus import EventBus
    from core.health import health_checks
    from core.logging import get_logger
    from core.state import GlobalState
    from core.supervisor import Supervisor

    logger = get_logger("Main")
    logger.info("speechshield_starting", mode=mode, env=cfg.env)

    hc = health_checks()
    hc.components.clear()  # only report components this mode actually runs
    hc.mark_starting("EventBus")

    bus = EventBus(subscriber_queue_size=cfg.pipeline.subscriber_queue_size)
    state = GlobalState(settings=cfg)
    state.system_info.update({"mode": mode, "env": cfg.env})
    if mode == "interceptor":
        state.system_info["censor_mode"] = cfg.interceptor.censor_mode
        logger.info(
            "interceptor_mode",
            input=cfg.interceptor.input_device_name,
            output=cfg.interceptor.output_device_name or "system default",
            censor=cfg.interceptor.censor_mode,
            fail=cfg.interceptor.fail_mode,
            delay_s=cfg.interceptor.delay_seconds,
        )

    # Import native ML stacks on the main thread before any worker or server
    # thread starts: concurrent first-imports of torch/CTranslate2 DLLs from
    # several threads can crash the interpreter on Windows. Blocking here is
    # deliberate — nothing else is running yet.
    _preload_native_libs(mode)

    supervisor = Supervisor(bus, max_restarts=cfg.pipeline.max_agent_restarts, backoff_seconds=cfg.pipeline.restart_backoff_seconds)
    for name, agent, critical in build_agents(cfg, mode, bus, state):
        supervisor.add(name, agent, critical=critical)

    bus_task = asyncio.create_task(bus.start(), name="EventBus")
    hc.mark_ready("EventBus")
    supervisor.start_all()

    # --- HTTP server: metrics, health, moderation API ---
    stop_server = None
    if cfg.server.enabled:
        from core.metrics_server import start_metrics_server
        gateway = ModerationGateway(bus, state, asyncio.get_running_loop(), timeout=cfg.rewrite.timeout_seconds + 10)
        stop_server = start_metrics_server(cfg.server.host, cfg.server.port, gateway=gateway, server_settings=cfg.server)

    # --- Dashboard (separate Streamlit process) ---
    dashboard_process = None
    if cfg.dashboard.enabled:
        dashboard_process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", os.path.join(ROOT, "dashboard.py"),
             "--server.port", str(cfg.dashboard.port), "--server.headless", "true",
             "--browser.gatherUsageStats", "false"],
            cwd=ROOT,
        )

    log_dir = os.path.join(ROOT, cfg.logging.log_dir)
    os.makedirs(log_dir, exist_ok=True)

    async def _dashboard_writer() -> None:
        """File fallback for the dashboard when the HTTP server is disabled."""
        dash_path = os.path.join(log_dir, "dashboard_state.json")
        while True:
            try:
                snap = state.get_dashboard_snapshot()
                snap["health"] = hc.get_status()
                _write_json_atomic(dash_path, snap)
            except Exception as exc:
                logger.debug("dashboard_write_error", error=str(exc))
            await asyncio.sleep(cfg.dashboard.refresh_interval)

    async def _announce_ready() -> None:
        while not hc.is_ready:  # noqa: ASYNC110 — health is updated from many places
            await asyncio.sleep(0.5)
        urls = []
        if cfg.server.enabled:
            urls.append(f"api=http://{cfg.server.host}:{cfg.server.port}")
        if cfg.dashboard.enabled:
            urls.append(f"dashboard=http://localhost:{cfg.dashboard.port}")
        logger.info("speechshield_ready", mode=mode, startup_s=round(time.time() - state.system_info["started_at"], 1), **dict(u.split("=", 1) for u in urls))
        print(f"\n  SpeechShield is ready ({mode} mode). Press Ctrl+C to stop.\n", flush=True)

    helpers = [asyncio.create_task(_dashboard_writer()), asyncio.create_task(_announce_ready())]

    exit_code = 0
    try:
        # Runs until Ctrl+C (cancellation) or a critical agent gives up.
        await supervisor.fatal.wait()
        logger.error("speechshield_fatal", reason=supervisor.fatal_reason)
        print(f"\n  SpeechShield stopped: {supervisor.fatal_reason}\n  Run `python main.py --doctor` for hints.\n", file=sys.stderr)
        exit_code = 1
    except asyncio.CancelledError:
        logger.info("shutdown_requested")
    finally:
        for h in helpers:
            h.cancel()
        await supervisor.stop_all(timeout=cfg.pipeline.graceful_shutdown_timeout)
        bus.stop()
        await asyncio.gather(bus_task, *helpers, return_exceptions=True)
        if stop_server:
            stop_server()
        if dashboard_process is not None:
            dashboard_process.terminate()
            try:
                dashboard_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                dashboard_process.kill()
        logger.info("speechshield_stopped")
    return exit_code


async def moderate_once(cfg: "Settings", text: str) -> dict:
    """Spin up the text pipeline, moderate *text*, shut down."""
    from agents.rewrite_agent import RewriteAgent
    from agents.toxicity_agent import ToxicityAgent
    from core.api import ModerationGateway
    from core.event_bus import EventBus
    from core.state import GlobalState

    bus = EventBus()
    state = GlobalState(settings=cfg)
    tox = ToxicityAgent(bus, state)
    rw = RewriteAgent(bus, state, verifier=tox.score)
    tasks = [asyncio.create_task(c) for c in (bus.start(), tox.start(), rw.start())]
    try:
        while not tox._ready.is_set():  # model loaded
            if tasks[1].done():
                tasks[1].result()  # surface the startup error
            await asyncio.sleep(0.05)
        await rw._llm.is_available()
        gateway = ModerationGateway(bus, state, asyncio.get_running_loop(), timeout=cfg.rewrite.timeout_seconds + 10)
        return await gateway._moderate(text, "cli")
    finally:
        tox.stop()
        rw.stop()
        bus.stop()
        await asyncio.wait(tasks, timeout=3)
        for t in tasks:
            t.cancel()


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    apply_cli_overrides(args)

    if args.list_devices:
        from core.audio_io import list_devices
        for d in list_devices():
            kind = "/".join(k for k, n in (("in", d["inputs"]), ("out", d["outputs"])) if n)
            print(f"[{d['index']:>2}] {d['name']}  ({kind}, {d['host_api']})")
        return 0

    from config.settings import load_config
    from core.logging import configure_logging

    cfg = load_config()
    configure_logging(
        level=cfg.logging.level,
        format=cfg.logging.format,
        log_dir=os.path.join(ROOT, cfg.logging.log_dir),
        enable_file_logging=cfg.logging.enable_file_logging,
    )
    mode = resolve_mode(cfg, args)

    if args.doctor:
        from core.doctor import run_doctor
        return run_doctor(cfg, mode)

    try:
        if args.text:
            print(json.dumps(asyncio.run(moderate_once(cfg, args.text)), indent=2, ensure_ascii=False))
            return 0
        return asyncio.run(run(cfg, mode))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
