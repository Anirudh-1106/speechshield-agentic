"""
Environment self-check: ``python main.py --doctor``.

Verifies every external dependency SpeechShield relies on and prints an
actionable hint for anything missing.  Exit code is non-zero if a check
required for the selected mode fails.
"""

from __future__ import annotations

import shutil
import socket
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from config.settings import Settings

OK, WARN, FAIL = "ok", "warn", "fail"
_ICON = {OK: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]"}


@dataclass
class Check:
    name: str
    status: str
    detail: str
    hint: str = ""


def _python() -> Check:
    v = sys.version_info
    ok = v >= (3, 11)
    return Check("Python", OK if ok else FAIL, f"{v.major}.{v.minor}.{v.micro}", "" if ok else "Python 3.11+ required")


def _ffmpeg() -> Check:
    path = shutil.which("ffmpeg")
    return Check("ffmpeg", OK if path else FAIL, path or "not found",
                 "" if path else "Needed to decode TTS audio. Install from https://ffmpeg.org and add it to PATH.")


def _audio(cfg: "Settings", mode: str) -> list[Check]:
    if mode == "api":
        return [Check("Audio devices", OK, "not needed in api mode")]
    try:
        from core.audio_io import list_devices
        devices = list_devices()
    except Exception as e:
        return [Check("PyAudio / PortAudio", FAIL, str(e), "pip install PyAudio (Linux: apt install portaudio19-dev)")]
    checks = [Check("PyAudio / PortAudio", OK, f"{len(devices)} devices")]
    if mode == "interceptor":
        name = cfg.interceptor.input_device_name.lower()
        found = [d for d in devices if name in d["name"].lower() and d["inputs"] > 0]
        checks.append(Check(
            f"Virtual cable input '{cfg.interceptor.input_device_name}'",
            OK if found else FAIL,
            found[0]["name"] if found else "not found",
            "" if found else "Install VB-Audio Virtual Cable (https://vb-audio.com/Cable/) — see VBCABLE_SETUP.md",
        ))
        if cfg.interceptor.output_device_name:
            out = cfg.interceptor.output_device_name.lower()
            found_out = [d for d in devices if out in d["name"].lower() and d["outputs"] > 0]
            checks.append(Check(
                f"Output device '{cfg.interceptor.output_device_name}'",
                OK if found_out else FAIL,
                found_out[0]["name"] if found_out else "not found",
                "" if found_out else "Run `python main.py --list-devices` and set SPEECHSHIELD_INTERCEPTOR_OUTPUT_DEVICE",
            ))
    else:
        has_in = any(d["inputs"] > 0 for d in devices)
        checks.append(Check("Microphone", OK if has_in else FAIL, "available" if has_in else "no input device",
                            "" if has_in else "Connect a microphone"))
    return checks


def _gpu(cfg: "Settings", mode: str) -> Check:
    if mode == "api":
        return Check("GPU (speech recognition)", OK, "not needed in api mode")
    from core.gpu import cuda_device_count
    if cuda_device_count() > 0:
        return Check("GPU (speech recognition)", OK, "CUDA usable — streaming censoring enabled")
    has_nvidia = shutil.which("nvidia-smi") is not None
    return Check("GPU (speech recognition)", WARN, "not usable — running on CPU (higher delay, no streaming)",
                 "pip install -r requirements-gpu.txt" if has_nvidia else "An NVIDIA GPU cuts delay ~3x")


def _voice_clone(cfg: "Settings", mode: str) -> Check:
    import os
    if not (mode == "interceptor" and cfg.interceptor.censor_mode == "voice_clone"):
        return Check("Voice clone", OK, "not used in this mode")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    py = cfg.voice_clone.python if os.path.isabs(cfg.voice_clone.python) else os.path.join(root, cfg.voice_clone.python)
    if not os.path.exists(py):
        return Check("Voice clone", FAIL, "worker venv missing",
                     r"powershell -ExecutionPolicy Bypass -File scripts\setup_voice_clone.ps1")
    return Check("Voice clone", OK, "worker venv installed (toxic words bleeped until a voice is learned)")


def _asr_model(cfg: "Settings") -> Check:
    import os
    from agents.asr_agent import resolve_model
    from core.gpu import cuda_device_count
    size = resolve_model(cfg.asr.model_size, "cuda" if cuda_device_count() > 0 else "cpu")
    cache = os.path.join(cfg.asr.model_dir, f"models--Systran--faster-whisper-{size}")
    if os.path.isdir(cache):
        return Check(f"Whisper model '{size}'", OK, "cached")
    return Check(f"Whisper model '{size}'", WARN, "not cached", "Will download on first start (~75 MB–1.5 GB depending on size)")


def _toxicity_model(cfg: "Settings") -> Check:
    name = cfg.toxicity.model_name
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache(name, "config.json")
        if isinstance(hit, str):
            return Check(f"Toxicity model '{name}'", OK, "cached")
    except Exception:
        pass
    return Check(f"Toxicity model '{name}'", WARN, "not cached", "Will download on first start (~440 MB)")


def _llm(cfg: "Settings") -> Check:
    r = cfg.rewrite
    if r.llm_provider == "local":
        return Check("LLM rewriter", OK, "disabled (rules only)")
    import httpx
    try:
        if r.llm_provider == "ollama":
            names = [m["name"] for m in httpx.get(f"{r.ollama_base_url}/api/tags", timeout=3).json().get("models", [])]
            if not names:
                return Check("Ollama", WARN, "running, no models", f"ollama pull {r.model}")
            if r.model in names or f"{r.model}:latest" in names:
                return Check("Ollama", OK, f"model '{r.model}' ready")
            return Check("Ollama", WARN, f"'{r.model}' missing; will use '{names[0]}'", f"ollama pull {r.model}")
        resp = httpx.get(f"{r.openai_base_url}/models", timeout=5,
                         headers={"Authorization": f"Bearer {r.openai_api_key}"} if r.openai_api_key else None)
        return Check("OpenAI-compatible LLM", OK if resp.status_code == 200 else WARN, f"HTTP {resp.status_code}")
    except Exception as e:
        return Check("LLM rewriter", WARN, f"unreachable ({type(e).__name__})",
                     "Optional. Install Ollama (https://ollama.com) — rule-based rewriting is used meanwhile.")


def _tts(cfg: "Settings", mode: str) -> Check:
    needed = mode == "standard" or (mode == "interceptor" and cfg.interceptor.censor_mode == "tts")
    if not needed:
        return Check("Edge TTS", OK, "not needed for this mode")
    try:
        socket.create_connection(("speech.platform.bing.com", 443), timeout=3).close()
        return Check("Edge TTS", OK, "reachable")
    except OSError:
        return Check("Edge TTS", WARN, "unreachable", "Edge TTS needs internet; toxic utterances will be muted instead")


def _diarization(cfg: "Settings") -> Check:
    import os
    if not cfg.diarization.enabled:
        return Check("Diarization", OK, "disabled")
    try:
        import pyannote.audio  # noqa: F401
    except ImportError:
        return Check("Diarization", FAIL, "pyannote.audio not installed", "pip install -r requirements-diarization.txt")
    token = cfg.diarization.hf_token or os.getenv("HF_TOKEN")
    return Check("Diarization", OK if token else WARN, "pyannote installed",
                 "" if token else "Set HF_TOKEN if the embedding model is gated")


def _port(name: str, host: str, port: int) -> Check:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        busy = s.connect_ex((host if host != "0.0.0.0" else "127.0.0.1", port)) == 0
    return Check(f"{name} port {port}", WARN if busy else OK, "in use" if busy else "free",
                 "Another process (or a running SpeechShield) is using it" if busy else "")


def run_doctor(cfg: "Settings", mode: str) -> int:
    probes: list[Callable[[], Check | list[Check]]] = [
        _python, _ffmpeg,
        lambda: _audio(cfg, mode),
        lambda: _gpu(cfg, mode),
        lambda: _voice_clone(cfg, mode),
        lambda: _asr_model(cfg),
        lambda: _toxicity_model(cfg),
        lambda: _llm(cfg),
        lambda: _tts(cfg, mode),
        lambda: _diarization(cfg),
        lambda: _port("HTTP server", cfg.server.host, cfg.server.port),
        lambda: _port("Dashboard", "127.0.0.1", cfg.dashboard.port),
    ]
    checks: list[Check] = []
    for probe in probes:
        try:
            out = probe()
        except Exception as e:  # a broken probe must not hide the others
            out = Check(getattr(probe, "__name__", "check"), WARN, f"check failed: {e}")
        checks.extend(out if isinstance(out, list) else [out])

    print(f"\nSpeechShield doctor — mode={mode} env={cfg.env}\n")
    width = max(len(c.name) for c in checks)
    for c in checks:
        print(f"  {_ICON[c.status]} {c.name.ljust(width)}  {c.detail}")
        if c.hint and c.status != OK:
            print(f"         {' ' * width}  -> {c.hint}")
    failed = [c for c in checks if c.status == FAIL]
    print(f"\n{'All required checks passed.' if not failed else f'{len(failed)} required check(s) failed.'}\n")
    return 1 if failed else 0
