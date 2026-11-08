"""
VoiceCloneAgent — replaces toxic words with polite ones in the speaker's own voice.

Used by ``interceptor.censor_mode = voice_clone``.  The heavy lifting runs
in a separate worker process (``voice_service/worker.py``, its own venv,
Chatterbox Turbo on the GPU); this agent:

  1. launches and supervises the worker,
  2. *enrolls* speakers from their clean utterances — as soon as ~6 s of
     clean speech is collected their voice can be cloned, and the whole
     replacement vocabulary is pre-synthesised in the background so later
     replacements are instant cache hits,
  3. answers ``clone_request`` events from the interceptor with
     ``clone_ready`` — PCM to splice in, ``b""`` to cut the word out, or
     ``None`` to fall back to a bleep.

Any failure (worker down, speaker not enrolled yet, timeout) degrades to
bleeping that word; the censor itself never waits on this agent for long.

Consent & safety: reference audio lives only in worker memory; the clone
can only say words from a fixed polite vocabulary (never arbitrary text);
Chatterbox watermarks everything it generates.
"""

from __future__ import annotations

import asyncio
import base64
import os
import subprocess
import sys
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Optional

import httpx

from core.event_bus import Event
from core.health import health_checks
from core.lexicon import SOFT_VOCABULARY, soft_replacement
from core.logging import get_logger
from core.metrics import metrics

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.state import GlobalState

logger = get_logger("agents.voice_clone_agent")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_ENROLL_SECONDS = 1.0  # shorter clips add little voice information


class VoiceCloneAgent:
    def __init__(self, event_bus: "EventBus", state: "GlobalState") -> None:
        self.bus = event_bus
        self.state = state
        self.cfg = state.settings.voice_clone
        self.base_url = f"http://{self.cfg.host}:{self.cfg.port}"
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.cfg.request_timeout_seconds)
        self._process: Optional[subprocess.Popen] = None
        self._audio: "OrderedDict[str, bytes]" = OrderedDict()  # recent utterance audio by correlation id
        self._ready_speakers: set[str] = set()
        self._worker_ready = False
        self._running = False
        self._stopped = asyncio.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._stopped.clear()
        self._running = True
        self.bus.subscribe("speech_segment", self._on_segment)
        self.bus.subscribe("toxicity_result", self._on_toxicity)
        self.bus.subscribe("clone_request", self._on_clone_request)
        self.state.system_info["voice_clone"] = "starting worker"
        health_checks().mark_degraded("VoiceCloneAgent", "worker starting — bleeping until ready")

        if self.cfg.autostart and not await self._ping():
            self._launch_worker()
        watcher = asyncio.create_task(self._wait_for_worker())
        try:
            await self._stopped.wait()
        finally:
            watcher.cancel()
            await self._client.aclose()
            self._terminate_worker()

    def stop(self) -> None:
        self._running = False
        self._stopped.set()

    def _launch_worker(self) -> None:
        python = self.cfg.python if os.path.isabs(self.cfg.python) else os.path.join(ROOT, self.cfg.python)
        if not os.path.exists(python):
            msg = f"voice venv not found at {python} — run scripts/setup_voice_clone.ps1"
            logger.error("voice_worker_missing", python=python)
            self.state.system_info["voice_clone"] = "unavailable (not installed)"
            health_checks().mark_degraded("VoiceCloneAgent", msg)
            return
        log_dir = os.path.join(ROOT, self.state.settings.logging.log_dir)
        os.makedirs(log_dir, exist_ok=True)
        log = open(os.path.join(log_dir, "voice_worker.log"), "ab")  # noqa: SIM115 — owned by the child
        cmd = [python, os.path.join(ROOT, "voice_service", "worker.py"),
               "--host", self.cfg.host, "--port", str(self.cfg.port), "--device", self.cfg.device,
               "--model", self.cfg.model, "--min-ref", str(self.cfg.min_reference_seconds),
               "--max-ref", str(self.cfg.max_reference_seconds)]
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0  # type: ignore[attr-defined]
        self._process = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        logger.info("voice_worker_launched", pid=self._process.pid, port=self.cfg.port, model=self.cfg.model)

    def _terminate_worker(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None

    async def _ping(self) -> Optional[dict]:
        try:
            r = await self._client.get("/health", timeout=2.0)
            return r.json() if r.status_code == 200 else None
        except httpx.HTTPError:
            return None

    async def _wait_for_worker(self) -> None:
        """Poll the worker until its model is loaded (first start downloads ~4 GB)."""
        t0 = time.time()
        while self._running:
            info = await self._ping()
            if info and info.get("ready"):
                self._worker_ready = True
                self.state.system_info["voice_clone"] = f"chatterbox-{self.cfg.model} ({info.get('device')})"
                health_checks().mark_ready("VoiceCloneAgent")
                logger.info("voice_worker_ready", seconds=round(time.time() - t0, 1), vram_gb=info.get("vram_gb"))
                return
            if info and info.get("error"):
                self.state.system_info["voice_clone"] = "unavailable (model failed)"
                health_checks().mark_degraded("VoiceCloneAgent", f"worker error: {info['error']}")
                logger.error("voice_worker_failed", error=info["error"])
                return
            if self._process is not None and self._process.poll() is not None:
                health_checks().mark_degraded("VoiceCloneAgent", "worker exited — see logs/voice_worker.log")
                self.state.system_info["voice_clone"] = "unavailable (worker exited)"
                return
            await asyncio.sleep(2.0)

    # ------------------------------------------------------------------
    # Enrollment: learn each speaker's voice from their clean speech
    # ------------------------------------------------------------------

    async def _on_segment(self, event: Event) -> None:
        audio = event.data.get("audio", b"")
        if len(audio) / 2 / event.data.get("sample_rate", 16000) >= MIN_ENROLL_SECONDS:
            self._audio[event.correlation_id] = audio
            while len(self._audio) > 32:
                self._audio.popitem(last=False)

    async def _on_toxicity(self, event: Event) -> None:
        audio = self._audio.pop(event.correlation_id, None)
        if audio is None or event.data.get("toxic") or not self._worker_ready:
            return  # only clean speech becomes a voice reference
        speaker = event.data.get("speaker", "default")
        try:
            r = await self._client.post("/enroll", json={"speaker": speaker, "pcm_b64": base64.b64encode(audio).decode()},
                                        timeout=30.0)
            info = r.json()
        except httpx.HTTPError as e:
            logger.warning("voice_enroll_failed", speaker=speaker, error=str(e))
            return
        if info.get("ready") and speaker not in self._ready_speakers:
            self._ready_speakers.add(speaker)
            logger.info("voice_enrolled", speaker=speaker, seconds=info.get("seconds"))
            try:  # build the speaker's replacement vocabulary in the background
                await self._client.post("/prewarm", json={"speaker": speaker, "texts": SOFT_VOCABULARY})
            except httpx.HTTPError:
                pass
        self.state.system_info["voice_clone_speakers"] = sorted(self._ready_speakers)

    # ------------------------------------------------------------------
    # Replacement requests
    # ------------------------------------------------------------------

    async def _on_clone_request(self, event: Event) -> None:
        key = event.data["key"]
        replacement = soft_replacement(event.data.get("text", ""))
        pcm: Optional[bytes]
        if replacement is None:
            pcm = None                      # no safe stand-in: bleep
        elif replacement == "":
            pcm = b""                       # intensifier/slur: cut it out
        else:
            pcm = await self._synthesize(event.data.get("speaker", "default"), replacement)
        logger.info("voice_clone_reply", correlation_id=event.correlation_id, text=event.data.get("text"),
                    replacement=replacement, result="splice" if pcm else ("cut" if pcm == b"" else "bleep"))
        await self.bus.emit(Event("clone_ready", {"key": key, "pcm": pcm, "replacement": replacement},
                                  correlation_id=event.correlation_id))

    async def _synthesize(self, speaker: str, text: str) -> Optional[bytes]:
        if not self._worker_ready or speaker not in self._ready_speakers:
            return None
        t0 = time.time()
        try:
            r = await self._client.post("/synthesize", json={"speaker": speaker, "text": text})
            if r.status_code != 200:
                return None
            body: dict[str, Any] = r.json()
        except httpx.HTTPError as e:
            logger.warning("voice_clone_failed", error=type(e).__name__)
            return None
        metrics().voice_clone_latency_seconds.labels(cache="hit" if body.get("cached") else "miss").observe(time.time() - t0)
        return base64.b64decode(body["pcm_b64"])
