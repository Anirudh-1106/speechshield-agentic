"""
SpeechShield voice-clone worker.

Runs in its own virtual environment (``.venv-voice``) because Chatterbox
pins a different PyTorch than the main pipeline.  Isolation also means a
crash or GPU out-of-memory here never takes the censor down: the main
process simply falls back to bleeping.

Endpoints (localhost only):
  GET  /health      — model state, enrolled speakers
  POST /enroll      — add clean reference speech for a speaker
  POST /synthesize  — speak text in a speaker's cloned voice (16 kHz PCM)

Model: Chatterbox Turbo (MIT licence, Resemble AI).  Every clip carries
Resemble's inaudible Perth watermark, so synthetic audio stays detectable.

Run (normally started automatically by main.py):
    .venv-voice/Scripts/python.exe voice_service/worker.py --port 9502
"""

from __future__ import annotations

import argparse
import base64
import gc
import logging
import os
import sys
import tempfile
import threading
import time
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [voice-worker] %(message)s")
log = logging.getLogger("voice-worker")

SR_OUT = 16000


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_ST_DTYPES = {
    "F32": "float32", "F16": "float16", "BF16": "bfloat16", "F64": "float64",
    "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8", "U8": "uint8", "BOOL": "bool",
}


def stream_safetensors(path: Path, device: str) -> dict:
    """Read a .safetensors file tensor-by-tensor with plain reads.

    safetensors memory-maps the whole file; on Windows machines with little
    free commit memory, paging a multi-GB mapping in fails with an access
    violation.  Sequential reads keep peak host RAM at one tensor.
    """
    import json
    import struct

    import torch

    out = {}
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        base = 8 + header_len
        items = sorted(((k, v) for k, v in header.items() if k != "__metadata__"), key=lambda kv: kv[1]["data_offsets"][0])
        for name, meta in items:
            start, end = meta["data_offsets"]
            f.seek(base + start)
            buf = bytearray(f.read(end - start))
            dtype = getattr(torch, _ST_DTYPES[meta["dtype"]])
            t = torch.frombuffer(buf, dtype=dtype) if buf else torch.empty(0, dtype=dtype)
            out[name] = t.reshape(meta["shape"]).to(device)
            del buf
    return out


def _to_device(obj: Any, device: Any) -> Any:
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(o, device) for o in obj)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


def _cpu_offload_class() -> Any:
    import torch

    class _CPUOffload(torch.nn.Module):
        """Run a module on the CPU while the rest of the model lives on the GPU."""

        def __init__(self, module: Any, device: str) -> None:
            super().__init__()
            # Registered first, so ``next(self.parameters())`` — which Chatterbox
            # uses to detect its device — still reports the GPU.
            self.device_anchor = torch.nn.Parameter(torch.zeros(1, device=device), requires_grad=False)
            self.inner = module
            self.target = device

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            out = self.inner(*_to_device(args, "cpu"), **_to_device(kwargs, "cpu"))
            return _to_device(out, self.target)

        def __getattr__(self, name: str) -> Any:
            try:
                return super().__getattr__(name)
            except AttributeError:
                if name == "inner":
                    raise
                return getattr(super().__getattr__("inner"), name)

    return _CPUOffload


def CPUOffload(module: Any, device: str) -> Any:  # noqa: N802 — factory mimicking a class
    return _cpu_offload_class()(module, device)


def load_turbo(device: str, half: bool = True) -> Any:
    """Load Chatterbox Turbo building every module directly on *device*.

    The stock loader instantiates the model in system RAM at fp32 and maps
    the checkpoint alongside it (~5 GB peak) — too much for 8 GB laptops.
    Constructing under ``torch.device(device)`` and streaming the safetensors
    straight to that device keeps host-RAM use to a few hundred MB.
    """
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from chatterbox.tts_turbo import ChatterboxTurboTTS, Conditionals, REPO_ID
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config
    from chatterbox.models.s3gen import S3Gen
    from chatterbox.models.voice_encoder import VoiceEncoder

    ckpt = Path(snapshot_download(
        repo_id=REPO_ID, token=os.getenv("HF_TOKEN") or None,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"],
    ))

    def load(module: Any, file: str, strict: bool = True) -> Any:
        state = stream_safetensors(ckpt / file, device)
        if "model" in state:
            state = state["model"][0]
        module.load_state_dict(state, strict=strict)
        del state
        gc.collect()
        return module.to(device).eval()

    with torch.device(device):
        ve = load(VoiceEncoder(), "ve.safetensors")

        hp = T3Config(text_tokens_dict_size=50276)
        hp.llama_config_name = "GPT2_medium"
        hp.speech_tokens_dict_size = 6563
        hp.input_pos_emb = None
        hp.speech_cond_prompt_len = 375
        hp.use_perceiver_resampler = False
        hp.emotion_adv = False
        t3 = load(T3(hp), "t3_turbo_v1.safetensors")
        del t3.tfmr.wte
        if half and device != "cpu":
            # The 350M-parameter token model dominates VRAM; fp16 halves it so
            # everything fits in 4 GB cards without spilling to system RAM
            # (which is ~20x slower). Generation runs under autocast.
            t3.half()
            torch.cuda.empty_cache()

        s3gen = load(S3Gen(meanflow=True), "s3gen_meanflow.safetensors")
        if half and device != "cpu":
            # (The flow decoder stays fp32: in fp16 the cloned words become
            # noticeably less intelligible — "hold on" was heard as "I'll dawn".)
            # The 124M-param speech tokenizer only runs when a new voice is
            # conditioned (rarely): keep it on the CPU and free ~0.5 GB of VRAM.
            s3gen.tokenizer = CPUOffload(s3gen.tokenizer.cpu(), device)
            torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    conds = Conditionals.load(ckpt / "conds.pt", map_location=device).to(device) if (ckpt / "conds.pt").exists() else None
    return ChatterboxTurboTTS(t3, s3gen, ve, tokenizer, device, conds=conds)


def load_standard(device: str) -> Any:
    from chatterbox.tts import ChatterboxTTS
    return ChatterboxTTS.from_pretrained(device=device)


# ---------------------------------------------------------------------------
# Voice bank
# ---------------------------------------------------------------------------

class VoiceBank:
    """Reference audio and prepared conditioning per speaker."""

    def __init__(self, model: Any, min_seconds: float, max_seconds: float) -> None:
        self.model = model
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds
        self.audio: dict[str, np.ndarray] = {}
        self.conds: dict[str, Any] = {}
        self.conds_seconds: dict[str, float] = {}
        self.lock = threading.Lock()  # one GPU, one job at a time
        self.urgent = 0               # live requests waiting: pre-warming yields to them
        self.cache: OrderedDict[tuple[str, str], bytes] = OrderedDict()

    def seconds(self, speaker: str) -> float:
        a = self.audio.get(speaker)
        return 0.0 if a is None else len(a) / SR_OUT

    def enroll(self, speaker: str, pcm: bytes) -> dict:
        new = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        joined = np.concatenate([self.audio.get(speaker, np.zeros(0, np.float32)), new])
        self.audio[speaker] = joined[-int(self.max_seconds * SR_OUT):]  # keep the most recent speech
        secs = self.seconds(speaker)
        # Re-condition when first possible, then whenever reference grew by 50 %.
        if secs >= self.min_seconds and secs >= 1.5 * self.conds_seconds.get(speaker, 0.0):
            self._condition(speaker)
        return {"speaker": speaker, "seconds": round(secs, 1), "ready": speaker in self.conds}

    def _condition(self, speaker: str) -> None:
        import soundfile as sf

        t0 = time.time()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            sf.write(path, self.audio[speaker], SR_OUT)
            import torch

            # fp32 here: the reference mel-spectrogram is not fp16-safe (only generation uses autocast).
            with self.lock, torch.inference_mode():
                self.model.prepare_conditionals(path)
                # prepare_conditionals builds a new object each call, so keeping
                # the reference is safe; switching speakers just swaps it back in.
                self.conds[speaker] = self.model.conds
        finally:
            os.unlink(path)  # reference audio never persists on disk
        self.conds_seconds[speaker] = self.seconds(speaker)
        self.cache = OrderedDict((k, v) for k, v in self.cache.items() if k[0] != speaker)
        log.info("conditioned voice %s on %.1fs in %.2fs", speaker, self.conds_seconds[speaker], time.time() - t0)

    def prewarm(self, speaker: str, texts: list[str]) -> None:
        """Synthesise *texts* in the background so later requests are cache hits."""
        def run() -> None:
            t0, done = time.time(), 0
            for text in texts:
                while self.urgent:
                    time.sleep(0.05)
                if speaker not in self.conds:
                    return
                try:
                    if not self.synthesize(speaker, text, priority=False)[1]:  # type: ignore[index]
                        done += 1
                except Exception as e:
                    log.warning("prewarm %r failed: %s", text, e)
            log.info("prewarmed %d phrases for %s in %.1fs", done, speaker, time.time() - t0)
        threading.Thread(target=run, daemon=True).start()

    def synthesize(self, speaker: str, text: str, priority: bool = True) -> Optional[tuple[bytes, bool]]:
        if speaker not in self.conds:
            return None
        key = (speaker, text.lower().strip())
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key], True

        import torch
        import torchaudio.functional as AF

        if priority:
            self.urgent += 1
        try:
            with self.lock, torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=self.model.device != "cpu"):
                self.model.conds = self.conds[speaker]
                wav = self.model.generate(text)
        finally:
            if priority:
                self.urgent -= 1
        wav = AF.resample(wav.float().cpu(), self.model.sr, SR_OUT).squeeze(0).numpy()
        wav = _trim_silence(wav)
        pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16).tobytes()
        self.cache[key] = pcm
        while len(self.cache) > 256:
            self.cache.popitem(last=False)
        return pcm, False


def private_memory_mb() -> float:
    """This process's committed private memory (what exhausts the page file)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                               "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                               "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                                               "PrivateUsage")]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.PrivateUsage / 2**20
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _trim_silence(wav: np.ndarray, rel_threshold: float = 0.03, pad: int = 640) -> np.ndarray:
    """Remove leading/trailing silence so a spliced word sits tight in the sentence.

    Threshold is relative to the clip's peak and 40 ms is kept either side,
    so soft consonants at word edges ("j" in "jerk") are not clipped.
    """
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    if peak <= 0:
        return wav
    loud = np.flatnonzero(np.abs(wav) > rel_threshold * peak)
    if not len(loud):
        return wav
    return wav[max(0, loud[0] - pad): loud[-1] + pad]


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------

try:  # request bodies must be module-level so FastAPI can resolve the annotations
    from pydantic import BaseModel

    class EnrollReq(BaseModel):
        speaker: str
        pcm_b64: str

    class SynthReq(BaseModel):
        speaker: str
        text: str

    class PrewarmReq(BaseModel):
        speaker: str
        texts: list[str]
except ImportError:  # pragma: no cover — only the worker venv needs these
    pass


def create_app(bank_holder: dict) -> Any:
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="SpeechShield voice worker")

    def bank() -> VoiceBank:
        b = bank_holder.get("bank")
        if b is None:
            raise HTTPException(503, bank_holder.get("error") or "model loading")
        return b

    @app.get("/health")
    def health() -> dict:
        b = bank_holder.get("bank")
        info: dict[str, Any] = {"ready": b is not None, "error": bank_holder.get("error"), "device": bank_holder.get("device"),
                                "private_mb": round(private_memory_mb())}
        if b is not None:
            info["speakers"] = {s: {"seconds": round(b.seconds(s), 1), "ready": s in b.conds} for s in b.audio}
            try:
                import torch
                if torch.cuda.is_available():
                    info["vram_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
            except Exception:
                pass
        return info

    @app.post("/enroll")
    def enroll(req: EnrollReq) -> dict:
        return bank().enroll(req.speaker, base64.b64decode(req.pcm_b64))

    @app.post("/prewarm")
    def prewarm(req: PrewarmReq) -> dict:
        b = bank()
        if req.speaker not in b.conds:
            raise HTTPException(409, "speaker not enrolled yet")
        b.prewarm(req.speaker, [t for t in req.texts if 0 < len(t) <= 200][:100])
        return {"queued": len(req.texts)}

    @app.post("/synthesize")
    def synthesize(req: SynthReq) -> dict:
        if not req.text.strip() or len(req.text) > 200:
            raise HTTPException(400, "text must be 1-200 characters")
        t0 = time.time()
        out = bank().synthesize(req.speaker, req.text)
        if out is None:
            raise HTTPException(409, "speaker not enrolled yet")
        pcm, cached = out
        return {"pcm_b64": base64.b64encode(pcm).decode(), "sample_rate": SR_OUT,
                "cached": cached, "ms": round((time.time() - t0) * 1000, 1)}

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9502)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", choices=("turbo", "standard"), default="turbo")
    ap.add_argument("--min-ref", type=float, default=6.0)
    ap.add_argument("--max-ref", type=float, default=15.0)
    args = ap.parse_args()

    import torch
    import uvicorn

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    holder: dict[str, Any] = {"device": device}

    def load() -> None:
        t0 = time.time()
        try:
            model = load_turbo(device) if args.model == "turbo" else load_standard(device)
            holder["bank"] = VoiceBank(model, args.min_ref, args.max_ref)
            log.info("model %s ready on %s in %.1fs", args.model, device, time.time() - t0)
        except Exception as e:  # report through /health instead of dying
            holder["error"] = f"{type(e).__name__}: {e}"
            log.exception("model load failed")

    threading.Thread(target=load, daemon=True).start()
    uvicorn.run(create_app(holder), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
