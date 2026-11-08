"""
ASRAgent — Automatic Speech Recognition agent.

Receives speech_segment events (whole utterances from the segmenter) and
transcribes them with Faster-Whisper, including per-word timestamps so the
interceptor can censor individual words.

Also handles speech_partial events (streaming): transcribes an utterance
still in progress and emits partial_transcript with the words that are
stable enough to judge.

Emits:
  - text_transcript   — {text, words, speaker, duration}
  - partial_transcript — {words (committed), safe_until}
  - transcript_empty  — the utterance held no recognisable speech, so the
                        interceptor can release it immediately

Metrics:
  - speechshield_asr_latency_seconds — Inference latency
  - speechshield_asr_errors_total — Error count by reason
  - speechshield_utterances_processed_total — Processed utterance count
"""

import asyncio
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from itertools import takewhile
from typing import TYPE_CHECKING, Any

import numpy as np

from core.event_bus import Event
from core.lexicon import normalize_token
from core.logging import get_logger
from core.metrics import metrics
from core.health import health_checks

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.state import GlobalState

logger = get_logger("agents.asr_agent")

SAMPLE_RATE = 16_000

# Phrases Whisper is known to hallucinate on silence/noise.  Only dropped
# when Whisper itself is unsure there was speech at all.
_HALLUCINATIONS = {
    "thank you.", "thank you", "thanks for watching!", "thanks for watching.",
    "thank you for watching.", "you", "bye.", "subtitles by the amara.org community",
    ".", "...",
}


def resolve_device(device: str, compute_type: str) -> tuple[str, str]:
    """Pick the inference device and compute type for 'auto' settings."""
    if device == "auto":
        from core.gpu import cuda_device_count
        device = "cuda" if cuda_device_count() > 0 else "cpu"
    if compute_type == "auto":
        # int8 weights + fp16 compute: fastest on GPU and half the VRAM of fp16.
        compute_type = "int8_float16" if device == "cuda" else "int8"
    return device, compute_type


def resolve_model(model_size: str, device: str) -> str:
    """'auto' = small on GPU (~0.25 s/utterance), base.en on CPU (~1 s/utterance)."""
    if model_size != "auto":
        return model_size
    return "small" if device == "cuda" else "base.en"


class ASRAgent:
    """Transcription agent — converts utterances into text with word timings."""

    def __init__(self, event_bus: "EventBus", state: "GlobalState") -> None:
        self.bus = event_bus
        self.state = state
        self._model: Any = None
        self.model_name: str = state.settings.asr.model_size
        self.device = "cpu"
        # One worker: CTranslate2 already parallelises internally, and a
        # single queue keeps transcripts in utterance order.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")
        self._running = False
        self._stopped = asyncio.Event()
        self._ready = asyncio.Event()
        # Streaming (LocalAgreement) state per utterance: last hypothesis + committed count
        self._hypotheses: "OrderedDict[str, tuple[list[dict], int]]" = OrderedDict()
        self._finalized: deque[str] = deque(maxlen=64)

    # ------------------------------------------------------------------
    # Model loading (once)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        cfg = self.state.settings.asr
        device, compute_type = resolve_device(cfg.device, cfg.compute_type)
        self.model_name = resolve_model(cfg.model_size, device)
        t0 = time.time()
        try:
            self._model = self._create_and_warm(WhisperModel, device, compute_type)
        except Exception as e:
            # A GPU can be visible without a usable CUDA runtime (e.g. cuBLAS
            # missing). In auto mode, fall back to CPU instead of failing.
            if device != "cuda" or cfg.device != "auto":
                logger.error("asr_model_load_failed", error=str(e))
                health_checks().mark_error("ASRAgent", f"Model load failed: {e}")
                raise
            logger.warning("asr_cuda_unusable_falling_back_to_cpu", error=str(e))
            device, compute_type = "cpu", ("int8" if cfg.compute_type == "auto" else cfg.compute_type)
            self.model_name = resolve_model(cfg.model_size, device)
            self._model = self._create_and_warm(WhisperModel, device, compute_type)
        self.device = device
        self.state.system_info["asr_device"] = device
        self.state.system_info["asr"] = f"faster-whisper {self.model_name} ({device}/{compute_type})"
        logger.info("asr_model_loaded", model=self.model_name, device=device, compute_type=compute_type,
                    load_ms=int((time.time() - t0) * 1000))

    def _create_and_warm(self, model_cls: Any, device: str, compute_type: str) -> Any:
        """Load the model and run one inference (pays one-off costs, proves the device works)."""
        cfg = self.state.settings.asr
        logger.info("asr_model_loading", model=self.model_name, device=device, compute_type=compute_type)
        model = model_cls(
            self.model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=cfg.cpu_threads,
            download_root=cfg.model_dir,
        )
        noise = (np.random.default_rng(0).standard_normal(SAMPLE_RATE) * 0.01).astype(np.float32)
        segments, _ = model.transcribe(noise, beam_size=1, language=cfg.language, vad_filter=False)
        list(segments)
        return model

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to speech_segment events, then load the model.

        Utterances captured while the model loads wait in the inbox.
        """
        self._stopped.clear()
        self._running = True
        self.bus.subscribe("speech_segment", self._handle_segment)
        self.bus.subscribe("speech_partial", self._handle_partial, queue_size=4)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._load_model)
        self._ready.set()
        health_checks().mark_ready("ASRAgent")
        logger.info("asr_agent_started", subscribed_to="speech_segment")

        await self._stopped.wait()

    def stop(self) -> None:
        self._running = False
        self._stopped.set()
        health_checks().mark_starting("ASRAgent")

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    async def _handle_partial(self, event: Event) -> None:
        """Streaming pass over an utterance still in progress.

        Uses the LocalAgreement policy: a word is *committed* once two
        consecutive hypotheses agree on it and it ends at least
        ``commit_guard_seconds`` before the live edge.  Committed words are
        stable enough to judge and release.
        """
        audio: bytes = event.data.get("audio", b"")
        cid = event.correlation_id
        if not audio or cid in self._finalized:
            return
        await self._ready.wait()
        if cid in self._finalized:  # the final transcription overtook this pass
            return
        loop = asyncio.get_running_loop()
        try:
            _, words = await asyncio.wait_for(
                loop.run_in_executor(self._executor, self._transcribe, audio),
                timeout=self.state.settings.asr.timeout_seconds,
            )
        except Exception as e:
            logger.debug("asr_partial_failed", correlation_id=cid, error=str(e))
            words = []
        if cid in self._finalized:
            return

        duration = len(audio) / 2 / event.data.get("sample_rate", SAMPLE_RATE)
        guard = self.state.settings.interceptor.commit_guard_seconds
        committed = self._commit(cid, words, duration - guard)
        await self.bus.emit(Event("partial_transcript", {
            "words": committed,
            "safe_until": committed[-1]["end"] if committed else 0.0,
            "duration": duration,
            "speaker": self.state.trace(cid).speaker or self.state.current_speaker,
        }, correlation_id=cid))

    def _commit(self, cid: str, words: list[dict], edge: float) -> list[dict]:
        """LocalAgreement-2: commit the prefix two consecutive passes agree on."""
        prev, committed = self._hypotheses.get(cid, ([], 0))
        agree = 0
        for a, b in zip(prev, words, strict=False):  # hypotheses differ in length by design
            if normalize_token(a["word"]) != normalize_token(b["word"]):
                break
            agree += 1
        stable = sum(1 for _ in takewhile(lambda w: w["end"] <= edge, words))
        committed = min(len(words), max(committed, min(agree, stable)))
        self._hypotheses[cid] = (words, committed)
        self._hypotheses.move_to_end(cid)
        while len(self._hypotheses) > 32:
            self._hypotheses.popitem(last=False)
        return words[:committed]

    async def _handle_segment(self, event: Event) -> None:
        audio: bytes = event.data.get("audio", b"")
        cid = event.correlation_id
        if not audio:
            return
        self._finalized.append(cid)
        self._hypotheses.pop(cid, None)
        await self._ready.wait()

        cfg = self.state.settings.asr
        m = metrics()
        loop = asyncio.get_running_loop()
        start_time = time.time()
        try:
            with m.asr_latency_seconds.time():
                text, words = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, self._transcribe, audio),
                    timeout=cfg.timeout_seconds,
                )
        except asyncio.TimeoutError:
            logger.error("asr_timeout", correlation_id=cid, timeout_s=cfg.timeout_seconds)
            m.asr_errors_total.labels(reason="timeout").inc()
            m.utterances_processed_total.labels(status="error").inc()
            health_checks().mark_degraded("ASRAgent", "Inference timeout")
            await self._emit_empty(cid, "timeout")
            return
        except Exception as e:
            logger.error("asr_error", correlation_id=cid, error=str(e), error_type=type(e).__name__)
            m.asr_errors_total.labels(reason="decode_error").inc()
            m.utterances_processed_total.labels(status="error").inc()
            health_checks().mark_degraded("ASRAgent", f"Inference error: {e}")
            await self._emit_empty(cid, "error")
            return

        elapsed_ms = int((time.time() - start_time) * 1000)
        stage_ms = self.state.mark_stage(cid, "asr")
        m.stage_latency_seconds.labels(stage="asr").observe(stage_ms / 1000.0)

        if not text:
            logger.debug("asr_no_speech_detected", correlation_id=cid, duration_ms=elapsed_ms)
            m.asr_errors_total.labels(reason="no_speech").inc()
            await self._emit_empty(cid, "no_speech")
            return

        trace = self.state.trace(cid)
        trace.transcript = text
        trace.words = words
        speaker = trace.speaker or self.state.current_speaker

        logger.info("asr_transcript_emitted", correlation_id=cid, text_length=len(text), words=len(words), duration_ms=elapsed_ms)
        m.utterances_processed_total.labels(status="success").inc()
        await self.bus.emit(Event("text_transcript", {
            "text": text,
            "words": words,
            "speaker": speaker,
            "duration": event.data.get("duration", 0.0),
            "timestamp": time.time(),
        }, correlation_id=cid))

    async def _emit_empty(self, cid: str, reason: str) -> None:
        await self.bus.emit(Event("transcript_empty", {"reason": reason, "timestamp": time.time()}, correlation_id=cid))

    # ------------------------------------------------------------------
    # Transcription helper
    # ------------------------------------------------------------------

    def _transcribe(self, raw_pcm: bytes) -> tuple[str, list[dict]]:
        """Run Faster-Whisper on int16 PCM; returns (text, words)."""
        cfg = self.state.settings.asr
        audio = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0

        segments, _info = self._model.transcribe(
            audio,
            beam_size=cfg.beam_size,
            best_of=1,
            temperature=0.0,                    # no fallback re-decoding: predictable latency
            language=cfg.language,
            condition_on_previous_text=False,   # utterances are independent; avoids runaway repeats
            vad_filter=cfg.vad_filter,
            vad_parameters={"min_silence_duration_ms": cfg.vad_min_silence_ms} if cfg.vad_filter else None,
            word_timestamps=cfg.word_timestamps,
            no_speech_threshold=cfg.no_speech_threshold,
            log_prob_threshold=cfg.min_avg_logprob,
        )

        texts: list[str] = []
        words: list[dict] = []
        for seg in segments:
            seg_text = seg.text.strip()
            if self._is_hallucination(seg_text, seg.no_speech_prob, seg.avg_logprob):
                metrics().asr_errors_total.labels(reason="hallucination").inc()
                continue
            texts.append(seg_text)
            for w in seg.words or ():
                words.append({
                    "word": w.word.strip(),
                    "start": round(float(w.start), 3),
                    "end": round(float(w.end), 3),
                    "probability": round(float(w.probability), 3),
                })
        return " ".join(t for t in texts if t).strip(), words

    def _is_hallucination(self, text: str, no_speech_prob: float, avg_logprob: float) -> bool:
        if not text:
            return True
        if text.lower() in _HALLUCINATIONS and (no_speech_prob > 0.3 or avg_logprob < -0.8):
            return True
        return False
