"""
ToxicityAgent — Detects toxic speech using HuggingFace unitary/toxic-bert.

Receives text_transcript events, classifies toxicity, works out *which
words* are toxic (time spans, from Whisper word timestamps), and emits
toxicity_result events.

If the model cannot be loaded the agent degrades to keyword-only
detection instead of taking the pipeline down.

Metrics:
  - speechshield_toxicity_latency_seconds — Classification latency
  - speechshield_toxicity_detections_total — Detections by severity
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from core.event_bus import Event
from core.lexicon import first_mild_word_start, has_strong_profanity, toxic_word_spans
from core.logging import get_logger
from core.metrics import metrics
from core.health import health_checks

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.state import GlobalState

logger = get_logger("agents.toxicity_agent")

MODEL_NAME = "unitary/toxic-bert"

# Utterances that are exactly one of these skip the model.  Only *whole*
# utterances match: prefix matching would let "No, you idiot" or
# "Please shut the f*** up" bypass classification.
_SAFE_PHRASES: set[str] = {
    "thank you", "thanks", "please", "sorry", "excuse me",
    "good morning", "good afternoon", "good evening", "good night",
    "hello", "hi", "hey", "goodbye", "bye", "welcome",
    "yes", "no", "okay", "ok", "sure", "of course",
    "i agree", "well done", "great job", "nice work",
    "thank you very much", "thanks a lot", "see you", "good game", "gg",
}


def _is_safe(text: str) -> bool:
    """Return True if *text* is, in its entirety, a known-benign phrase."""
    normalised = " ".join(text.strip().lower().replace(",", " ").split()).rstrip(".!?,")
    return normalised in _SAFE_PHRASES


def _has_profanity(text: str) -> bool:
    """Return True if text contains strong profanity, a slur or a threat phrase."""
    return has_strong_profanity(text)


def judge_partial(words: list[dict], prefix_toxic: bool, safe_until: float) -> tuple[float, list[dict], bool]:
    """Verdict for the committed words of an utterance still in progress.

    Returns ``(safe_until, spans, meaning_toxic)``: how far audio may be
    released, which words to censor, and whether the prefix is toxic in a way
    no single word explains (then the rest waits for the final verdict).
    """
    spans = [sp.to_dict() for sp in toxic_word_spans(words, utterance_toxic=prefix_toxic)]
    if not prefix_toxic:
        # Mild words ("shut up", "trash") are only censored when the whole
        # utterance turns out toxic — which a fragment can't tell us yet.
        # Never release past one until a later pass or the final verdict decides.
        safe_until = min(safe_until, first_mild_word_start(words))
    return safe_until, spans, prefix_toxic and not spans


def _severity(confidence: float) -> str:
    return "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"


class ToxicityAgent:
    """Classifies transcripts as toxic / non-toxic and locates toxic words."""

    # Labels from toxic-bert that indicate toxicity
    _TOXIC_LABELS = {"toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"}

    def __init__(self, event_bus: "EventBus", state: "GlobalState") -> None:
        self.bus = event_bus
        self.state = state
        self._model: Any = None
        self._tokenizer: Any = None
        self._label_names: list[str] = []
        self._keyword_only = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="toxicity")
        self._running = False
        self._stopped = asyncio.Event()
        self._ready = asyncio.Event()

    # ------------------------------------------------------------------
    # Model loading (once)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None or self._keyword_only:
            return
        cfg = self.state.settings.toxicity
        logger.info("toxicity_model_loading", model=cfg.model_name)
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            from transformers.utils import logging as hf_logging

            hf_logging.set_verbosity_error()
            hf_logging.disable_progress_bar()
            if cfg.num_threads:
                torch.set_num_threads(cfg.num_threads)
            try:  # cached copy first: no network round-trip on every start
                self._tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, local_files_only=True)
                model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, local_files_only=True)
            except OSError:
                logger.info("toxicity_model_downloading", model=cfg.model_name)
                self._tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
                model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name)
            model.eval()
            quantized = False
            if cfg.quantize and not torch.cuda.is_available():
                try:
                    model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
                    quantized = True
                except Exception as qe:  # unsupported backend — run in fp32
                    logger.warning("toxicity_quantization_failed", error=str(qe))
            self._model = model
            # Label order from the model config (toxic, severe_toxic, obscene, threat, insult, identity_hate)
            self._label_names = list(model.config.id2label.values())
            self._classify("warm up")  # first call pays one-off allocation costs
            self.state.system_info["toxicity"] = f"{cfg.model_name}{' (int8)' if quantized else ''}"
            logger.info("toxicity_model_loaded", model=cfg.model_name, labels=len(self._label_names), quantized=quantized)
        except Exception as e:
            # Fail safe, not fail dead: keep moderating with the lexicon.
            self._keyword_only = True
            self._model = None
            self.state.system_info["toxicity"] = "keyword-only (model unavailable)"
            logger.error("toxicity_model_load_failed", error=str(e), fallback="keyword_only")
            health_checks().mark_degraded("ToxicityAgent", f"Model load failed, keyword-only: {e}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._stopped.clear()
        # Subscribe first: transcripts arriving while the model loads wait in
        # the inbox instead of being lost.
        self._running = True
        self.bus.subscribe("text_transcript", self._handle_transcript)
        self.bus.subscribe("partial_transcript", self._handle_partial, queue_size=4)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._load_model)
        self._ready.set()
        if not self._keyword_only:
            health_checks().mark_ready("ToxicityAgent")
        logger.info("toxicity_agent_started", subscribed_to="text_transcript", keyword_only=self._keyword_only)

        await self._stopped.wait()

    def stop(self) -> None:
        self._running = False
        self._stopped.set()
        health_checks().mark_starting("ToxicityAgent")

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    async def _handle_transcript(self, event: Event) -> None:
        text: str = event.data.get("text", "")
        speaker: str = event.data.get("speaker", "default")
        words: list[dict] = event.data.get("words") or []
        cid = event.correlation_id
        if not text:
            return
        await self._ready.wait()

        m = metrics()
        start_time = time.time()
        cfg = self.state.settings.toxicity

        result: dict[str, Any]
        if _is_safe(text):
            result = {"toxic": False, "confidence": 0.0, "labels": {}}
            logger.debug("safe_phrase_skip", correlation_id=cid, text_length=len(text))
        else:
            loop = asyncio.get_running_loop()
            try:
                with m.toxicity_latency_seconds.time():
                    result = await asyncio.wait_for(
                        loop.run_in_executor(self._executor, self._classify, text),
                        timeout=cfg.timeout_seconds,
                    )
            except asyncio.TimeoutError:
                # Never wave an utterance through just because the model was slow.
                logger.error("toxicity_timeout", correlation_id=cid, text_length=len(text))
                health_checks().mark_degraded("ToxicityAgent", "Classification timeout")
                result = self._keyword_classify(text)

        toxic: bool = result["toxic"]
        confidence: float = result["confidence"]
        labels: dict = result["labels"]

        spans = [s.to_dict() for s in toxic_word_spans(words, utterance_toxic=toxic)] if words else []
        if toxic and words and not spans:
            # Toxic by meaning ("I will find you") rather than vocabulary:
            # censor the whole utterance.
            spans = [{"start": words[0]["start"], "end": words[-1]["end"], "text": text, "whole": True}]

        # Update shared state
        self.state.record_utterance(speaker, toxic)
        profile = self.state.get_speaker(speaker)
        trace = self.state.trace(cid)
        trace.toxic, trace.confidence, trace.labels, trace.toxic_spans = toxic, confidence, labels, spans
        stage_ms = self.state.mark_stage(cid, "toxicity")
        m.stage_latency_seconds.labels(stage="toxicity").observe(stage_ms / 1000.0)

        if toxic:
            m.toxicity_detections_total.labels(severity=_severity(confidence)).inc()

        logger.info(
            "toxicity_classification_complete",
            correlation_id=cid,
            toxic=toxic,
            confidence=round(confidence, 3),
            spans=len(spans),
            text_length=len(text),
            duration_ms=int((time.time() - start_time) * 1000),
        )

        await self.bus.emit(Event("toxicity_result", {
            "text": text,
            "speaker": speaker,
            "toxic": toxic,
            "confidence": confidence,
            "labels": labels,
            "toxic_spans": spans,
            "strictness": profile.rewrite_strictness,
            "source": event.data.get("source", "audio"),
            "timestamp": time.time(),
        }, correlation_id=cid))

    async def _handle_partial(self, event: Event) -> None:
        """Judge the committed words of an utterance still in progress (streaming)."""
        words: list[dict] = event.data.get("words") or []
        cid = event.correlation_id
        spans: list[dict] = []
        meaning_toxic = False
        safe_until = float(event.data.get("safe_until", 0.0))
        if words:
            await self._ready.wait()
            text = " ".join(w["word"] for w in words)
            loop = asyncio.get_running_loop()
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, self._classify, text),
                    timeout=self.state.settings.toxicity.timeout_seconds,
                )
            except asyncio.TimeoutError:
                result = self._keyword_classify(text)
            safe_until, spans, meaning_toxic = judge_partial(words, bool(result["toxic"]), safe_until)
        await self.bus.emit(Event("partial_verdict", {
            "safe_until": safe_until,
            "toxic_spans": spans,
            "meaning_toxic": meaning_toxic,
            "timestamp": time.time(),
        }, correlation_id=cid))

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    async def score(self, text: str) -> float:
        """Toxicity score of arbitrary text (used to verify LLM rewrites)."""
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self._executor, self._classify, text)
        return float(result["confidence"])

    def _keyword_classify(self, text: str) -> dict:
        hit = _has_profanity(text)
        return {"toxic": hit, "confidence": 0.9 if hit else 0.0, "labels": {"_keyword_override": 1.0} if hit else {}}

    def _classify(self, text: str) -> dict:
        """Run toxic-bert with sigmoid on raw logits for correct multi-label scores.

        Also runs a keyword check so slurs and threats are never missed even
        when the model under-scores them.
        """
        if self._model is None:
            return self._keyword_classify(text)

        import torch

        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.inference_mode():
            logits = self._model(**inputs).logits          # shape: (1, 6)
        probs = torch.sigmoid(logits).squeeze(0).tolist()  # list of 6 floats

        label_map: dict[str, float] = {name: round(float(probs[i]), 4) for i, name in enumerate(self._label_names)}

        # Flag as toxic if ANY toxic label exceeds the threshold.
        threshold = self.state.settings.toxicity.threshold
        max_toxic_score = max((label_map.get(lbl, 0.0) for lbl in self._TOXIC_LABELS), default=0.0)
        is_toxic = max_toxic_score >= threshold

        # --- Keyword safety net ---
        if not is_toxic and self.state.settings.toxicity.keyword_safety_net and _has_profanity(text):
            logger.info("keyword_override", text_length=len(text))
            is_toxic = True
            max_toxic_score = max(max_toxic_score, 0.90)
            label_map["_keyword_override"] = 1.0

        return {
            "toxic": is_toxic,
            "confidence": max_toxic_score,
            "labels": label_map,
        }
