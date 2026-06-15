"""
ToxicityAgent — Detects toxic speech using HuggingFace unitary/toxic-bert.

Receives text_transcript events, classifies toxicity, and emits
toxicity_result events with toxic flag and confidence score.

Metrics:
  - speechshield_toxicity_latency_seconds — Classification latency
  - speechshield_toxicity_detections_total — Detections by severity
"""

import asyncio
import re
import time
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from core.event_bus import Event
from core.logging import get_logger
from core.metrics import metrics
from core.health import health_checks

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.state import GlobalState

logger = get_logger("agents.toxicity_agent")

MODEL_NAME = "unitary/toxic-bert"

# Phrases that should never be flagged as toxic regardless of model score.
# Checked case-insensitively against the full transcript.
_SAFE_PHRASES: set[str] = {
    "thank you", "thanks", "please", "sorry", "excuse me",
    "good morning", "good afternoon", "good evening", "good night",
    "hello", "hi", "hey", "goodbye", "bye", "welcome",
    "yes", "no", "okay", "ok", "sure", "of course",
    "i agree", "well done", "great job", "nice work",
}


def _is_safe(text: str) -> bool:
    """Return True if *text* is clearly benign and should skip classification."""
    normalised = text.strip().lower().rstrip(".!?,")
    # Exact match
    if normalised in _SAFE_PHRASES:
        return True
    # Starts with a safe greeting/pleasantry
    for phrase in _SAFE_PHRASES:
        if normalised.startswith(phrase):
            return True
    return False


# ------------------------------------------------------------------
# Profanity keyword pre-check — catches obvious slurs the model may
# under-score.  Matched as whole words, case-insensitive.
# ------------------------------------------------------------------
_PROFANE_WORDS: set[str] = {
    "fuck", "fucking", "fucked", "fucker", "fuckers", "fucks",
    "shit", "shitty", "shitting", "bullshit",
    "ass", "asshole", "assholes", "arse", "arsehole",
    "bitch", "bitches", "bitching",
    "damn", "damned", "dammit", "goddamn",
    "dick", "dicks", "dickhead",
    "cock", "cocks", "cocksucker",
    "cunt", "cunts",
    "pussy", "pussies",
    "bastard", "bastards",
    "piss", "pissed", "pissing",
    "slut", "sluts", "slutty",
    "whore", "whores",
    "nigger", "nigga", "niggas",
    "faggot", "fag", "fags",
    "retard", "retarded", "retards",
    "stfu", "gtfo", "wtf", "lmfao",
    "motherfucker", "motherfucking",
    "crap", "crappy",
    "twat", "wanker", "tosser",
    "kills", "kill", "die", "murder",
}

# Pre-compiled regex for fast matching
_PROFANE_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in _PROFANE_WORDS) + r")\b",
    re.IGNORECASE,
)


def _has_profanity(text: str) -> bool:
    """Return True if text contains any known profane word."""
    return bool(_PROFANE_PATTERN.search(text))


class ToxicityAgent:
    """Classifies transcripts as toxic / non-toxic."""

    def __init__(self, event_bus: "EventBus", state: "GlobalState") -> None:
        self.bus = event_bus
        self.state = state
        self._model = None
        self._tokenizer = None
        self._label_names: list[str] = []
        self._running = False

    # ------------------------------------------------------------------
    # Model loading (once)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is None:
            logger.info("toxicity_model_loading", model="toxic-bert")
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                self._model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
                self._model.eval()
                # Label order from the model config (toxic, severe_toxic, obscene, threat, insult, identity_hate)
                self._label_names = list(self._model.config.id2label.values())
                logger.info("toxicity_model_loaded", model="toxic-bert", labels=len(self._label_names))
            except Exception as e:
                error_msg = f"Model load failed: {str(e)}"
                logger.error("toxicity_model_load_failed", error=str(e))
                health_checks().mark_error("ToxicityAgent", error_msg)
                raise

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._load_model()
        self._running = True
        self.bus.subscribe("text_transcript", self._handle_transcript)
        health_checks().mark_ready("ToxicityAgent")
        logger.info("toxicity_agent_started", subscribed_to="text_transcript")

        while self._running:
            await asyncio.sleep(0.1)

    def stop(self) -> None:
        self._running = False
        health_checks().mark_starting("ToxicityAgent")

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    async def _handle_transcript(self, event: Event) -> None:
        text: str = event.data.get("text", "")
        speaker: str = event.data.get("speaker", "default")
        if not text:
            return

        m = metrics()
        start_time = time.time()

        # Fast-path: skip model entirely for clearly safe utterances
        if _is_safe(text):
            logger.debug("safe_phrase_skip", text_length=len(text), speaker=speaker)
            self.state.record_utterance(speaker, False)
            await self.bus.emit(
                Event(
                    "toxicity_result",
                    {
                        "text": text,
                        "speaker": speaker,
                        "toxic": False,
                        "confidence": 0.0,
                        "labels": {},
                        "strictness": self.state.get_speaker(speaker).rewrite_strictness,
                        "timestamp": time.time(),
                    },
                    correlation_id=event.correlation_id,
                )
            )
            return

        try:
            loop = asyncio.get_running_loop()
            
            with m.toxicity_latency_seconds.time():
                result = await loop.run_in_executor(None, self._classify, text)

            toxic: bool = result["toxic"]
            confidence: float = result["confidence"]
            labels: dict = result["labels"]

            # Update shared state
            self.state.record_utterance(speaker, toxic)
            profile = self.state.get_speaker(speaker)

            # Record metrics
            if toxic:
                severity = "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
                m.toxicity_detections_total.labels(severity=severity).inc()

            elapsed = time.time() - start_time
            logger.info(
                "toxicity_classification_complete",
                correlation_id=event.correlation_id,
                toxic=toxic,
                confidence=round(confidence, 3),
                text_length=len(text),
                duration_ms=int(elapsed * 1000),
            )

            await self.bus.emit(
                Event(
                    "toxicity_result",
                    {
                        "text": text,
                        "speaker": speaker,
                        "toxic": toxic,
                        "confidence": confidence,
                        "labels": labels,
                        "strictness": profile.rewrite_strictness,
                        "timestamp": time.time(),
                    },
                    correlation_id=event.correlation_id,
                )
            )
        except asyncio.TimeoutError:
            logger.error(
                "toxicity_timeout",
                correlation_id=event.correlation_id,
                error="Classification timed out",
                text_length=len(text),
            )
            health_checks().mark_error("ToxicityAgent", "Classification timeout")
        except Exception as e:
            logger.error(
                "toxicity_error",
                correlation_id=event.correlation_id,
                error=str(e),
                error_type=type(e).__name__,
                text_length=len(text),
            )
            health_checks().mark_error("ToxicityAgent", f"Classification error: {str(e)}")

    # ------------------------------------------------------------------
    # Classification helper
    # ------------------------------------------------------------------

    # Labels from toxic-bert that indicate toxicity
    _TOXIC_LABELS = {"toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"}

    def _classify(self, text: str) -> dict:
        """Run toxic-bert with manual sigmoid on raw logits for correct multi-label scores.

        Also runs a keyword-based profanity check so obvious slurs are never
        missed even when the model under-scores them.
        """
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = self._model(**inputs).logits          # shape: (1, 6)
        probs = torch.sigmoid(logits).squeeze(0).tolist()  # list of 6 floats

        label_map: dict[str, float] = {}
        for i, label_name in enumerate(self._label_names):
            label_map[label_name] = probs[i]

        # Flag as toxic if ANY toxic label exceeds the threshold.
        threshold = self.state.config.get("toxicity_threshold", 0.5)
        max_toxic_score = 0.0
        for label in self._TOXIC_LABELS:
            score = label_map.get(label, 0.0)
            if score > max_toxic_score:
                max_toxic_score = score

        is_toxic = max_toxic_score >= threshold

        # --- Keyword safety net ---
        # If the model missed it but a known profane word is present,
        # override the decision and assign a high confidence.
        if not is_toxic and _has_profanity(text):
            logger.info("Keyword override — profanity detected in '%s'", text[:80])
            is_toxic = True
            max_toxic_score = max(max_toxic_score, 0.90)
            label_map["_keyword_override"] = 1.0

        return {
            "toxic": is_toxic,
            "confidence": max_toxic_score,
            "labels": label_map,
        }
