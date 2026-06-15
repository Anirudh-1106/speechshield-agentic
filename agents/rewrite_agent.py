"""
RewriteAgent — Rewrites toxic speech into polite alternatives.

Receives toxicity_result events.
- If toxic: attempts LLM-based rewrite (Ollama) first, falls back to rules.
- If not toxic: passes through the original text unchanged.
Emits rewrite_result events.

Metrics:
  - speechshield_rewrite_latency_seconds — Rewrite latency
  - speechshield_rewrite_attempts_total — Attempt count by status
  - speechshield_rewrite_fallback_total — Fallback count by reason
"""

import asyncio
import re
import time
from typing import TYPE_CHECKING

from core.event_bus import Event
from core.logging import get_logger
from core.metrics import metrics
from core.health import health_checks
from agents.llm_rewriter import LLMRewriter

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.state import GlobalState

logger = get_logger("agents.rewrite_agent")

# -----------------------------------------------------------------------
# Rule-based rewrite table  (deterministic, no randomness)
#
# DESIGN PRINCIPLE — *word-level drop-in replacements*.
# Every replacement must be a short phrase that can sit naturally in the
# *same grammatical slot* as the original word so the rest of the
# sentence continues to flow.  Never replace a word with a full
# independent sentence — that breaks continuity.
#
# Example:  "You are such a stupid person and I can't stand it"
#        →  "You are such an unwise person and I can't stand it"
# -----------------------------------------------------------------------
_REWRITE_RULES: list[tuple[re.Pattern, str]] = [
    # --- Multi-word phrases (check first so they aren't partially matched) ---
    (re.compile(r"\bshut\s+the\s+fuck\s+up\b",   re.I), "please be quiet"),
    (re.compile(r"\bshut\s+up\b",                 re.I), "please be quiet"),
    (re.compile(r"\bstfu\b",                      re.I), "please be quiet"),
    (re.compile(r"\bfuck\s+off\b",                re.I), "please go away"),
    (re.compile(r"\bfuck\s+you\b",                re.I), "I disagree with you"),
    (re.compile(r"\bfucking\b",                   re.I), "very"),
    (re.compile(r"\bfucked\s+up\b",               re.I), "messed up"),
    (re.compile(r"\bbull\s*shit\b",                re.I), "nonsense"),
    (re.compile(r"\bass\s*hole\b",                 re.I), "rude person"),
    (re.compile(r"\byou\s+suck\b",                re.I), "you are not doing well"),
    (re.compile(r"\bkill\s+yourself\b",            re.I), "please reconsider"),
    (re.compile(r"\bkill\s+you\b",                re.I), "confront you"),
    (re.compile(r"\bhate\s+you\b",                re.I), "disagree with you"),
    (re.compile(r"\bi\s+hate\b",                  re.I), "I dislike"),
    (re.compile(r"\bgo\s+to\s+hell\b",            re.I), "please go away"),
    (re.compile(r"\bwhat\s+the\s+hell\b",         re.I), "what on earth"),
    (re.compile(r"\bwhat\s+the\s+fuck\b",         re.I), "what on earth"),
    (re.compile(r"\bpiece\s+of\s+shit\b",         re.I), "poor-quality thing"),
    (re.compile(r"\bson\s+of\s+a\s+bitch\b",      re.I), "unpleasant person"),
    (re.compile(r"\bdumbass\b",                   re.I), "foolish person"),
    (re.compile(r"\bjackass\b",                   re.I), "rude person"),

    # --- Single-word swaps (drop-in, same grammar slot) ---
    (re.compile(r"\bfuck\b",       re.I), "darn"),
    (re.compile(r"\bshit\b",       re.I), "stuff"),
    (re.compile(r"\bbitch\b",      re.I), "rude person"),
    (re.compile(r"\bdamn\b",       re.I), "darn"),
    (re.compile(r"\bhell\b",       re.I), "heck"),
    (re.compile(r"\bcrap\b",       re.I), "stuff"),
    (re.compile(r"\bass\b",        re.I), "butt"),
    (re.compile(r"\bidiot\b",      re.I), "fool"),
    (re.compile(r"\bidiots\b",     re.I), "fools"),
    (re.compile(r"\bstupid\b",     re.I), "unwise"),
    (re.compile(r"\bdumb\b",       re.I), "silly"),
    (re.compile(r"\bmoron\b",      re.I), "fool"),
    (re.compile(r"\bimbecile\b",   re.I), "fool"),
    (re.compile(r"\bretarded\b",   re.I), "misguided"),
    (re.compile(r"\bretard\b",     re.I), "person"),
    (re.compile(r"\bloser\b",      re.I), "person"),
    (re.compile(r"\blosers\b",     re.I), "people"),
    (re.compile(r"\bugly\b",       re.I), "unappealing"),
    (re.compile(r"\bdisgust(ing)?\b", re.I), "unpleasant"),
    (re.compile(r"\bpathetic\b",   re.I), "unfortunate"),
    (re.compile(r"\bhorrible\b",   re.I), "poor"),
    (re.compile(r"\bterrible\b",   re.I), "poor"),
    (re.compile(r"\bawful\b",      re.I), "poor"),
    (re.compile(r"\btrash\b",      re.I), "low-quality"),
    (re.compile(r"\bgarbage\b",    re.I), "low-quality"),
    (re.compile(r"\bsucks\b",      re.I), "is lacking"),
    (re.compile(r"\bsuck\b",       re.I), "lack"),
    (re.compile(r"\bscrew\s+you\b",re.I), "forget you"),
    (re.compile(r"\bpiss\s+off\b", re.I), "go away"),
    (re.compile(r"\bpissed\b",     re.I), "upset"),
    (re.compile(r"\bpiss\b",       re.I), "annoyance"),

    # --- Additional slang / slurs ---
    (re.compile(r"\bdick\s*head\b",  re.I), "rude person"),
    (re.compile(r"\bdick\b",         re.I), "jerk"),
    (re.compile(r"\bdicks\b",        re.I), "jerks"),
    (re.compile(r"\bcock\b",         re.I), "fool"),
    (re.compile(r"\bbastard\b",      re.I), "rude person"),
    (re.compile(r"\bbastards\b",     re.I), "rude people"),
    (re.compile(r"\bwhore\b",        re.I), "person"),
    (re.compile(r"\bslut\b",         re.I), "person"),
    (re.compile(r"\bsluts\b",        re.I), "people"),
    (re.compile(r"\bpervert\b",      re.I), "creep"),
    (re.compile(r"\bnigger\b",       re.I), "person"),
    (re.compile(r"\bnigga\b",        re.I), "person"),
    (re.compile(r"\bfaggot\b",       re.I), "person"),
    (re.compile(r"\bfag\b",          re.I), "person"),
    (re.compile(r"\bcunt\b",         re.I), "jerk"),
    (re.compile(r"\bpussy\b",        re.I), "coward"),
    (re.compile(r"\bpussies\b",      re.I), "cowards"),
    (re.compile(r"\bwanker\b",       re.I), "fool"),
    (re.compile(r"\btwat\b",         re.I), "fool"),
    (re.compile(r"\bprick\b",        re.I), "jerk"),
    (re.compile(r"\bfreak\b",        re.I), "odd person"),
    (re.compile(r"\bcreep\b",        re.I), "odd person"),
    (re.compile(r"\bscum\b",         re.I), "bad person"),
    (re.compile(r"\bscumbag\b",      re.I), "bad person"),
    (re.compile(r"\bnasty\b",        re.I), "unpleasant"),
    (re.compile(r"\bfilthy\b",       re.I), "messy"),
    (re.compile(r"\bdegenerate\b",   re.I), "misguided person"),
    (re.compile(r"\bmoronic\b",      re.I), "foolish"),
    (re.compile(r"\bcrappy\b",       re.I), "poor"),
    (re.compile(r"\bshitty\b",       re.I), "poor"),
    (re.compile(r"\bbitchy\b",       re.I), "rude"),
    (re.compile(r"\bworst\b",        re.I), "least good"),
]


class RewriteAgent:
    """Hybrid rewriter: LLM-first with rule-based fallback."""

    def __init__(self, event_bus: "EventBus", state: "GlobalState") -> None:
        self.bus = event_bus
        self.state = state
        self._running = False
        self._llm = LLMRewriter()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        self.bus.subscribe("toxicity_result", self._handle_toxicity)
        health_checks().mark_ready("RewriteAgent")
        logger.info("rewrite_agent_started", subscribed_to="toxicity_result")

        while self._running:
            await asyncio.sleep(0.1)

    def stop(self) -> None:
        self._running = False
        health_checks().mark_starting("RewriteAgent")

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    async def _handle_toxicity(self, event: Event) -> None:
        text: str = event.data.get("text", "")
        toxic: bool = event.data.get("toxic", False)
        speaker: str = event.data.get("speaker", "default")
        strictness: float = event.data.get("strictness", 1.0)
        rewrite_method = "passthrough"
        m = metrics()
        
        start_time = time.time()

        try:
            if toxic:
                # Try LLM first (in executor — it's synchronous HTTP)
                loop = asyncio.get_running_loop()
                
                with m.rewrite_latency_seconds.time():
                    llm_result = await loop.run_in_executor(None, self._llm.rewrite, text)

                if llm_result:
                    rewritten = llm_result
                    rewrite_method = "llm"
                    m.rewrite_attempts_total.labels(status="success").inc()
                    logger.info(
                        "rewrite_llm_success",
                        correlation_id=event.correlation_id,
                        original_length=len(text),
                        rewritten_length=len(rewritten),
                    )
                else:
                    rewritten = self._rewrite(text, strictness)
                    rewrite_method = "rules"
                    m.rewrite_fallback_total.labels(reason="llm_timeout").inc()
                    logger.info(
                        "rewrite_fallback_to_rules",
                        correlation_id=event.correlation_id,
                        reason="llm_timeout",
                        original_length=len(text),
                    )
            else:
                rewritten = text
                logger.debug(
                    "rewrite_passthrough",
                    correlation_id=event.correlation_id,
                    text_length=len(text),
                )

            elapsed = time.time() - start_time
            logger.info(
                "rewrite_complete",
                correlation_id=event.correlation_id,
                method=rewrite_method,
                duration_ms=int(elapsed * 1000),
            )

            await self.bus.emit(
                Event(
                    "rewrite_result",
                    {
                        "original_text": text,
                        "rewritten_text": rewritten,
                        "was_toxic": toxic,
                        "rewrite_method": rewrite_method,
                        "speaker": speaker,
                        "timestamp": time.time(),
                    },
                    correlation_id=event.correlation_id,
                )
            )
        except asyncio.TimeoutError:
            logger.error(
                "rewrite_timeout",
                correlation_id=event.correlation_id,
                error="Rewrite timed out",
                text_length=len(text),
            )
            m.rewrite_attempts_total.labels(status="timeout").inc()
            health_checks().mark_error("RewriteAgent", "Rewrite timeout")
        except Exception as e:
            logger.error(
                "rewrite_error",
                correlation_id=event.correlation_id,
                error=str(e),
                error_type=type(e).__name__,
                text_length=len(text),
            )
            m.rewrite_attempts_total.labels(status="error").inc()
            health_checks().mark_error("RewriteAgent", f"Rewrite error: {str(e)}")

    # ------------------------------------------------------------------
    # Rule-based rewriting
    # ------------------------------------------------------------------

    @staticmethod
    def _rewrite(text: str, strictness: float = 1.0) -> str:
        """
        Apply word-level substitutions that preserve sentence continuity.

        At high strictness (>= 2.0, repeat offender) we still apply the
        same word-level rules — continuity is always maintained.
        """
        result = text
        for pattern, replacement in _REWRITE_RULES:
            result = pattern.sub(replacement, result)

        # Clean up any double spaces introduced by substitutions
        result = re.sub(r"  +", " ", result).strip()

        # Fix article mismatches introduced by substitution
        # "an fool" → "a fool", "a unwise" → "an unwise"
        _VOWELS = set("aeiouAEIOU")

        def _fix_article(m: re.Match) -> str:
            article = m.group(1)  # "a" or "an" (possibly capitalised)
            word = m.group(2)
            if word[0] in _VOWELS:
                correct = "an" if article.islower() else "An"
            else:
                correct = "a" if article.islower() else "A"
            return f"{correct} {word}"

        result = re.sub(r"\b(an?|An?)\s+(\w+)", _fix_article, result)
        return result
