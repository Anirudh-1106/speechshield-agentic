"""
RewriteAgent — Rewrites toxic speech into polite alternatives.

Receives toxicity_result events.
- If toxic: attempts a guarded LLM rewrite first (when the rewrite will be
  heard), falls back to deterministic rules.
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
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

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
    (re.compile(r"\bfuckin'?(?=\W|$)",            re.I), "very"),
    (re.compile(r"\bf\*+(?:ing|in)\b",            re.I), "very"),
    (re.compile(r"\bmotherfuck\w*",               re.I), "jerk"),
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

    # --- Catch-alls (run last, after the more natural specific swaps) ---
    (re.compile(r"\bfuck\w*",        re.I), "darn"),
    (re.compile(r"\bshit\w*",        re.I), "stuff"),
    (re.compile(r"\bbitch\w*",       re.I), "rude person"),
    # Whisper often self-censors swearing as "f***" / "****": drop those tokens.
    (re.compile(r"(?<!\S)\S*\*{2,}\S*"), ""),
]


class RewriteAgent:
    """Hybrid rewriter: LLM-first with guarded, rule-based fallback."""

    def __init__(
        self,
        event_bus: "EventBus",
        state: "GlobalState",
        *,
        verifier: Optional[Callable[[str], Awaitable[float]]] = None,
    ) -> None:
        self.bus = event_bus
        self.state = state
        self._running = False
        self._stopped = asyncio.Event()
        self._llm = LLMRewriter(state.settings.rewrite)
        # Re-score LLM output with the toxicity model before it is ever spoken.
        self._llm.verifier = verifier
        self._llm.verify_threshold = state.settings.toxicity.threshold

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._stopped.clear()
        self._running = True
        self.bus.subscribe("toxicity_result", self._handle_toxicity)
        health_checks().mark_ready("RewriteAgent")
        provider = self.state.settings.rewrite.llm_provider
        logger.info("rewrite_agent_started", subscribed_to="toxicity_result", provider=provider)

        # Loading the LLM costs ~2 GB of RAM: skip it when no rewrite will be heard.
        warmup = asyncio.create_task(self._warmup() if self._llm_wanted("audio") else self._probe())
        try:
            await self._stopped.wait()
        finally:
            warmup.cancel()
            await self._llm.close()

    async def _warmup(self) -> None:
        try:
            await self._llm.warmup()
        except Exception as exc:
            logger.debug("llm_warmup_failed", error=str(exc))
        available = await self._llm.is_available()
        self.state.system_info["rewrite"] = (
            f"{self._llm.provider}:{self._llm.active_model}" if available else "rules (LLM unavailable)"
        )

    async def _probe(self) -> None:
        """Check the LLM is reachable (for API requests) without loading the model."""
        available = await self._llm.is_available()
        self.state.system_info["rewrite"] = f"rules (LLM {self._llm.active_model} on demand)" if available else "rules"

    def stop(self) -> None:
        self._running = False
        self._stopped.set()
        health_checks().mark_starting("RewriteAgent")

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    def _llm_wanted(self, source: str) -> bool:
        """Only pay LLM latency when the rewrite will actually be heard or returned."""
        if self.state.settings.rewrite.llm_provider == "local":
            return False
        icfg = self.state.settings.interceptor
        if source == "api" or not icfg.enabled:
            return True
        return icfg.censor_mode == "tts"

    async def _handle_toxicity(self, event: Event) -> None:
        text: str = event.data.get("text", "")
        toxic: bool = event.data.get("toxic", False)
        speaker: str = event.data.get("speaker", "default")
        strictness: float = event.data.get("strictness", 1.0)
        source: str = event.data.get("source", "audio")
        cid = event.correlation_id
        rewrite_method = "passthrough"
        llm_reason = ""
        m = metrics()
        start_time = time.time()
        context = self.state.get_history(speaker)

        if toxic:
            rewritten = ""
            if self._llm_wanted(source):
                with m.rewrite_latency_seconds.time():
                    outcome = await self._llm.rewrite(text, context=context, strictness=strictness)
                llm_reason = outcome.reason
                if outcome.text:
                    rewritten, rewrite_method = outcome.text, "llm"
                    m.rewrite_attempts_total.labels(status="success").inc()
                else:
                    if outcome.reason.startswith("guardrail:"):
                        m.rewrite_guardrail_rejections_total.labels(reason=outcome.reason.split(":", 1)[1]).inc()
                    elif outcome.reason in ("timeout", "error"):
                        m.rewrite_attempts_total.labels(status=outcome.reason).inc()
                    m.rewrite_fallback_total.labels(reason=outcome.reason.split(":", 1)[0]).inc()
            if not rewritten:
                rewritten, rewrite_method = self._rewrite(text, strictness), "rules"
            logger.info(
                "rewrite_complete",
                correlation_id=cid,
                method=rewrite_method,
                llm_reason=llm_reason or None,
                duration_ms=int((time.time() - start_time) * 1000),
            )
        else:
            rewritten = text
            logger.debug("rewrite_passthrough", correlation_id=cid, text_length=len(text))

        self.state.add_history(speaker, text)
        trace = self.state.trace(cid)
        trace.rewritten, trace.rewrite_method = rewritten, rewrite_method
        stage_ms = self.state.mark_stage(cid, "rewrite")
        m.stage_latency_seconds.labels(stage="rewrite").observe(stage_ms / 1000.0)

        await self.bus.emit(Event("rewrite_result", {
            "original_text": text,
            "rewritten_text": rewritten,
            "was_toxic": toxic,
            "rewrite_method": rewrite_method,
            "llm_reason": llm_reason,
            "speaker": speaker,
            "source": source,
            "timestamp": time.time(),
        }, correlation_id=cid))

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
