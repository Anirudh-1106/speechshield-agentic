"""
LLM-based rewriter — context-aware polite rewrites via a local or remote LLM.

Backends:
  • ollama  — local Ollama server (/api/chat). Falls back to any installed
              model if the configured one has not been pulled.
  • openai  — any OpenAI-compatible /chat/completions endpoint (OpenAI,
              LM Studio, vLLM, llama.cpp server, …).
  • local   — no LLM; the RewriteAgent uses its rule engine.

Safety:
  • Guardrails reject output that still contains toxic vocabulary, reads
    like a refusal/explanation, or wildly changes length.  Rejected output
    falls back to rules — the user never hears an unchecked rewrite.
  • A circuit breaker stops calling a failing backend for a cool-down
    period so a dead LLM never adds its timeout to every utterance.

Requires Ollama running locally for the default backend:
    1. Install Ollama: https://ollama.com/download
    2. Pull a model:   ollama pull qwen2.5:3b
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

import httpx

from core.lexicon import tokenize, toxic_token_indices
from core.logging import get_logger

if TYPE_CHECKING:
    from config.settings import RewriteSettings

logger = get_logger("agents.llm_rewriter")

SYSTEM_PROMPT = (
    "You are a real-time speech moderation filter. You receive one spoken sentence "
    "that was flagged as toxic and you rewrite it so it is polite and respectful.\n"
    "Rules:\n"
    "1. Preserve the speaker's meaning and intent (disagreement, frustration, a request) — "
    "only the hostility, profanity and insults go.\n"
    "2. Keep it the same person, tense and roughly the same length. One sentence, spoken style.\n"
    "3. Never add facts, apologies, advice or commentary that were not in the original.\n"
    "4. Output ONLY the rewritten sentence — no quotes, labels, notes or explanations.\n"
    "5. Threats or slurs with no salvageable meaning become a short neutral statement "
    "of the underlying feeling (e.g. \"I'm really angry right now.\")."
)

# Few-shot examples teach format and tone more reliably than rules alone.
FEW_SHOT: list[tuple[str, str]] = [
    ("You're a fucking idiot, you missed the easy shot.", "You missed the easy shot, come on."),
    ("Shut the fuck up and listen to me.", "Please be quiet and listen to me."),
    ("This stupid ass game is complete bullshit.", "This game is really frustrating."),
    ("I'm gonna kill you if you take my loot again.", "Please don't take my loot again, I'm really annoyed."),
    ("You're so dumb, why would you push alone?", "Why would you push alone? That wasn't a good move."),
    ("I'll beat your ass if you do that again.", "Please don't do that again, I'm really angry."),
]

_STRICTNESS_HINTS = {
    1: "",
    2: " The speaker has been repeatedly toxic: make the rewrite calm and neutral.",
    3: " The speaker is persistently abusive: reduce the sentence to its plainest, most neutral form.",
}

_PREAMBLE_RE = re.compile(
    r"^(?:sure[,!.]?\s*|here(?:'s| is)[^:]*:\s*|rewritten(?: sentence| version)?:\s*|polite(?: version)?:\s*|output:\s*)+",
    re.IGNORECASE,
)
_REFUSAL_RE = re.compile(
    r"\b(i can(?:no|')t|i cannot|i'm sorry, but|as an ai|i won't|i am unable|language model)\b",
    re.IGNORECASE,
)


@dataclass
class RewriteOutcome:
    """Result of one LLM rewrite attempt."""
    text: Optional[str]
    reason: str = "ok"          # ok | unavailable | circuit_open | timeout | error | guardrail:<why>
    latency_ms: float = 0.0
    model: str = ""


class LLMRewriter:
    """Context-aware rewriter powered by an LLM, with guardrails and a circuit breaker."""

    AVAILABLE_TTL = 300.0     # re-check a healthy backend this often
    UNAVAILABLE_TTL = 30.0    # retry a missing backend this often

    def __init__(
        self,
        settings: "RewriteSettings | None" = None,
        *,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        if settings is None:
            from config.settings import RewriteSettings
            settings = RewriteSettings()
        self.settings = settings
        self.provider = provider or settings.llm_provider
        default_url = settings.openai_base_url if self.provider == "openai" else settings.ollama_base_url
        # Accept legacy full endpoint URLs like http://host:11434/api/generate
        self.base_url = re.sub(r"/api/(generate|chat|tags)/?$", "", (base_url or default_url).rstrip("/"))
        self.model = model or settings.model
        self.active_model: str = self.model
        self.timeout = timeout or settings.timeout_seconds
        headers = {}
        if self.provider == "openai" and settings.openai_api_key:
            headers["Authorization"] = f"Bearer {settings.openai_api_key}"
        self._client = httpx.AsyncClient(timeout=self.timeout, headers=headers)

        # Optional second opinion on every rewrite: async text -> toxicity score.
        self.verifier: Optional[Callable[[str], Awaitable[float]]] = None
        self.verify_threshold = 0.5

        self._available: Optional[bool] = None
        self._checked_at = 0.0
        self._failures = 0
        self._open_until = 0.0

    # ------------------------------------------------------------------
    # Availability / circuit breaker
    # ------------------------------------------------------------------

    @property
    def circuit_open(self) -> bool:
        return time.monotonic() < self._open_until

    def _record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.settings.circuit_breaker_failures:
            self._open_until = time.monotonic() + self.settings.circuit_breaker_cooldown_seconds
            logger.warning("llm_circuit_opened", failures=self._failures, cooldown_s=self.settings.circuit_breaker_cooldown_seconds)
            self._failures = 0
            _set_circuit_gauge(1)

    def _record_success(self) -> None:
        self._failures = 0
        if self._open_until:
            self._open_until = 0.0
            _set_circuit_gauge(0)

    async def is_available(self) -> bool:
        """Check the backend is reachable and has a usable model (cached with TTL)."""
        if self.provider == "local":
            return False
        ttl = self.AVAILABLE_TTL if self._available else self.UNAVAILABLE_TTL
        if self._available is not None and time.monotonic() - self._checked_at < ttl:
            return self._available

        self._checked_at = time.monotonic()
        try:
            if self.provider == "ollama":
                resp = await self._client.get(f"{self.base_url}/api/tags", timeout=3.0)
                resp.raise_for_status()
                names = [m["name"] for m in resp.json().get("models", [])]
                self.active_model = self._pick_model(names)
                self._available = bool(self.active_model)
            else:
                resp = await self._client.get(f"{self.base_url}/models", timeout=5.0)
                self._available = resp.status_code == 200
        except Exception as exc:
            if self._available is not False:
                logger.info("llm_unavailable", provider=self.provider, url=self.base_url, error=type(exc).__name__)
            self._available = False
        return bool(self._available)

    def _pick_model(self, installed: list[str]) -> str:
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        for name in installed:
            if name in (self.model, wanted):
                logger.info("llm_model_selected", model=name)
                return name
        if installed:
            logger.warning(
                "llm_model_missing_using_fallback",
                wanted=self.model,
                using=installed[0],
                hint=f"ollama pull {self.model}",
            )
            return installed[0]
        logger.warning("llm_no_models_installed", hint=f"ollama pull {self.model}")
        return ""

    async def warmup(self) -> None:
        """Load the model into memory so the first real rewrite is fast."""
        if await self.is_available():
            t0 = time.time()
            await self.rewrite("this is so damn annoying")
            logger.info("llm_warmed_up", model=self.active_model, ms=int((time.time() - t0) * 1000))

    # ------------------------------------------------------------------
    # Rewrite
    # ------------------------------------------------------------------

    def build_messages(self, text: str, context: Optional[list[str]] = None, strictness: float = 1.0) -> list[dict]:
        level = 3 if strictness >= 2.5 else 2 if strictness >= 1.75 else 1
        messages = [{"role": "system", "content": SYSTEM_PROMPT + _STRICTNESS_HINTS[level]}]
        for src, dst in FEW_SHOT:
            messages.append({"role": "user", "content": src})
            messages.append({"role": "assistant", "content": dst})
        prompt = text
        if context:
            ctx = "\n".join(f"- {c}" for c in context[-self.settings.context_turns:])
            prompt = f"(Earlier in the conversation the speaker said:\n{ctx})\nRewrite: {text}"
        banned = _toxic_terms(text)
        if banned:
            prompt += f"\n(Do not use: {', '.join(banned)})"
        messages.append({"role": "user", "content": prompt})
        return messages

    async def rewrite(
        self,
        text: str,
        context: Optional[list[str]] = None,
        strictness: float = 1.0,
    ) -> RewriteOutcome:
        """Rewrite *text*; ``outcome.text`` is None when the caller should fall back."""
        if self.circuit_open:
            return RewriteOutcome(None, "circuit_open")
        if not await self.is_available():
            return RewriteOutcome(None, "unavailable")

        messages = self.build_messages(text, context, strictness)
        attempts = 1 + self.settings.max_retries
        start = time.time()
        last_reason = "error"
        for attempt in range(attempts):
            try:
                raw = await self._complete(messages)
            except httpx.TimeoutException:
                # A timeout already spent the whole budget — don't retry.
                self._record_failure()
                return RewriteOutcome(None, "timeout", (time.time() - start) * 1000, self.active_model)
            except Exception as exc:
                last_reason = "error"
                logger.warning("llm_request_failed", attempt=attempt + 1, error=f"{type(exc).__name__}: {exc}")
                continue

            cleaned, why = validate_rewrite(text, raw)
            self._record_success()  # the backend works, even if this answer is unusable
            if cleaned is not None and self.verifier is not None:
                # Vocabulary checks miss hostility in clean words ("I'll beat your head").
                if await self.verifier(cleaned) >= self.verify_threshold:
                    cleaned, why = None, "still_toxic_model"
            elapsed = (time.time() - start) * 1000
            if cleaned is not None:
                return RewriteOutcome(cleaned, "ok", elapsed, self.active_model)
            if why in ("still_toxic", "still_toxic_model") and attempt < attempts - 1:
                # One corrective turn naming the problem fixes most misses.
                problem = (f"That still contains: {', '.join(_toxic_terms(raw))}." if why == "still_toxic"
                           else "That is still hostile, insulting or threatening.")
                messages = messages + [
                    {"role": "assistant", "content": raw.strip()},
                    {"role": "user", "content": f"{problem} Rewrite this again so it is calm and "
                                                f"non-threatening, keeping only its intent: {text}"},
                ]
                last_reason = f"guardrail:{why}"
                continue
            return RewriteOutcome(None, f"guardrail:{why}", elapsed, self.active_model)
        if last_reason.startswith("guardrail:"):
            return RewriteOutcome(None, last_reason, (time.time() - start) * 1000, self.active_model)

        self._record_failure()
        return RewriteOutcome(None, last_reason, (time.time() - start) * 1000, self.active_model)

    async def _complete(self, messages: list[dict]) -> str:
        if self.provider == "ollama":
            resp = await self._client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.active_model,
                    "messages": messages,
                    "stream": False,
                    "keep_alive": self.settings.keep_alive,
                    "options": {"temperature": 0.2, "num_predict": 96},
                },
            )
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "")
        resp = await self._client.post(
            f"{self.base_url}/chat/completions",
            json={"model": self.active_model, "messages": messages, "temperature": 0.2, "max_tokens": 96},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def close(self) -> None:
        await self._client.aclose()


def _toxic_terms(text: str) -> list[str]:
    """Censorable words in *text*, in order, de-duplicated."""
    tokens = tokenize(text)
    return list(dict.fromkeys(tokens[i] for i in sorted(toxic_token_indices(tokens, include_mild=True))))


def validate_rewrite(original: str, raw: str) -> tuple[Optional[str], str]:
    """Clean LLM output and check it is safe to speak. Returns (text, reason)."""
    if not raw or not raw.strip():
        return None, "empty"
    text = raw.strip().splitlines()[0].strip()
    text = _PREAMBLE_RE.sub("", text).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'“”":
        text = text[1:-1].strip()
    text = text.strip("“”\"").strip()
    if not text:
        return None, "empty"
    if _REFUSAL_RE.search(text):
        return None, "refusal"
    if len(text) > max(len(original) * 3, len(original) + 80):
        return None, "too_long"
    if toxic_token_indices(tokenize(text), include_mild=True):
        return None, "still_toxic"
    return text, "ok"


def _set_circuit_gauge(value: int) -> None:
    try:
        from core.metrics import metrics
        metrics().llm_circuit_open.set(value)
    except Exception:
        pass
