"""
LLM-based rewriter — uses a local Ollama model for context-aware rewrites.

Falls back to rule-based rewriting if Ollama is unavailable or too slow.

Requires Ollama running locally:
    1. Install Ollama: https://ollama.com/download
    2. Pull a model:   ollama pull phi3:mini
    3. Ollama serves on http://localhost:11434 by default
"""

import json
import logging
import time
from typing import Optional

import httpx

logger = logging.getLogger("LLMRewriter")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:1b"  # Small, fast, good at instruction-following
TIMEOUT_SECONDS = 5.0  # Hard cap — fall back to rules if exceeded

SYSTEM_PROMPT = (
    "You are a speech moderation assistant. "
    "Your ONLY job is to rewrite the given toxic/offensive sentence into a polite, "
    "respectful version that preserves the original meaning and intent. "
    "Rules:\n"
    "1. Keep the same sentence structure — do NOT split into multiple sentences.\n"
    "2. Replace offensive words with neutral/polite equivalents.\n"
    "3. Preserve the speaker's intent (disagreement, frustration, etc.).\n"
    "4. Output ONLY the rewritten sentence — no explanations, no quotes, no labels.\n"
    "5. Keep it concise — similar length to the original.\n"
    "6. Do NOT add content that wasn't in the original."
)


class LLMRewriter:
    """Context-aware rewriter powered by a local Ollama LLM."""

    def __init__(self, model: str = MODEL_NAME, base_url: str = OLLAMA_URL) -> None:
        self._model = model
        self._url = base_url
        self._available: Optional[bool] = None  # Lazy-checked
        self._client = httpx.Client(timeout=TIMEOUT_SECONDS)

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is pulled."""
        if self._available is not None:
            return self._available

        try:
            # Ping Ollama's tag list
            resp = self._client.get(
                self._url.replace("/api/generate", "/api/tags"),
                timeout=3.0,
            )
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                # Match model name (with or without :latest tag)
                self._available = any(
                    self._model in m or self._model.split(":")[0] in m
                    for m in models
                )
                if self._available:
                    logger.info("Ollama available — model '%s' found.", self._model)
                else:
                    logger.warning(
                        "Ollama running but model '%s' not found. "
                        "Available: %s. Run: ollama pull %s",
                        self._model, models, self._model,
                    )
            else:
                self._available = False
        except Exception:
            self._available = False
            logger.info("Ollama not available — will use rule-based fallback.")

        return self._available

    # ------------------------------------------------------------------
    # Rewrite
    # ------------------------------------------------------------------

    def rewrite(self, text: str) -> Optional[str]:
        """
        Rewrite *text* using the LLM.

        Returns the rewritten string, or None if the LLM is unavailable
        or the request fails/times out.
        """
        if not self.is_available():
            return None

        prompt = f"Rewrite this sentence politely:\n{text}"

        try:
            start = time.time()
            resp = self._client.post(
                self._url,
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "system": SYSTEM_PROMPT,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Near-deterministic
                        "num_predict": 128,  # Cap output length
                    },
                },
                timeout=TIMEOUT_SECONDS,
            )
            elapsed = (time.time() - start) * 1000
            logger.debug("LLM rewrite took %.0f ms", elapsed)

            if resp.status_code != 200:
                logger.warning("Ollama returned %d", resp.status_code)
                return None

            data = resp.json()
            result = data.get("response", "").strip()

            # Basic sanity: reject empty or absurdly long outputs
            if not result or len(result) > len(text) * 3:
                logger.warning("LLM output rejected (empty or too long).")
                return None

            # Strip surrounding quotes if the model wrapped them
            if result.startswith('"') and result.endswith('"'):
                result = result[1:-1]

            return result

        except httpx.TimeoutException:
            logger.warning("LLM rewrite timed out (%.0f ms budget).", TIMEOUT_SECONDS * 1000)
            return None
        except Exception as exc:
            logger.warning("LLM rewrite failed: %s", exc)
            return None

    def close(self) -> None:
        self._client.close()
