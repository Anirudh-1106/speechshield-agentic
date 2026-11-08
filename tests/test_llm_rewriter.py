"""
Tests for the LLM rewriter: offline fallback, guardrails, corrective
retries, the circuit breaker, model selection and both backends.

HTTP is faked with httpx.MockTransport — no Ollama needed.
"""

import json

import httpx
import pytest

from agents.llm_rewriter import LLMRewriter, validate_rewrite
from config.settings import RewriteSettings


def make_rewriter(handler, **settings) -> LLMRewriter:
    cfg = RewriteSettings(**{"max_retries": 1, "circuit_breaker_failures": 2, **settings})
    r = LLMRewriter(cfg)
    r._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return r


def ollama(replies: list[str], models=("qwen2.5:3b",), calls: list | None = None):
    """Fake Ollama: /api/tags lists *models*, /api/chat pops *replies*."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": m} for m in models]})
        if request.url.path == "/api/chat":
            if calls is not None:
                calls.append(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": replies.pop(0)}})
        return httpx.Response(404)
    return handler


class TestLLMRewriterOffline:
    """Tests when the LLM is NOT running (expected in CI)."""

    async def test_is_available_returns_false_when_offline(self) -> None:
        rewriter = LLMRewriter(base_url="http://127.0.0.1:9/api/generate")
        assert await rewriter.is_available() is False
        await rewriter.close()

    async def test_rewrite_returns_none_when_offline(self) -> None:
        rewriter = LLMRewriter(base_url="http://127.0.0.1:9")
        outcome = await rewriter.rewrite("You are an idiot")
        assert outcome.text is None
        assert outcome.reason == "unavailable"
        await rewriter.close()

    async def test_availability_cached(self) -> None:
        rewriter = LLMRewriter(base_url="http://127.0.0.1:9")
        await rewriter.is_available()
        assert rewriter._available is False
        assert await rewriter.is_available() is False
        await rewriter.close()

    async def test_local_provider_never_calls_out(self) -> None:
        rewriter = LLMRewriter(RewriteSettings(llm_provider="local"))
        assert await rewriter.is_available() is False
        await rewriter.close()

    async def test_close_does_not_raise(self) -> None:
        rewriter = LLMRewriter()
        await rewriter.close()


class TestRewriteWithBackend:
    async def test_successful_rewrite(self) -> None:
        r = make_rewriter(ollama(["Please be quiet and listen."]))
        out = await r.rewrite("shut the fuck up and listen")
        assert out.text == "Please be quiet and listen."
        assert out.reason == "ok"
        assert out.model == "qwen2.5:3b"

    async def test_missing_model_falls_back_to_installed(self) -> None:
        r = make_rewriter(ollama(["Fine."], models=("llama3.2:1b",)), model="gemma3:1b")
        assert await r.is_available()
        assert r.active_model == "llama3.2:1b"

    async def test_no_models_installed_is_unavailable(self) -> None:
        r = make_rewriter(ollama([], models=()))
        assert await r.is_available() is False

    async def test_legacy_generate_url_accepted(self) -> None:
        r = LLMRewriter(base_url="http://localhost:11434/api/generate")
        assert r.base_url == "http://localhost:11434"
        await r.close()

    async def test_prompt_lists_banned_words_and_context(self) -> None:
        calls: list = []
        r = make_rewriter(ollama(["You missed again."], calls=calls))
        await r.rewrite("you stupid idiot, you missed", context=["push mid", "we need to group"])
        user_msg = calls[0]["messages"][-1]["content"]
        assert "stupid" in user_msg and "idiot" in user_msg  # banned-word hint
        assert "we need to group" in user_msg  # conversation context
        assert calls[0]["keep_alive"] == "30m"

    async def test_still_toxic_triggers_one_corrective_retry(self) -> None:
        calls: list = []
        r = make_rewriter(ollama(["You're kind of dumb.", "That was not a great move."], calls=calls))
        out = await r.rewrite("you're such a dumbass")
        assert out.text == "That was not a great move."
        assert len(calls) == 2
        assert "dumb" in calls[1]["messages"][-1]["content"]  # correction names the word

    async def test_guardrail_rejection_after_retries_returns_none(self) -> None:
        r = make_rewriter(ollama(["still an idiot", "still an idiot"]))
        out = await r.rewrite("you idiot")
        assert out.text is None
        assert out.reason == "guardrail:still_toxic"

    async def test_verifier_rejects_hostile_rewrite(self) -> None:
        r = make_rewriter(ollama(["I'll find you and beat your head.", "Please stop, I'm upset."]))

        async def verifier(text: str) -> float:
            return 0.95 if "beat" in text else 0.01

        r.verifier = verifier
        out = await r.rewrite("I'm gonna beat your ass")
        assert out.text == "Please stop, I'm upset."

    async def test_circuit_breaker_opens_after_failures(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/tags":
                return httpx.Response(200, json={"models": [{"name": "qwen2.5:3b"}]})
            return httpx.Response(500)

        r = make_rewriter(handler, max_retries=0, circuit_breaker_failures=2)
        assert (await r.rewrite("you idiot")).reason == "error"
        assert (await r.rewrite("you idiot")).reason == "error"
        assert r.circuit_open
        assert (await r.rewrite("you idiot")).reason == "circuit_open"

    async def test_timeout_is_not_retried(self) -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/tags":
                return httpx.Response(200, json={"models": [{"name": "qwen2.5:3b"}]})
            calls.append(1)
            raise httpx.ReadTimeout("slow", request=request)

        r = make_rewriter(handler, max_retries=3)
        assert (await r.rewrite("you idiot")).reason == "timeout"
        assert len(calls) == 1

    async def test_openai_compatible_backend(self) -> None:
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": []})
            seen["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json={"choices": [{"message": {"content": "Please stop that."}}]})

        cfg = RewriteSettings(llm_provider="openai", openai_base_url="http://llm.local/v1", openai_api_key="sk-test", model="gpt-x")
        r = LLMRewriter(cfg)
        r._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), headers={"Authorization": "Bearer sk-test"})
        out = await r.rewrite("stop that you idiot")
        assert out.text == "Please stop that."
        assert seen["auth"] == "Bearer sk-test"


class TestValidateRewrite:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ('"Please be quiet."', "Please be quiet."),
            ("Sure! Here's the rewrite: Please be quiet.", "Please be quiet."),
            ("Rewritten: Please be quiet.\nThis keeps the intent.", "Please be quiet."),
        ],
    )
    def test_cleans_output(self, raw: str, expected: str) -> None:
        assert validate_rewrite("shut up", raw) == (expected, "ok")

    @pytest.mark.parametrize(
        "raw,reason",
        [
            ("", "empty"),
            ("I'm sorry, but I can't help with that.", "refusal"),
            ("As an AI language model I cannot", "refusal"),
            ("You absolute idiot", "still_toxic"),
            ("word " * 200, "too_long"),
        ],
    )
    def test_rejects_unsafe_output(self, raw: str, reason: str) -> None:
        assert validate_rewrite("you idiot", raw) == (None, reason)
