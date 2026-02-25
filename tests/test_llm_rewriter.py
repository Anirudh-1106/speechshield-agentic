"""
Tests for the LLM rewriter fallback logic.

These tests verify the LLMRewriter gracefully handles an unavailable
Ollama server (which is the expected state in CI / test environments).
"""

import pytest
from agents.llm_rewriter import LLMRewriter


class TestLLMRewriterOffline:
    """Tests when Ollama is NOT running (expected in CI)."""

    def test_is_available_returns_false_when_offline(self) -> None:
        rewriter = LLMRewriter(base_url="http://localhost:99999/api/generate")
        assert rewriter.is_available() is False

    def test_rewrite_returns_none_when_offline(self) -> None:
        rewriter = LLMRewriter(base_url="http://localhost:99999/api/generate")
        result = rewriter.rewrite("You are an idiot")
        assert result is None

    def test_availability_cached(self) -> None:
        rewriter = LLMRewriter(base_url="http://localhost:99999/api/generate")
        rewriter.is_available()
        # Second call should use cached value
        assert rewriter._available is False
        assert rewriter.is_available() is False

    def test_close_does_not_raise(self) -> None:
        rewriter = LLMRewriter()
        rewriter.close()  # Should not raise
