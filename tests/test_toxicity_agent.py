"""
Tests for agents.toxicity_agent — safe-phrase whitelist and classification.

Only tests the whitelist logic (no model loading required for unit tests).
Model-dependent tests are marked as slow/integration.
"""

import pytest
from agents.toxicity_agent import _is_safe, _SAFE_PHRASES


class TestSafePhraseWhitelist:
    """Tests for the safe-phrase bypass logic."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "Thank you",
            "thank you",
            "THANK YOU",
            "Thanks",
            "Please",
            "Sorry",
            "Hello",
            "Hi",
            "Goodbye",
            "Good morning",
            "Yes",
            "No",
            "Okay",
            "Great job",
            "Well done",
        ],
    )
    def test_safe_phrases_detected(self, phrase: str) -> None:
        assert _is_safe(phrase) is True, f"'{phrase}' should be safe"

    @pytest.mark.parametrize(
        "phrase",
        [
            "Thank you.",
            "thank you!",
            "Thanks!",
            "Please,",
            "Sorry?",
        ],
    )
    def test_safe_phrases_with_punctuation(self, phrase: str) -> None:
        assert _is_safe(phrase) is True, f"'{phrase}' with punctuation should be safe"

    @pytest.mark.parametrize(
        "phrase",
        [
            "Thank you for your help",
            "Good morning everyone",
            "Hello there how are you",
        ],
    )
    def test_safe_prefix_match(self, phrase: str) -> None:
        assert _is_safe(phrase) is True, f"'{phrase}' starts with safe prefix"

    @pytest.mark.parametrize(
        "phrase",
        [
            "You are a terrible person",
            "I will kill you",
            "This is complete shit",
            "Shut up and go away",
        ],
    )
    def test_toxic_phrases_not_safe(self, phrase: str) -> None:
        assert _is_safe(phrase) is False, f"'{phrase}' should NOT be safe"

    def test_safe_phrases_set_not_empty(self) -> None:
        assert len(_SAFE_PHRASES) >= 20

    def test_empty_string_not_safe(self) -> None:
        # Empty string doesn't match any safe phrase
        assert _is_safe("") is False
