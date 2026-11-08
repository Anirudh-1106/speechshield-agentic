"""
Tests for agents.rewrite_agent — rule-based rewriting logic.

Does NOT require any ML models or external services.
"""

import re
import pytest
from agents.rewrite_agent import RewriteAgent, _REWRITE_RULES


class TestRewriteRules:
    """Test that rule-based substitutions work correctly."""

    @pytest.mark.parametrize(
        "toxic_input,expected_fragment",
        [
            ("You are such a stupid person", "unwise"),
            ("That's fucking ridiculous", "very ridiculous"),
            ("Shut up and listen", "please be quiet and listen"),
            ("What the fuck is this", "what on earth is this"),
            ("Go to hell", "please go away"),
            ("He is an asshole", "rude person"),
            ("You are a dick", "jerk"),
            ("This is bullshit and you know it", "nonsense and you know it"),
            ("You piece of shit", "You poor-quality thing"),
            ("That's so damn annoying", "darn annoying"),
            ("Stop being a bitch about it", "Stop being a rude person about it"),
            ("You are an idiot", "You are a fool"),
            ("He's a moron", "He's a fool"),
            ("Such a loser", "Such a person"),
            ("Kill yourself", "please reconsider"),
            ("I hate you so much", "I disagree with you so much"),
        ],
    )
    def test_toxic_rewrite_contains_polite_word(self, toxic_input: str, expected_fragment: str) -> None:
        result = RewriteAgent._rewrite(toxic_input)
        assert expected_fragment.lower() in result.lower(), (
            f"Expected '{expected_fragment}' in rewrite of '{toxic_input}', got: '{result}'"
        )

    @pytest.mark.parametrize(
        "toxic_input",
        [
            "You are stupid and I hate this",
            "What the fuck is wrong with this damn code",
            "Shut the fuck up you moron",
        ],
    )
    def test_original_toxic_words_removed(self, toxic_input: str) -> None:
        """After rewriting, the original toxic word should be gone."""
        result = RewriteAgent._rewrite(toxic_input)
        # The exact toxic words should not appear in the output
        toxic_words = ["stupid", "fuck", "damn", "moron", "shut up"]
        for word in toxic_words:
            if word.lower() in toxic_input.lower():
                assert word.lower() not in result.lower(), (
                    f"Toxic word '{word}' still in output: '{result}'"
                )

    def test_passthrough_nontoxic(self) -> None:
        """Non-toxic text should pass through unchanged (at rule level)."""
        clean = "I would like to discuss this calmly"
        assert RewriteAgent._rewrite(clean) == clean

    def test_sentence_continuity_preserved(self) -> None:
        """Rewrite should preserve sentence structure (not inject full sentences)."""
        result = RewriteAgent._rewrite("This stupid thing is really awful to use")
        # Should still be one sentence, not split apart
        assert result.count(".") <= 1  # At most one period at the end
        # Should contain connecting words from original
        assert "really" in result.lower()
        assert "to use" in result.lower()

    def test_double_spaces_cleaned(self) -> None:
        """No double spaces in output."""
        result = RewriteAgent._rewrite("You  are  a  stupid  idiot")
        assert "  " not in result

    @pytest.mark.parametrize(
        "toxic_input,forbidden",
        [
            ("you f***ing idiot", "*"),
            ("this is **** man", "*"),
            ("shut up you motherfucker", "motherfuck"),
            ("you fuckers are fuckin trash", "fuck"),
            ("that is a shitload of work", "shit"),
        ],
    )
    def test_variants_and_censored_tokens(self, toxic_input: str, forbidden: str) -> None:
        result = RewriteAgent._rewrite(toxic_input)
        assert forbidden not in result.lower(), result
        assert "  " not in result

    def test_multiple_toxic_words_replaced(self) -> None:
        """Multiple toxic words in one sentence all get replaced."""
        result = RewriteAgent._rewrite("You're a stupid ugly moron")
        assert "stupid" not in result.lower()
        assert "ugly" not in result.lower()
        assert "moron" not in result.lower()


class TestRewriteRuleCoverage:
    """Verify all rules in the table are well-formed."""

    def test_all_rules_are_compiled_patterns(self) -> None:
        for pattern, replacement in _REWRITE_RULES:
            assert isinstance(pattern, re.Pattern), f"Rule not compiled: {pattern}"
            assert isinstance(replacement, str), f"Replacement not a string: {replacement}"

    def test_no_empty_replacements(self) -> None:
        """Only the rule that drops Whisper's self-censored tokens ("f***") may delete text."""
        empty = [p.pattern for p, replacement in _REWRITE_RULES if not replacement]
        assert len(empty) == 1 and r"\*" in empty[0], empty

    def test_rules_count_reasonable(self) -> None:
        """We should have a substantial rule set."""
        assert len(_REWRITE_RULES) >= 30
