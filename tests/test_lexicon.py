"""Tests for the shared toxic-vocabulary helpers."""

import pytest

from core.lexicon import has_strong_profanity, mask_text, toxic_word_spans


@pytest.mark.parametrize("text", [
    "you fucking idiot", "what a bitch", "f*** off", "kill yourself", "go die", "STFU noob",
])
def test_strong_profanity_detected(text: str) -> None:
    assert has_strong_profanity(text)


@pytest.mark.parametrize("text", [
    "I killed the boss", "my character will die", "that shot was sick", "this game is trash",
    "you're a stupid idiot",  # mild insults are left to the model
])
def test_benign_or_mild_not_strong(text: str) -> None:
    assert not has_strong_profanity(text)


def words(*items):
    return [{"word": w, "start": s, "end": e} for w, s, e in items]


def test_spans_from_word_timestamps() -> None:
    w = words((" You", 0.0, 0.2), (" stupid", 0.2, 0.5), (" fucking", 0.5, 0.8), (" idiot,", 0.8, 1.1), (" stop.", 1.2, 1.5))
    spans = toxic_word_spans(w, utterance_toxic=True)
    assert len(spans) == 1  # adjacent words merge into one bleep
    assert spans[0].start == 0.2 and spans[0].end == 1.1


def test_mild_words_only_censored_in_toxic_utterances() -> None:
    w = words((" this", 0.0, 0.2), (" game", 0.2, 0.4), (" is", 0.4, 0.5), (" trash", 0.5, 0.9))
    assert toxic_word_spans(w, utterance_toxic=False) == []
    assert len(toxic_word_spans(w, utterance_toxic=True)) == 1


def test_phrase_spans() -> None:
    w = words((" just", 0.0, 0.2), (" shut", 0.3, 0.5), (" up", 0.5, 0.7), (" please", 0.8, 1.0))
    spans = toxic_word_spans(w, utterance_toxic=True)
    assert [(s.start, s.end) for s in spans] == [(0.3, 0.7)]


def test_mask_text() -> None:
    assert mask_text("You stupid idiot!") == "You s***** i****!"
    assert mask_text("hello there") == "hello there"
