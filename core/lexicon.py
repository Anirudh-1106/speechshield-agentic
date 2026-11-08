"""
Lexicon — word lists and matching helpers shared by detection and censoring.

Two tiers:

  • STRONG terms (slurs, hard profanity) are toxic on their own.  The
    toxicity agent uses them as a safety net when the model under-scores.
  • MILD terms (insults, soft swearing) are only censored inside an
    utterance that is already judged toxic — "this game is trash" should
    pass, "you are trash, idiot" should be bleeped.

Words such as "kill" or "die" are deliberately *not* listed: in gaming
and sports chat they are overwhelmingly benign ("I killed the boss").
Real threats ("kill yourself", "I'll kill you") are covered as phrases.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# Prefix stems — any token starting with these is strong profanity.
_STRONG_STEMS = (
    "fuck", "motherfuck", "shit", "bitch", "cunt", "nigg", "fag", "whore",
    "slut", "asshole", "arsehole", "cocksuck", "dickhead", "bullshit", "retard",
)
_STRONG_WORDS = {
    "dick", "dicks", "cock", "cocks", "pussy", "pussies", "bastard", "bastards",
    "twat", "twats", "wanker", "wankers", "stfu", "gtfo", "kys", "wtf",
}
_MILD_WORDS = {
    "ass", "arse", "damn", "dammit", "goddamn", "crap", "crappy", "piss", "pissed",
    "hell", "idiot", "idiots", "stupid", "moron", "morons", "moronic", "dumb",
    "dumbass", "loser", "losers", "jackass", "prick", "pricks", "scum", "scumbag",
    "imbecile", "pathetic", "worthless", "trash", "garbage", "ugly", "freak",
    "creep", "pervert", "degenerate", "tosser", "sucks", "suck",
}
# Multi-word threats / insults, as token tuples.  Strong: toxic on their own.
_STRONG_PHRASES = (
    ("kill", "yourself"), ("go", "kill", "yourself"), ("kill", "you"), ("i'll", "kill", "you"),
    ("hope", "you", "die"), ("go", "die"), ("neck", "yourself"),
)
_MILD_PHRASES = (
    ("shut", "up"), ("screw", "you"), ("piss", "off"), ("go", "to", "hell"),
    ("hate", "you"), ("son", "of", "a", "bitch"), ("piece", "of", "shit"),
)

_TOKEN_RE = re.compile(r"[a-z0-9*']+")


def normalize_token(token: str) -> str:
    """Lower-case and strip punctuation so Whisper's ' Idiot,' becomes 'idiot'."""
    m = _TOKEN_RE.findall(token.lower())
    return "".join(m).strip("'")


def is_strong(token: str) -> bool:
    t = normalize_token(token)
    if not t:
        return False
    # Whisper frequently self-censors swearing as "f***" or "****".
    if "*" in t:
        return True
    return t in _STRONG_WORDS or t.startswith(_STRONG_STEMS)


def is_mild(token: str) -> bool:
    return normalize_token(token) in _MILD_WORDS


def tokenize(text: str) -> list[str]:
    return [t for t in (normalize_token(w) for w in text.split()) if t]


def has_strong_profanity(text: str) -> bool:
    """True if *text* contains a strong term or threat phrase."""
    tokens = tokenize(text)
    if any(is_strong(t) for t in tokens):
        return True
    return bool(_phrase_hits(tokens, _STRONG_PHRASES))


def _phrase_hits(tokens: list[str], phrases: tuple[tuple[str, ...], ...]) -> list[tuple[int, int]]:
    """(start, end_exclusive) token index ranges matching any phrase."""
    hits = []
    for phrase in phrases:
        n = len(phrase)
        for i in range(len(tokens) - n + 1):
            if tuple(tokens[i:i + n]) == phrase:
                hits.append((i, i + n))
    return hits


def toxic_token_indices(tokens: list[str], *, include_mild: bool) -> set[int]:
    """Indices of tokens that should be censored."""
    idx: set[int] = set()
    for i, t in enumerate(tokens):
        if is_strong(t) or (include_mild and is_mild(t)):
            idx.add(i)
    phrases = _STRONG_PHRASES + (_MILD_PHRASES if include_mild else ())
    for start, end in _phrase_hits(tokens, phrases):
        idx.update(range(start, end))
    return idx


@dataclass
class Span:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "text": self.text}


def toxic_word_spans(words: list[dict], *, utterance_toxic: bool) -> list[Span]:
    """Time spans of words to censor, from Whisper word timestamps.

    *words* are dicts with ``word``, ``start``, ``end`` (seconds from the
    start of the utterance audio).  Mild terms are included only when the
    utterance as a whole was judged toxic.
    """
    tokens = [normalize_token(w.get("word", "")) for w in words]
    hits = toxic_token_indices(tokens, include_mild=utterance_toxic)
    spans = [
        Span(float(words[i]["start"]), float(words[i]["end"]), str(words[i].get("word", "")).strip())
        for i in sorted(hits)
    ]
    return merge_spans(spans)


def first_mild_word_start(words: list[dict]) -> float:
    """Start time of the first word that is censorable only in a toxic context (inf if none)."""
    tokens = [normalize_token(w.get("word", "")) for w in words]
    mild = toxic_token_indices(tokens, include_mild=True) - toxic_token_indices(tokens, include_mild=False)
    return float(words[min(mild)]["start"]) if mild else math.inf


def merge_spans(spans: list[Span], gap: float = 0.05) -> list[Span]:
    """Merge overlapping / nearly touching spans."""
    merged: list[Span] = []
    for s in sorted(spans, key=lambda s: s.start):
        if merged and s.start <= merged[-1].end + gap:
            last = merged[-1]
            last.end = max(last.end, s.end)
            last.text = f"{last.text} {s.text}".strip()
        else:
            merged.append(Span(s.start, s.end, s.text))
    return merged


def mask_text(text: str, *, include_mild: bool = True) -> str:
    """Replace censored words with asterisks, keeping first letter and punctuation.

    "You stupid idiot!" -> "You s***** i****!"
    """
    words = text.split()
    tokens = [normalize_token(w) for w in words]
    hits = toxic_token_indices(tokens, include_mild=include_mild)
    out = []
    for i, w in enumerate(words):
        if i in hits:
            out.append(w[0] + re.sub(r"[A-Za-z0-9']", "*", w[1:]))
        else:
            out.append(w)
    return " ".join(out)


# ---------------------------------------------------------------------------
# Word-level replacements (voice_clone censor mode)
# ---------------------------------------------------------------------------
# Spoken in the speaker's own cloned voice, so they must be short, neutral
# and fit the same grammatical slot.  None = drop the word entirely
# (intensifiers and slurs: "you f***ing idiot" -> "you goof").

_DROP = object()

_SOFT_WORDS: dict[str, object] = {
    # intensifiers / exclamations: cut out, the sentence reads fine without them
    "fucking": _DROP, "fuckin": _DROP, "freaking": _DROP, "bloody": _DROP, "goddamn": _DROP,
    "damn": "dang", "dammit": "dang it", "hell": "heck", "wtf": "what the heck",
    "fuck": "dang", "shit": "shoot", "crap": "shoot", "bullshit": "nonsense",
    # insults -> mild, still-meaningful nouns/adjectives
    "idiot": "goof", "idiots": "goofs", "moron": "goof", "morons": "goofs", "imbecile": "goof",
    "dumbass": "goof", "jackass": "goof", "stupid": "silly", "dumb": "silly", "moronic": "silly",
    "asshole": "meanie", "assholes": "meanies", "bitch": "meanie", "bastard": "meanie",
    "dick": "meanie", "dickhead": "meanie", "prick": "meanie", "cunt": "meanie", "twat": "goof",
    "wanker": "goof", "motherfucker": "meanie", "loser": "rookie", "losers": "rookies",
    "trash": "rough", "garbage": "rough", "pathetic": "unlucky", "worthless": "unlucky",
    "ass": "butt", "pissed": "annoyed", "sucks": "is rough", "suck": "struggle",
    "scum": "meanie", "scumbag": "meanie", "ugly": "odd", "stfu": "hold on", "gtfo": "go away",
    # slurs carry nothing worth keeping
    "retard": _DROP, "retarded": _DROP, "nigger": _DROP, "nigga": _DROP, "faggot": _DROP,
    "fag": _DROP, "whore": _DROP, "slut": _DROP, "pussy": _DROP,
}
_SOFT_PHRASES: dict[tuple[str, ...], str] = {
    ("shut", "up"): "hold on", ("shut", "the", "fuck", "up"): "hold on",
    ("fuck", "off"): "go away", ("piss", "off"): "go away", ("go", "to", "hell"): "go away",
    ("fuck", "you"): "come on", ("screw", "you"): "come on", ("what", "the", "fuck"): "what the heck",
    ("kill", "yourself"): "take a break", ("go", "kill", "yourself"): "take a break",
    ("go", "die"): "take a break", ("kys",): "take a break",
    ("piece", "of", "shit"): "piece of work", ("son", "of", "a", "bitch"): "son of a gun",
    ("hate", "you"): "disagree with you",
}

# Every distinct replacement, for pre-synthesising a speaker's voice bank.
SOFT_VOCABULARY: list[str] = sorted({v for v in _SOFT_WORDS.values() if isinstance(v, str)} | set(_SOFT_PHRASES.values()))


def soft_replacement(span_text: str) -> str | None:
    """Polite stand-in for a censored span.

    Returns the replacement text, ``""`` to cut the span out, or ``None``
    if any word has no safe stand-in (caller bleeps instead).
    """
    tokens = tokenize(span_text)
    out: list[str] = []
    matched = False  # at least one word came from the table
    i = 0
    while i < len(tokens):
        for n in (4, 3, 2, 1):
            phrase = tuple(tokens[i:i + n])
            if len(phrase) == n and phrase in _SOFT_PHRASES:
                out.append(_SOFT_PHRASES[phrase])
                matched = True
                i += n
                break
        else:
            tok = tokens[i]
            rep = _SOFT_WORDS.get(tok)
            if rep is None:
                if "*" in tok and tok.endswith(("ing", "in")):
                    rep, matched = _DROP, True  # "f***ing"
                elif is_strong(tok) or is_mild(tok):
                    return None
                else:
                    rep = tok  # a clean word caught inside a merged span
            else:
                matched = True
            if rep is not _DROP:
                out.append(str(rep))
            i += 1
    # Nothing recognised: never make the speaker's voice say unvetted words.
    return " ".join(out) if matched else None
