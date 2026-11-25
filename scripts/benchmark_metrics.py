"""Shared text metrics for ASR evaluation (not production inference)."""
from __future__ import annotations

import unicodedata


def normalize_text(text: str) -> str:
    """Remove spacing and punctuation; retain kana, kanji, and actual words."""
    value = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in value if not ch.isspace()
                   and not unicodedata.category(ch).startswith("P"))


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, letter in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (letter != other)))
        previous = current
    return previous[-1]
