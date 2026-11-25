"""Shared, model-independent translation text checks."""
from __future__ import annotations

import re


def useful_context(text: str) -> bool:
    """Check if text contains meaningful content worth keeping in context."""
    compact = re.sub(r"[\s、。！？!?…ー〜～]+", "", text)
    return bool(compact) and compact not in {
        "はい", "うん", "ううん", "え", "ええ", "あ", "ああ", "お", "おお", "そう", "そうそう",
    }


def missing_names(english: str, names: list[str]) -> list[str]:
    """Find names that are missing in the translation.

    Args:
        english: the translated English text.
        names: English name strings to check for (e.g. ["Miko", "Subaru"]).
    """
    missing = []
    for name in names:
        tokens = re.split(r"[- ]+", name)
        pattern = r"(?<!\w)" + r"[\s\-‐‑–—]+".join(re.escape(token) for token in tokens) + r"(?!\w)"
        if not re.search(pattern, english, re.IGNORECASE):
            missing.append(name)
    return missing
