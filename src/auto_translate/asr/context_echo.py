"""Evidence for reviewing a hypothesis against the context used for its decode."""
import re
import unicodedata
from difflib import SequenceMatcher

import numpy as np


def speech_key(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def context_match(text: str, context: str) -> float:
    """A score triggers an independent decode, never a vocabulary blacklist."""
    key = speech_key(text)
    if not key or not context:
        return 0.0
    if len(key) >= 10 and key == speech_key(context):
        return 1.0
    parts = {speech_key(p) for p in re.split(r"[、,，;；・。\n]+", text)} - {""}
    for line in context.splitlines():
        label, sep, value = unicodedata.normalize("NFKC", line).partition(":")
        listed = sep and label in ("用語", "人名")
        value = value if listed else line
        hints = {speech_key(p) for p in re.split(r"[、,，;；・]+", value)} - {""}
        shared = len(hints & parts)
        if shared >= 3:
            coverage = shared / min(len(parts), len(hints))
            if coverage >= .75:
                return coverage
        reference = speech_key(value)
        if len(key) < 10 or not reference:
            continue
        # A single correct proper name is ordinary speech, even if it is long.
        if listed and (len(hints) < 2 or key in hints):
            continue
        if key in reference:
            return 1.0
        if .85 <= len(key) / len(reference) <= 1.18:
            similarity = SequenceMatcher(None, key, reference, autojunk=False).ratio()
            if similarity >= .90:
                return similarity
    return 0.0


def active_seconds(audio: np.ndarray) -> float:
    """Time above a low RMS floor, measured on original PCM before ASR gain."""
    if not len(audio):
        return 0.0
    starts = np.arange(0, len(audio), 160)
    counts = np.minimum(160, len(audio) - starts)
    mean = np.add.reduceat(audio, starts) / counts
    power = np.add.reduceat(np.square(audio), starts) / counts
    # A DC offset in quiet capture must not count as acoustic support.
    return float(counts[power - mean * mean >= .001 ** 2].sum()) / 16000
