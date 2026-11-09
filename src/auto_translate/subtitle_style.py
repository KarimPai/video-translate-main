"""Conservative cleanup for concise English-only subtitles.

Whisper is not an instruction-following LLM. Remove only evident disfluencies,
annotation-only laughs, and repeated interjections. Never rewrite substantive
words, negate speech, hallucinate shorter paraphrases, or remove profanity.
"""
from __future__ import annotations

import re

# Isolated non-verbal subtitle output is not useful in the English overlay.
_NONVERBAL_ONLY = re.compile(
    r"(?i)^\s*(?:(?:ha){2,}|(?:he){3,}|lol+|lmao|"
    r"\[?(?:laugh(?:s|ing|ter)?|giggl(?:e|es|ing))\]?)"
    r"[\s,!?._-]*$"
)
# These tokens are disfluency, not substantive semantic content.
_FILLER = r"(?:uh+|um+|erm+|hmm+|you know|well)"
_START_FILLER = re.compile(
    rf"(?i)^\s*(?:(?:{_FILLER})\s*[,;:]\s*)+"
)
_START_LAUGH = re.compile(
    r"(?i)^\s*(?:(?:ha){2,}|(?:he){3,}|lol+)\b[!,. ]*\s*"
)
# Only collapse three or more *identical* small interjections.
# Never collapse 'no', 'not', proper names, or informative repeated nouns.
_REPEAT_INTERJECTION = re.compile(
    r"(?i)\b(yeah|yes|well|okay|ok|uh|um|ah|oh)\b"
    r"(?:(?:\s*[,!?.]?\s*)\1\b){2,}"
)
_ANNOTATION = re.compile(r"(?i)\s*(?:\[(?:laughter|laughs|giggles)\]|\((?:laughter|laughs|giggles)\))\s*")


def concise_english(
    text: str,
    *,
    remove_fillers: bool = True,
    collapse_repetitions: bool = True,
) -> str:
    """Return shorter equivalent text, or '' for pure filler/laughter.

    This is not semantic translation; it never substitutes content words.
    """
    value = " ".join(text.split())
    if not value:
        return ""
    if remove_fillers and _NONVERBAL_ONLY.fullmatch(value):
        return ""
    if remove_fillers:
        value = _ANNOTATION.sub(" ", value).strip()
        value = _START_LAUGH.sub("", value).strip()
    if collapse_repetitions:
        value = _REPEAT_INTERJECTION.sub(lambda match: match.group(1), value)
    if remove_fillers:
        for _ in range(3):
            cleaned = _START_FILLER.sub("", value).strip()
            if cleaned == value:
                break
            value = cleaned
        if re.fullmatch(r"(?i)(?:uh+|um+|erm+|hmm+|well)[.!? ]*", value):
            return ""
    value = " ".join(value.split()).strip()
    value = value.strip(" ,;:")
    if not value or re.fullmatch(r"(?i)(?:ha){2,}[.!? ]*", value):
        return ""
    if value[0].isalpha() and value[0].islower():
        value = value[0].upper() + value[1:]
    return value
