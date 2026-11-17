"""Conservative cleanup for concise English-only subtitles.

Preserve ASR/translation meaning while removing only evident disfluencies,
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


# Additional strict filler and repeated multi-sentence loop checks.
_FILLER_ONLY = re.compile(r"(?i)^(?:um+|uh+|erm+|hmm+)(?:[\s,;:.!?…]*)$")
_LEAD_FILLER = re.compile(
    r"(?i)^(?:(?:um+|uh+|erm+|hmm+)(?:\s*[,;:.!?…]+\s*|\s+))+"
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“‘])")
_WORD = re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)?", re.I)



def normalized_cue(text: str) -> str:
    """Stable exact-word identity for identifying repeated short cues."""
    return " ".join(_WORD.findall(text.casefold().replace("’", "'")))



def has_generation_loop(text: str) -> bool:
    """Detect repeated 6–48-word phrases, including multi-sentence loops.

    Earlier checks only recognized 2–6-word spans, missing repetition of
    complete sentences. Three consecutive identical phrases are suspicious.
    Short shouted repeats ("go, go, go") are intentionally excluded.
    """
    words = _WORD.findall(text.casefold().replace("’", "'"))
    for width in range(6, min(48, len(words) // 3) + 1):
        for start in range(len(words) - 3 * width + 1):
            phrase = words[start:start + width]
            if (words[start + width:start + 2 * width] == phrase
                    and words[start + 2 * width:start + 3 * width] == phrase):
                return True
    return False



def looks_like_prompt_echo(text: str) -> bool:
    """Only reject unmistakable prompt/template leakage, not normal dialogue."""
    phrase = " ".join(text.casefold().split())
    markers = (
        "recently mentioned glossary names/terms:",
        "recent tentative term translations (use only if",
        "earlier japanese for scene continuity",
        "chat_template_kwargs",
        "if the current japanese does not explicitly identify",
        "japanese often omits the subject or possessor",
        "for english-only speech, laughter or game sounds, output nothing",
        "translate japanese speech to english only",
    )
    return any(marker in phrase for marker in markers)


def unmatched_double_quote_cleanup(text: str) -> str:
    """Remove orphan double quotes while retaining balanced dialogue and apostrophes.

    Each cue is self-contained: an opener from a previous subtitle must never
    authorize a dangling closer in this one. No missing words/quotes are added.
    """
    pending: list[tuple[int, str]] = []
    paired: set[int] = set()
    for index, char in enumerate(text):
        if char in '“„‟':
            if char == '“' and pending and pending[-1][1] == '„':
                opening, _ = pending.pop()
                paired.update((opening, index))
            else:
                pending.append((index, char))
        elif char == '”':
            if pending:
                opening, _ = pending.pop()
                paired.update((opening, index))
        elif char == '"':
            if pending:
                opening, _ = pending.pop()
                paired.update((opening, index))
            else:
                pending.append((index, char))
    return ''.join(char for index, char in enumerate(text)
                   if char not in '"“”„‟' or index in paired).strip()


# The translator must return English; Latin-script names, honorifics, and cultural
# terms remain valid. Japanese script is not a romanized name or honorific.
_JAPANESE_OUTPUT = re.compile(
    r"[\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff"
    r"\uf900-\ufaff\uff66-\uff9f\U00020000-\U000323af]"
)


def english_output_issue(text: str) -> str | None:
    """Return a hard output failure, independent of proper-name validation."""
    if not text.strip():
        return "empty_english"
    if _JAPANESE_OUTPUT.search(text):
        return "untranslated_japanese"
    if has_generation_loop(text) or looks_like_prompt_echo(text):
        return "prompt_echo_or_loop"
    return None


def sanitize_english_output(text: str) -> str:
    """Shared final boundary for models, history, caches, and all renderers."""
    value = unmatched_double_quote_cleanup(text)
    return "" if english_output_issue(value) else value


def clean_english_output(
    text: str,
    *,
    remove_fillers: bool = True,
    collapse_repetitions: bool = True,
) -> str:
    """Discard nonverbal filler and repeated *identical* sentences only."""
    value = concise_english(
        sanitize_english_output(text),
        remove_fillers=remove_fillers,
        collapse_repetitions=collapse_repetitions,
    )
    if not value:
        return ""
    if remove_fillers:
        if _FILLER_ONLY.fullmatch(value):
            return ""
        value = _LEAD_FILLER.sub("", value).strip()
        if not value:
            return ""

    if collapse_repetitions:
        parts = _SENTENCE_BOUNDARY.split(value)
        if len(parts) > 1:
            kept: list[str] = []
            previous = ""
            for sentence in parts:
                key = normalized_cue(sentence)
                # Only drop exact neighboring sentences with >=4 words.
                # A repeated "No!" or "Run!" can be meaningful and stays.
                if key == previous and len(key.split()) >= 4:
                    continue
                kept.append(sentence)
                previous = key
            value = " ".join(kept)
    return value.strip()
