"""Conservative text-only shortening for direct VoicePing AST outputs.

Remove filler-only cues and exact adjacent duplicate sentences. Never
paraphrase content words or delete negation, profanity or names.
"""

from __future__ import annotations

import re

from .subtitle_style import concise_english

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

    Prior checks only recognized 2–6-word spans, missing Qwen repetition of
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



def concise_voiceping(
    text: str,
    *,
    remove_fillers: bool = True,
    collapse_repetitions: bool = True,
) -> str:
    """Discard nonverbal filler and repeated *identical* sentences only."""
    value = concise_english(
        text,
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
