"""Conservative cleanup for concise English-only subtitles.

Preserve ASR/translation meaning while removing only evident disfluencies,
annotation-only laughs, and repeated interjections. Never rewrite substantive
words, negate speech, hallucinate shorter paraphrases, or remove profanity.
"""
from __future__ import annotations

import re
import unicodedata
from contextlib import contextmanager
from contextvars import ContextVar

_ALLOW_JAPANESE = ContextVar("subtitle_allow_japanese", default=False)


@contextmanager
def subtitle_language_policy(allow_japanese: bool):
    """Use the session policy for model checks, cached subtitles and renderers."""
    token = _ALLOW_JAPANESE.set(allow_japanese)
    try:
        yield
    finally:
        _ALLOW_JAPANESE.reset(token)


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
_ENGLISH_REACTION = re.compile(r"(?i)^(?:hmm+|uh+|erm+|(?:ha){2,}|lol+|lmao)[.!?…]*$")
_THINKING_SOUND = re.compile(r"^(?:うーん+|ううーん+|うーむ+)$")
_SPOKEN_LAUGHTER = re.compile(r"^(?:(?:あ?は){2,}|あは+|わは+|ふふ+|へへ+|ほほ+|くく+)[っー〜～]*$")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“‘])")
_WORD = re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)?", re.IGNORECASE)



def normalized_cue(text: str) -> str:
    """Stable exact-word identity for identifying repeated short cues."""
    return " ".join(_WORD.findall(text.casefold().replace("’", "'")))



def has_generation_loop(text: str) -> bool:
    """Detect repeated 6–48-word phrases, including multi-sentence loops.

    Earlier checks only recognized 2–6-word spans, missing repetition of
    complete sentences. Three consecutive identical phrases are suspicious.
    Short shouted repeats ("go, go, go") are intentionally excluded.

    Uses tuple n-gram comparison for O(n^2) instead of O(n^3) list comparison.
    """
    words = _WORD.findall(text.casefold().replace("’", "'"))
    n = len(words)
    if n < 18:
        return False
    for width in range(6, min(48, n // 3) + 1):
        for start in range(n - 3 * width + 1):
            phrase = tuple(words[start:start + width])
            if (tuple(words[start + width:start + 2 * width]) == phrase
                    and tuple(words[start + 2 * width:start + 3 * width]) == phrase):
                return True
    return False



def looks_like_prompt_echo(text: str) -> bool:
    """Only reject unmistakable prompt/template leakage, not normal dialogue."""
    phrase = " ".join(text.casefold().split())
    markers = (
        # Original markers
        "recently mentioned glossary names/terms:",
        "recent tentative term translations (use only if",
        "earlier japanese for scene continuity",
        "chat_template_kwargs",
        "if the current japanese does not explicitly identify",
        "japanese often omits the subject or possessor",
        "for english-only speech, laughter or game sounds, output nothing",
        "translate japanese speech to english only",
        # System prompt echo markers (updated for current prompt)
        "you are a japanese-to-english subtitle translator",
        "preserve japanese names and nicknames exactly as spoken",
        "do not invent speakers, names, actions, or details",
        "do not add explanatory notes, definitions, or parenthetical",
        "do not output slash-separated alternatives",
        "when the japanese omits a subject, do not invent one",
        # Legacy prompt markers (still detect old versions)
        "you are a professional japanese-to-english subtitle translator",
        "output exactly one english translation per utterance",
        "never output slash-separated alternatives",
        "never add explanatory text, translation notes",
        "prioritize meaning and natural flow over literal",
        "preserve japanese names, nicknames, slang",
        # User prompt template markers
        "reference context (not part of the speech to translate)",
        "translate this japanese text:",
        "name/term glossary:",
        "translation style:",
        # Reasoning tag leakage markers
        "<think",
        "<thinking",
        "<reasoning",
        "</think",
        "</thinking",
        "</reasoning",
        # Glossary leakage markers
        "→",  # Arrow used in glossary format: "ja → en"
        # Instruction echo markers
        "output only the english translation, nothing else",
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

# Gemma 4 and other models may emit control tokens abnormally when generation
# degrades (e.g. upstream llama.cpp issue #21338).  Detect floods of repeated
# special tokens before they reach the subtitle renderer.
_SPECIAL_TOKENS = {
    "<unused49>", "<unused8>", "<fin>", "<start_of_turn>", "<end_of_turn>",
    "<eos>", "<bos>", "<pad>", "<mask>", "<extra_id_0>", "<extra_id_1>",
}
_SPECIAL_TOKEN_PATTERN = re.compile(
    r"|".join(re.escape(t) for t in _SPECIAL_TOKENS)
)
# A flood is 3+ occurrences of the same special token, or 2+ different special tokens
# that dominate the output (more than 50% of non-whitespace tokens).
_MIN_FLOOD_COUNT = 3
_DOMINANCE_THRESHOLD = 0.5


def has_special_token_flood(text: str) -> bool:
    """Detect abnormal repetition of model control tokens in generation output.

    Gemma 4 is known to flood output with repeated special tokens such as
    ``<unused49>`` when reasoning mode misbehaves (llama.cpp issue #21338).
    This check rejects such output before it reaches the subtitle renderer.

    Returns True if:
    - Any single special token appears 3+ times, OR
    - Special tokens dominate the output (>50% of non-whitespace tokens)
    """
    if not text or len(text) < 10:
        return False

    matches = _SPECIAL_TOKEN_PATTERN.findall(text)
    if len(matches) < 2:
        return False

    # Check for single-token flood
    from collections import Counter
    counts = Counter(matches)
    if counts.most_common(1)[0][1] >= _MIN_FLOOD_COUNT:
        return True

    # Check for dominance: special tokens > 50% of all non-whitespace tokens
    all_tokens = text.split()
    return bool(len(all_tokens) >= 4 and len(matches) / len(all_tokens) > _DOMINANCE_THRESHOLD)


def english_output_issue(text: str, *, allow_japanese: bool | None = None) -> str | None:
    """Return a hard output failure, independent of proper-name validation."""
    if not text.strip():
        return "empty_english"
    if allow_japanese is None:
        allow_japanese = _ALLOW_JAPANESE.get()
    if not allow_japanese and _JAPANESE_OUTPUT.search(text):
        return "untranslated_japanese"
    if has_generation_loop(text) or looks_like_prompt_echo(text):
        return "prompt_echo_or_loop"
    if has_special_token_flood(text):
        return "special_token_flood"
    # Slash-separated content is no longer rejected: ambiguous slashes like
    # "yes/no", "24/7", "input/output" are valid spoken content. Preserving
    # them is better than silently deleting meaning.
    return None


# Diagnostic-only pattern for slash-separated content.
# This is no longer used to reject output — ambiguous slashes like "yes/no",
# "24/7", "input/output" are valid spoken content. The function remains
# available for logging/diagnostics only.
_SLASH_ALT_PATTERN = re.compile(
    r'\b\w+\s*/\s*\w+\b'
)


def has_slash_alternatives(text: str) -> bool:
    """Detect slash-separated patterns in output (diagnostic only).

    Returns True if the text contains patterns like "Hello/Hi" or "yes/no".
    This is a diagnostic helper only — it no longer causes output rejection.
    Ambiguous slashes are valid content and should be preserved.
    """
    if not text or '/' not in text:
        return False

    matches = _SLASH_ALT_PATTERN.findall(text)
    return len(matches) > 0


def sanitize_english_output(text: str, *, allow_japanese: bool | None = None) -> str:
    """Shared final boundary for models, history, caches, and all renderers."""
    value = unmatched_double_quote_cleanup(text)
    return "" if english_output_issue(value, allow_japanese=allow_japanese) else value


def preserved_reaction(text: str, source_ja: str) -> str | None:
    """Keep only English thinking sounds or laughter grounded in Japanese speech."""
    value = " ".join(text.split())
    if not _ENGLISH_REACTION.fullmatch(value):
        return None
    source = unicodedata.normalize("NFKC", source_ja)
    source = "".join(chr(ord(char) - 0x60) if "ァ" <= char <= "ヶ" else char for char in source)
    compact = re.sub(r"[\s、。！？!?…,.]+", "", source)
    if re.fullmatch(r"(?i)hmm+[.!?…]*|uh+[.!?…]*|erm+[.!?…]*", value):
        if _THINKING_SOUND.fullmatch(compact):
            return "Uh..." if value.casefold().startswith(("uh", "erm")) else "Hmm..."
        return ""
    if re.fullmatch(r"(?i)(?:ha){2,}[.!?…]*|lol+[.!?…]*|lmao[.!?…]*", value):
        return "Haha." if _SPOKEN_LAUGHTER.fullmatch(compact) else ""
    return None


def clean_english_output(
    text: str,
    *,
    remove_fillers: bool = True,
    collapse_repetitions: bool = True,
    allow_japanese: bool = False,
    source_ja: str = "",
) -> str:
    """Discard nonverbal filler and repeated *identical* sentences only."""
    if remove_fillers and source_ja:
        reaction = preserved_reaction(text, source_ja)
        if reaction is not None:
            return reaction
    value = concise_english(
        sanitize_english_output(text, allow_japanese=allow_japanese),
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
