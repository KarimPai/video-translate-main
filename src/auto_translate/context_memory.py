"""Bounded recent-context hints and conservative repetition repair for JP subtitles.

History comes from the current sequential video segment only. It is cleared by
the translation pipeline on a seek. No extra ASR/model request is made.
"""

import re
import unicodedata
from collections import Counter
from typing import Iterable

# ASR commonly varies the spelling of short Katakana game terms. Do not
# fuzzy-rewrite longer sentences, Kanji, Hiragana speech or named people.
_KATAKANA_TERM = re.compile(r"[ァ-ヶー]{4,12}\Z")
_SPLIT = re.compile(r"([、。，,.!！?？\s]+)")
_TRIM = " \r\n\t、。，,.!！?？「」『』（）()"
_EN_TERM = re.compile(r"[A-Za-z][A-Za-z '\-]{1,27}\Z")
_IGNORE_EN = {
    "yes", "no", "okay", "ok", "hello", "sorry", "thanks", "thank you",
    "wait", "what", "there", "right", "yeah", "oh", "huh", "hey",
}


def _word(text: str) -> str:
    return unicodedata.normalize("NFKC", text.strip(_TRIM))


def _one_edit(a: str, b: str) -> bool:
    """One substitution, deletion or insertion; bounded tiny strings only."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(left != right for left, right in zip(a, b)) == 1
    shorter, longer = sorted((a, b), key=len)
    i = j = errors = 0
    while i < len(shorter) and j < len(longer):
        if shorter[i] == longer[j]:
            i += 1
            j += 1
        else:
            errors += 1
            if errors > 1:
                return False
            j += 1
    return True


def _recent_words(sources: Iterable[str]) -> list[str]:
    words: list[str] = []
    for ja in sources:
        source = _word(ja)
        if _KATAKANA_TERM.fullmatch(source):
            words.append(source)
        # Also recognize short, punctuated repetitions in a prior subtitle.
        elif len(source) <= 36:
            bits = [_word(part) for part in _SPLIT.split(source)]
            counts = Counter(x for x in bits if _KATAKANA_TERM.fullmatch(x))
            words.extend(word for word, count in counts.items() if count >= 2)
    return words



def _repair_compact_triple(japanese: str) -> str:
    """Repair e.g. シャッターシャッタシャッター without punctuation."""
    def replace(match: re.Match[str]) -> str:
        word = match.group()
        for size in range(4, min(12, (len(word) - 3) // 2) + 1):
            repeated = word[:size]
            if (
                not _KATAKANA_TERM.fullmatch(repeated)
                or not word.endswith(repeated)
            ):
                continue
            middle = word[size:-size]
            if len(middle) >= 3 and _one_edit(middle, repeated):
                return repeated * 3
        return word

    return re.sub(r"[ァ-ヶー]{11,36}", replace, japanese)



def repair_near_repeated_word(
    japanese: str,
    history: list[tuple[str, str]],
) -> str:
    """Correct only one-edit Katakana repeats supported by nearby Japanese.

    Examples:
        シャッター -> シャッタ (next cue) : restore シャッター
        シャッター、シャタ、シャッター : keep uncertain two-edit middle
        シャッター、シャッタ、シャッター : restore middle

    No correction when a prior term is absent, similarity is ambiguous,
    current Japanese is a longer sentence, or a different word is equally
    plausible. This intentionally cannot repair arbitrary ASR hallucinations.
    """
    if not japanese:
        return japanese

    japanese = _repair_compact_triple(japanese)
    parts = _SPLIT.split(japanese)
    tokens = [(i, _word(piece)) for i, piece in enumerate(parts)
              if _KATAKANA_TERM.fullmatch(_word(piece))]
    # For in-line repetitions, two exact repeats within this cue are strong.
    in_line_counts = Counter(word for _, word in tokens)
    strong = {word for word, count in in_line_counts.items() if count >= 2}
    recent = _recent_words(ja for ja, _ in history[-2:])
    standalone = bool(_KATAKANA_TERM.fullmatch(_word(japanese)))
    # Only allow one preceding occurrence when the CURRENT cue is a complete
    # short Katakana word; otherwise require two exact occurrences in-line.
    eligible = set(recent) if standalone else strong
    if not eligible:
        return japanese

    for i, token in tokens:
        matches = [reference for reference in eligible
                   if reference != token and _one_edit(token, reference)]
        if len(matches) != 1:
            continue
        raw = parts[i]
        # Keep outside punctuation and whitespace exactly as recognized.
        begin = len(raw) - len(raw.lstrip(_TRIM))
        end = len(raw.rstrip(_TRIM))
        if end < begin:
            continue
        parts[i] = raw[:begin] + matches[0] + raw[end:]
    return "".join(parts)



def reuse_recent_short_term(
    japanese: str,
    history: list[tuple[str, str]],
) -> str | None:
    """Reuse an identical isolated Katakana term's recent one-word English.

    This prevents e.g. シャッター -> Shutter / shutter / shudder on adjacent
    cues. Never reuse entire dialogue, names inferred from other Japanese,
    or multiword model guesses. A bad first translation may still persist
    briefly, so this memory is deliberately limited to the last two cues.
    """
    source = _word(japanese)
    if not _KATAKANA_TERM.fullmatch(source):
        return None

    for previous_ja, previous_en in reversed(history[-2:]):
        if _word(previous_ja) != source:
            continue
        target = previous_en.strip(_TRIM).strip("'\"“”")
        if (
            not _EN_TERM.fullmatch(target)
            or len(target.split()) != 1
            or target.casefold() in _IGNORE_EN
        ):
            return None
        punctuation = ""
        if japanese.rstrip().endswith(("？", "?")):
            punctuation = "?"
        elif japanese.rstrip().endswith(("！", "!")):
            punctuation = "!"
        return target + punctuation
    return None


def split_completed_japanese_sentences(text: str, *, min_chars: int = 8, max_parts: int = 3) -> list[str]:
    """Split only at explicit terminal punctuation, never infer a speaker turn.

    Keep short reactions attached to their following clause. Avoid splitting
    inside quoted dialogue and require substantial text on both sides.
    Output keeps the original words and punctuation unchanged.
    """
    source = text.strip()
    if not source:
        return []
    boundaries = []
    beginning = 0
    closers = '」』”’\'"'
    terminals = "。！？"
    quote_stack: list[str] = []
    quote_pairs = {"「": "」", "『": "』", "“": "”", "‘": "’"}
    for i, char in enumerate(source):
        if char in quote_pairs:
            quote_stack.append(quote_pairs[char])
            continue
        if quote_stack and char == quote_stack[-1]:
            quote_stack.pop()
            continue
        if char == '"':
            if quote_stack and quote_stack[-1] == '"':
                quote_stack.pop()
            else:
                quote_stack.append('"')
            continue
        if quote_stack:
            continue
        if char not in terminals:
            continue
        following = source[i + 1:i + 2]
        if following and (following in closers or following in terminals):
            continue
        current = source[beginning:i + 1].strip()
        remainder = source[i + 1:].strip()
        if len(current) < min_chars or len(remainder) < min_chars:
            continue
        boundaries.append(i + 1)
        beginning = i + 1
        if len(boundaries) >= max_parts:
            return [source]  # do not generate excessive translation requests
    if not boundaries:
        return [source]
    result = []
    start = 0
    for boundary in boundaries:
        result.append(source[start:boundary].strip())
        start = boundary
    result.append(source[start:].strip())
    return result


def build_recent_context(
    history: list[tuple[str, str]],
    glossary,
    *,
    context_lines: int = 8,
    memory_lines: int = 24,
) -> str:
    """Short, non-authoritative memory beyond the normal PLaMO chat history."""
    recent = history[-max(0, memory_lines):] if memory_lines else []
    if not recent:
        return ""

    lines: list[str] = []
    found: list[str] = []
    seen: set[str] = set()
    # The static glossary is reliable even if past PLaMO output was not.
    for ja, _ in reversed(recent):
        for source, target in glossary.find(ja):
            if source in seen:
                continue
            seen.add(source)
            found.append(f"{source} = {target}")
            if len(found) >= 6:
                break
        if len(found) >= 6:
            break
    if found:
        lines.append("Recently mentioned glossary names/terms: " + "; ".join(found))

    # Do not repeat the model's prior English translations as new hints:
    # a mistaken name or pronoun must not gain authority on later turns.
    # Older JP provides scene continuity, without reinforcing mistranslated
    # English pronouns. Keep it short to minimize tokens and latency.
    older = recent[:-context_lines] if context_lines > 0 else recent
    older_ja = [ja.strip()[:64] for ja, _ in older[-3:] if ja.strip()]
    if older_ja:
        lines.append("Earlier Japanese for scene continuity (do not translate): " + " / ".join(older_ja)[:195])
    return "\n".join(lines)
