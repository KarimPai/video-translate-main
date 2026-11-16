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


def _overlap(left: str, right: str) -> int:
    """Cheap Japanese character-bigram overlap; never infer a missing word."""
    def grams(text):
        # Ignore punctuation and one-character particles as matching evidence.
        chunks = re.findall(r"[ぁ-ヺ一-鿿A-Za-z0-9]{2,}", _word(text))
        return {chunk[i:i + 2] for chunk in chunks for i in range(len(chunk) - 1)}
    return len(grams(left) & grams(right))


def select_dialogue_history(
    history: list[tuple[str, str]], current_ja: str, *, context_lines: int = 3,
    memory_lines: int = 16, max_chars: int = 640,
) -> list[tuple[str, str]]:
    """Keep adjacent turns and optionally retrieve one relevant older exchange.

    Selection only reads finalized preceding cues. Results preserve chronology
    and never contain partial examples; a large history cannot grow the prompt.
    The immediately preceding cue has priority for omitted referents. An older
    cue needs two matching bigrams before displacing the oldest recent example.
    """
    count = min(20, max(0, context_lines))
    window = min(20, max(count, memory_lines))
    recent = history[-window:] if window else []
    if not recent or not count or max_chars <= 0:
        return []
    selected = list(range(max(0, len(recent) - count), len(recent)))
    older = range(max(0, len(recent) - count))
    ranked = sorted(older, key=lambda i: (_overlap(current_ja, recent[i][0]), i), reverse=True)
    if count > 1 and ranked and _overlap(current_ja, recent[ranked[0]][0]) >= 2:
        selected = sorted([ranked[0]] + selected[1:])
    used = 0
    kept = []
    # Keep the most recent examples when the total character budget is tight.
    for index in reversed(selected):
        source, translation = recent[index]
        size = len(source) + len(translation)
        if not source.strip() or size > max_chars - used:
            continue
        used += size
        kept.append((source, translation))
    return list(reversed(kept))


def recurring_term_hints(
    current_ja: str, history: list[tuple[str, str]], glossary,
) -> list[tuple[str, str]]:
    """Learn only consistently translated, isolated Katakana term repetitions.

    A reviewed glossary always wins. Ambiguous, conflicting or single prior
    translations supply no learned hint. Full dialogue and English pronouns are
    never learned as terminology. These remain weak spelling hints, not facts.
    """
    reviewed = {source for source, _ in glossary.find(current_ja)}
    observations: dict[str, Counter] = {}
    ambiguous: set[str] = set()
    for source, english in history[-20:]:
        term = _word(source)
        if not _KATAKANA_TERM.fullmatch(term) or term in reviewed:
            continue
        translation = english.strip(_TRIM).strip("'\"“”")
        if (not _EN_TERM.fullmatch(translation) or len(translation.split()) != 1
                or translation.casefold() in _IGNORE_EN):
            ambiguous.add(term)
            continue
        observations.setdefault(term, Counter())[translation] += 1
    result = []
    for source, counts in observations.items():
        if source in ambiguous or len(counts) != 1 or sum(counts.values()) < 2:
            continue
        if re.search(r"(?<![ァ-ヶー])" + re.escape(source) + r"(?![ァ-ヶー])", current_ja):
            result.append((source, next(iter(counts))))
    return result[:4]


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
    for i, char in enumerate(source):
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
    current_ja: str = "",
    max_chars: int = 640,
) -> str:
    """Relevant bounded reference data beyond the normal PLaMo chat history.

    Only current-source glossary terms may carry over into a new cue. This
    prevents an earlier full stage name overriding a newly spoken nickname.
    English dialogue is never copied into hints; recurring isolated terms need
    independent agreement across prior finalized cues.
    """
    recent = history[-min(20, max(0, memory_lines)):] if memory_lines else []
    if not recent or max_chars <= 0:
        return ""

    lines: list[str] = []
    found: list[str] = []
    seen: set[str] = set()
    current_terms = {source for source, _ in glossary.find(current_ja)} if current_ja else None
    # The static glossary is reliable even if past PLaMO output was not.
    for ja, _ in reversed(recent):
        for source, target in glossary.find(ja):
            if (source in seen or len(source) > 40 or len(target) > 60
                    or (current_terms is not None and source not in current_terms)):
                continue
            seen.add(source)
            found.append(f"{source} = {target}")
            if len(found) >= 6:
                break
        if len(found) >= 6:
            break
    if found:
        lines.append("Recently mentioned glossary names/terms: " + "; ".join(found))

    if current_ja:
        learned = recurring_term_hints(current_ja, recent, glossary)
        if learned:
            lines.append("Consistent prior term spellings (weak data only): "
                         + "; ".join(f"{source} = {target}" for source, target in learned))

    # Do not repeat the model's prior English translations as new hints:
    # a mistaken name or pronoun must not gain authority on later turns.
    # Older JP provides scene continuity, without reinforcing mistranslated
    # English pronouns. Keep it short to minimize tokens and latency.
    older = recent[:-context_lines] if context_lines > 0 else recent
    if current_ja:
        older = sorted(
            ((index, pair) for index, pair in enumerate(older)
             if _overlap(current_ja, pair[0]) >= 2),
            key=lambda item: (_overlap(current_ja, item[1][0]), item[0]),
            reverse=True,
        )[:3]
        older = [pair for _, pair in sorted(older)]
    older_ja = [ja.strip()[:64] for ja, _ in older[-3:] if ja.strip()]
    if older_ja:
        lines.append("Earlier Japanese for scene continuity (do not translate): " + " / ".join(older_ja)[:195])
    # Do not cut a glossary value or a source fragment mid-line: omit whole
    # lower-priority lines when the caller's available context budget is tight.
    bounded = []
    remaining = max_chars
    for line in lines:
        if len(line) + bool(bounded) <= remaining:
            bounded.append(line)
            remaining -= len(line) + (len(bounded) > 1)
    return "\n".join(bounded)
