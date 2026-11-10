"""Conservative context and fidelity safeguards for Japanese-to-English subtitles.

None of these helpers changes an ASR transcript or its timestamps. Nearby
fragments only become reference context for PLaMo, not combined subtitle text.
Numerical and name repairs require direct evidence in the CURRENT Japanese.
"""
from __future__ import annotations

import re
import unicodedata

# Only nearby, nontrivial pieces of speech can inform a fragmented utterance.
_SMALL_REACTIONS = frozenset({
    "あ", "え", "えっ", "ああ", "お", "うん", "はい", "そう", "うわ",
    "おお", "ん", "うーん", "はいはい", "あの", "えー", "へえ",
})
_JP_DIGITS = {"〇": 0, "零": 0, "一": 1, "二": 2, "三": 3,
              "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
_LARGE_UNITS = {"万": 10000, "億": 100000000}
_NUM_CHARS = "〇零一二三四五六七八九十百千万億"
_JP_NUM = re.compile(
    r"(?<![0-9０-９〇零一二三四五六七八九十百千万億])"
    r"(?:[0-9０-９][0-9０-９,，]*|[〇零一二三四五六七八九十百千万億]{2,})"
    r"(?![0-9０-９〇零一二三四五六七八九十百千万億])"
)
# A period/colon followed by a digit belongs to a decimal or time.
# A terminal period/colon is punctuation and must NOT hide the number.
# Guard both sides so 8:28 and 3.14 are never mistaken for standalone IDs.
_EN_NUM = re.compile(
    r"(?<![\w.:])(?:\d{1,3}(?:,\d{3})+|\d+)(?!\w|[.:]\d)"
)
_NAME_ALIASES = (
    ("マリン", "Marine", re.compile(r"\bMarin\b", re.IGNORECASE)),
    ("ベイ", "Bae", re.compile(r"\bBay\b", re.IGNORECASE)),
    ("ベイちゃん", "Bae", re.compile(r"\bBay\b", re.IGNORECASE)),
    ("ベーちゃん", "Bae", re.compile(r"\bBay\b", re.IGNORECASE)),
)


def nearby_japanese_fragments(
    recent: list[tuple[float, float, str]],
    current_start_s: float,
    current_ja: str,
    *,
    max_gap_s: float = 1.8,
    max_fragments: int = 2,
) -> tuple[str, ...]:
    """Look behind only, without increasing playback or translation latency.

    The last two timestamp-adjacent, useful JP utterances provide grammar
    context for short/fragmented lines. Never cross a long pause, seek, or a
    backchannel that may mark a different speaker. Do not fuse timestamps.
    """
    current = current_ja.strip()
    if (
        max_fragments <= 0 or max_gap_s <= 0 or not current
        or current in _SMALL_REACTIONS or len(current) > 34
    ):
        return ()
    selected: list[str] = []
    cursor = current_start_s
    for start_s, end_s, japanese in reversed(recent):
        gap = cursor - end_s
        if gap < -0.05 or gap > max_gap_s:
            break
        phrase = japanese.strip()
        if phrase in _SMALL_REACTIONS or len(phrase) < 2:
            break
        if len(phrase) > 34:
            break
        selected.append(phrase[:34])
        cursor = start_s
        if len(selected) >= max_fragments:
            break
    return tuple(reversed(selected))


def _japanese_number(raw: str) -> int | None:
    """Convert explicit Kanji/ASCII numeric runs, not ordinary Japanese words."""
    value = unicodedata.normalize("NFKC", raw).replace(",", "")
    if value.isascii() and value.isdecimal():
        return int(value)
    if any(char not in _NUM_CHARS for char in value):
        return None
    if not any(char in _JP_DIGITS for char in value):
        return None
    if not any(char in _SMALL_UNITS or char in _LARGE_UNITS for char in value):
        if len(value) < 2:
            return None
        return int("".join(str(_JP_DIGITS[ch]) for ch in value))

    total = 0
    section = 0
    digit = None
    previous_small = 10000
    previous_large = 1000000000
    saw_number = False
    for char in value:
        if char in _JP_DIGITS:
            if digit is not None:
                return None  # ambiguous positional digits mixed with units
            digit = _JP_DIGITS[char]
            saw_number = True
        elif char in _SMALL_UNITS:
            unit = _SMALL_UNITS[char]
            if unit >= previous_small:
                return None
            section += (1 if digit is None else digit) * unit
            previous_small = unit
            digit = None
        else:
            unit = _LARGE_UNITS[char]
            if unit >= previous_large or (section == 0 and digit is None):
                return None  # excludes idiom 万一
            total += (section + (digit or 0)) * unit
            previous_large = unit
            previous_small = 10000
            section = 0
            digit = None
    if not saw_number:
        return None
    return total + section + (digit or 0)


def japanese_number_facts(japanese: str, *, limit: int = 4) -> list[tuple[str, int]]:
    """Numerals explicitly present in the CURRENT ASR output only."""
    facts: list[tuple[str, int]] = []
    for match in _JP_NUM.finditer(japanese):
        number = _japanese_number(match.group())
        if number is None or number < 0 or number > 999999999:
            continue
        facts.append((match.group(), number))
        if len(facts) >= limit:
            break
    return facts


def enforce_numeric_fidelity(japanese: str, english: str) -> str:
    """Fix a changed Arabic numeral only when source/target counts align.

    No speculative insertion if the translation omitted a number or spelled
    it in words. This cannot and does not validate misheard Japanese numbers.
    """
    source_numbers = [value for _, value in japanese_number_facts(japanese)]
    english_matches = list(_EN_NUM.finditer(english))
    if not source_numbers or not english_matches:
        return english
    if len(source_numbers) != len(english_matches):
        return english  # ambiguity: never silently replace unrelated numbers
    target_numbers = source_numbers

    result = english
    for match, correct in reversed(list(zip(english_matches, target_numbers))):
        spoken = match.group().replace(",", "")
        if int(spoken) != correct:
            result = result[:match.start()] + str(correct) + result[match.end():]
    return result


def enforce_known_name_spellings(japanese: str, english: str, glossary) -> str:
    """Correct only a known Romanization variant of an explicit glossary name.

    Never use audio/title context to guess an unrecognized name, and never
    rewrite generic English words absent a matching Japanese source term.
    """
    glossary_hits = set(glossary.find(japanese))
    for source, target, spelling in _NAME_ALIASES:
        if (source, target) in glossary_hits:
            english = spelling.sub(target, english)
    return english
