"""Conservative, deterministic checks to decide whether PLaMo merits a retry.

These heuristics are *not* a semantic accuracy score. They only flag a few
high-confidence errors; clean-looking text can still be mistranslated.
"""
import re
import unicodedata
from collections import Counter
from decimal import Decimal, InvalidOperation

from .subtitle_clean import has_generation_loop, looks_like_prompt_echo

_SHORT_POSITIVE_JA = re.compile(
    r"^(?:すごい|凄い|すげえ|すげー|いいね|良いね|よかった|良かった|"
    r"ナイス|最高)[ねよ！!。〜～ー]*$"
)
_NEGATIVE_EN = re.compile(
    r"\b(?:bad|terrible|awful|horrible|worst|wrong|lame|sucks)\b|"
    r"\b(?:not|no)\s+good\b",
    re.IGNORECASE,
)
_NEGATIVE_JA = re.compile(
    r"(?:できない|出来ない|知らない|分からない|わからない|"
    r"いらない|要らない|じゃない|ではない|行かない|"
    r"欲しくない|無理)"
)
_NEGATIVE_EN_MEANING = re.compile(
    r"\b(?:not|never|no|nothing|nobody|none|without|can't|cannot|"
    r"couldn't|won't|don't|doesn't|didn't|isn't|aren't|wasn't|"
    r"weren't|unable|impossible|unknown|unclear|unsure|clueless|"
    r"refuse|refused|lack|lacking)\b",
    re.IGNORECASE,
)
# Compare numeric *values*, not surface digits: 100万 == 1 million,
# 8400万 == 84 million, and 3億 == 300 million.
_JP_VALUE = re.compile(r"\\d+(?:\\.\\d+)?(?:億|万|千)?")
_EN_VALUE = re.compile(
    r"(?<![a-z0-9])\\d[\\d,]*(?:\\.\\d+)?"
    r"(?:\\s*(?:billion|million|thousand|bn|mm|m|k))?(?![a-z])",
    re.I,
)
_JP_SCALE = {"億": Decimal(100_000_000), "万": Decimal(10_000),
             "千": Decimal(1_000)}
_EN_SCALE = {"billion": Decimal(1_000_000_000), "bn": Decimal(1_000_000_000),
             "million": Decimal(1_000_000), "mm": Decimal(1_000_000),
             "m": Decimal(1_000_000), "thousand": Decimal(1_000),
             "k": Decimal(1_000)}


def _number_values(text: str, japanese: bool) -> Counter:
    """Canonicalize common unit-suffixed numbers without estimating words."""
    counts = Counter()
    pattern = _JP_VALUE if japanese else _EN_VALUE
    for hit in pattern.finditer(text):
        match = hit.group().replace(",", "").strip()
        unit = ""
        if japanese and match[-1:] in _JP_SCALE:
            unit, match = match[-1], match[:-1]
        elif not japanese:
            scale_hit = re.search(r"(billion|million|thousand|bn|mm|m|k)$", match, re.I)
            if scale_hit:
                unit = scale_hit.group().lower()
                match = match[:scale_hit.start()].strip()
        try:
            counts[Decimal(match) * ((_JP_SCALE if japanese else _EN_SCALE).get(unit, 1))] += 1
        except InvalidOperation:
            pass
    return counts
# Only distinctive entity names. Ordinary descriptive glossary terms can be
# rendered naturally and should not trigger unnecessary retries.
_GUARDED_NAMES = frozenset({"hololive", "watame", "pekora", "miko", "korone", "subaru"})


def suspicious_translation(
    japanese: str,
    english: str,
    *,
    names: list[tuple[str, str]] | None = None,
) -> str | None:
    """Return a reason for at most one context-reduced retry, or None."""
    en = english.strip()
    if not en:
        return "empty_english"
    if has_generation_loop(en) or looks_like_prompt_echo(en):
        return "prompt_echo_or_loop"

    jp = unicodedata.normalize("NFKC", japanese).strip()
    if len(jp) <= 24:
        if (_SHORT_POSITIVE_JA.fullmatch(jp) and _NEGATIVE_EN.search(en)
                and not re.search(r"\bnot\s+bad\b", en, re.I)):
            return "positive_reaction_reversed"
        if _NEGATIVE_JA.search(jp) and not _NEGATIVE_EN_MEANING.search(en):
            return "possible_negation_loss"

    src_numbers = _number_values(jp, japanese=True)
    eng_numbers = _number_values(unicodedata.normalize("NFKC", en), japanese=False)
    if src_numbers and eng_numbers and src_numbers != eng_numbers:
        return "numeric_mismatch"

    for source, target in (names or []):
        if target.casefold() in _GUARDED_NAMES and source in jp:
            if not re.search(r"(?<![a-z])" + re.escape(target) + r"(?![a-z])", en, re.I):
                return "proper_name_mismatch"
    return None
