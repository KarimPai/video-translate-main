"""Conservative, deterministic checks to decide whether PLaMo merits a retry.

These heuristics are *not* a semantic accuracy score. They only flag a few
high-confidence errors; clean-looking text can still be mistranslated.
"""
import re
import logging
import unicodedata
from collections import Counter
from decimal import Decimal, InvalidOperation

from .name_hints import SPOKEN_NAME_FORMS
from .subtitle_clean import english_output_issue

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
_JP_VALUE = re.compile(r"\d+(?:\.\d+)?(?:億|万|千)?")
_EN_VALUE = re.compile(
    r"(?<![a-z0-9])\d[\d,]*(?:\.\d+)?"
    r"(?:\s*(?:billion|million|thousand|bn|mm|m|k))?(?![a-z])",
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
_GUARDED_NAMES = frozenset({"hololive", "watame", "pekora", "miko", "korone", "subaru", "haachama", "laplus", "botan", "shishiron"})


def _mention_value(mention, field, default=None):
    return mention.get(field, default) if isinstance(mention, dict) else getattr(mention, field, default)


def resolved_name_issue(japanese: str, english: str, mentions, *, names=None) -> str | None:
    """Check independently resolved occurrences, including kanji spellings.

    This checks evidence and exact name counts, not English semantic accuracy.
    In particular, Miko cannot satisfy Mikochi or Miko-chan, and a name found
    elsewhere in a mixed cue cannot excuse a missing second mention.
    """
    if mentions is None and names is None:
        return None
    expected = Counter()
    literal_families = set()
    for mention in mentions or ():
        source = _mention_value(mention, "source", "")
        start, end = _mention_value(mention, "start", -1), _mention_value(mention, "end", -1)
        if type(start) is not int or type(end) is not int or japanese[start:end] != source:
            continue
        classification = _mention_value(mention, "classification")
        family = _mention_value(mention, "family", "")
        if classification == "PERSON":
            target = _mention_value(mention, "target")
            if target:
                expected[target] += 1
        elif classification == "COMMON_NOUN":
            literal_families.add(family)
    if expected:
        # One English span can satisfy one occurrence only. Matching longest
        # first prevents Natsuiro Matsuri from also satisfying a spoken Matsuri.
        pattern = r"(?<![\w-])(?:" + "|".join(
            re.escape(target) for target in sorted(expected, key=len, reverse=True)) + r")(?![\w-])"
        actual = Counter(match.group().casefold() for match in re.finditer(pattern, english, re.I))
        if any(actual[target.casefold()] < count for target, count in expected.items()):
            return "proper_name_mismatch"
    supported = set(expected)
    supported.update(target for source, target in (names or ()) if source in japanese)
    for full, family in (("Natsuiro Matsuri", "Matsuri"), ("Sakura Miko", "Miko"),
                         ("Shishiro Botan", "Botan"),
                         ("Tsunomaki Watame", "Watame"), ("Fuwawa Abyssgard", "Fuwawa")):
        if (any(target == family or target.startswith(family + "-") for target in supported)
                and full not in supported and re.search(r"\b" + re.escape(full) + r"\b", english, re.I)):
            return "proper_name_expansion"
    for family, pattern in (("Matsuri", r"\b(?:Natsuiro\s+)?Matsuri\b"),
                            ("Botan", r"\b(?:Shishiro\s+)?Botan\b"),
                            ("Miko", r"\b(?:Sakura\s+)?Miko(?:chi)?\b")):
        if family in literal_families:
            allowed = sum(count for target, count in expected.items()
                          if re.search(pattern, target, re.I))
            if len(re.findall(pattern, english, re.I)) > allowed:
                return "literal_noun_mismatch"
    # Mask complete supported English spans first. A distinct short token
    # inside a supplied full name (Fuwawa in Fuwawa Abyssgard) is not a new
    # unsupported mention. Earlier expansion/count checks remain independent.
    remaining = english
    if supported:
        allowed_pattern = r"(?<![\w-])(?:" + "|".join(
            re.escape(target) for target in sorted(supported, key=len, reverse=True)) + r")(?![\w-])"
        remaining = re.sub(allowed_pattern, lambda m: "\0" * len(m.group()), remaining, flags=re.I)
    distinctive = (r"(?:Natsuiro\s+)?Matsuri|(?:Sakura\s+)?Miko(?:chi)?|(?:Shishiro\s+)?Botan|Shishiron|"
                   r"(?:Tsunomaki\s+)?Watamee?|Laplus|Haachama|Fuwawa|Iofi")
    if re.search(r"(?<![\w-])(?:" + distinctive
                 + r")(?:-(?:chan|san|sama|senpai|kun))?(?![\w-])", remaining, re.I):
        return "unsupported_proper_name"
    literal_patterns = {
        "Botan": r"\b(?:buttons?|switch(?:es)?|controls?|knobs?)\b",
        "Matsuri": r"\b(?:festivals?|festivit(?:y|ies)|fairs?|carnivals?|celebrations?|events?|f[eê]tes?)\b",
        "Miko": r"\b(?:shrine\s+(?:maidens?|attendants?)|priestess(?:es)?)\b",
    }
    if any(not re.search(literal_patterns[family], english, re.I)
           for family in literal_families if family in literal_patterns):
        # English can omit/rephrase a noun legitimately. This is a review
        # signal for a bounded retry, never proof warranting a global repair.
        return "possible_literal_noun_loss"
    return None


def suspicious_translation(
    japanese: str,
    english: str,
    *,
    names: list[tuple[str, str]] | None = None,
    mentions=None,
) -> str | None:
    """Return a reason for at most one context-reduced retry, or None."""
    en = english.strip()
    output_issue = english_output_issue(en)
    if output_issue:
        return output_issue

    jp = unicodedata.normalize("NFKC", japanese).strip()
    name_issue = resolved_name_issue(japanese, en, mentions, names=names)
    if name_issue:
        return name_issue
    if len(jp) <= 24:
        if (_SHORT_POSITIVE_JA.fullmatch(jp) and _NEGATIVE_EN.search(en)
                and not re.search(r"\bnot\s+bad\b", en, re.I)):
            return "positive_reaction_reversed"
        if _NEGATIVE_JA.search(jp) and not _NEGATIVE_EN_MEANING.search(en):
            return "possible_negation_loss"

    # Longer clauses and inflected negatives are review signals only.
    if (re.search(r"(?:でき|出来|知ら|行か|分から|わから|食べ|見|じゃ|では)(?:ない|なかった|ず)"
                  r"|(?:し|き|り|み|れ|せ|け|べ)ません(?:でした)?", jp)
            and not _NEGATIVE_EN_MEANING.search(en.replace("’", "'"))):
        logging.getLogger(__name__).info("Review possible negation loss: JA=%r EN=%r", jp, en)

    src_numbers = _number_values(jp, japanese=True)
    eng_numbers = _number_values(unicodedata.normalize("NFKC", en), japanese=False)
    if src_numbers and eng_numbers and src_numbers != eng_numbers:
        return "numeric_mismatch"

    for source, target in (names or []):
        if (target in SPOKEN_NAME_FORMS or target.casefold() in _GUARDED_NAMES) and source in japanese:
            if not re.search(r"(?<![a-z])" + re.escape(target) + r"(?![a-z])", en, re.I):
                return "proper_name_mismatch"
    return None
