"""Conservative ASR uncertainty and translation-grounding checks.

Haas's current speech.cpp interface exposes text, NOT confidence scores.
These flags are weak diagnostics, never a license to invent missing speech.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

# Standalone unfamiliar katakana often contains speculative ASR phonetics.
# Do not suppress it: streamers frequently say real names and game terms.
_KATAKANA = re.compile(r"^[ァ-ヶー]{5,16}$")
_COMMON_KATAKANA = (
    "カージャック", "カーチェイス", "パトカー", "プリズン",
    "ハンドル", "ギガノト", "スバル", "ラミィ", "マリン",
    "エイペックス", "ヴァロラント", "ホロライブ", "ワタメ",
    "タイミング", "シルエット", "ユニコーン",
)
_JP_NUM = re.compile(r"[0-9]+(?:[.,][0-9]+)*")
_EN_NUM = re.compile(r"(?<![a-zA-Z0-9])[0-9]+(?:[.,][0-9]+)*(?![a-zA-Z0-9])")
_SIMPLE_NEGATIVE = re.compile(r"^[^、,。！？!?]{1,18}(?:なかった|ない|ませんでした|ません)[。！？!?]*$")
_EN_NEGATIVE = re.compile(
    r"\b(?:no|not|never|none|nothing|nobody|without|unable|"
    r"cannot|can't|couldn't|doesn't|don't|didn't|isn't|wasn't|"
    r"weren't|won't|failed|missing|lack(?:s|ed|ing)?)\b", re.I
)


def uncertain_asr_text(
    japanese: str,
    *,
    caption: str = "",
    caption_coverage: float = 0.0,
    caption_overlap_s: float = 0.0,
) -> bool:
    """Flag uncertain text for context isolation, NEVER discard it."""
    source = unicodedata.normalize("NFKC", japanese).strip(" 。！？!?")
    if _KATAKANA.fullmatch(source) and not any(term in source for term in _COMMON_KATAKANA):
        return True
    if (
        caption and caption_coverage >= .90 and caption_overlap_s >= .75
        and len(source) >= 5 and len(caption) >= 5
        and SequenceMatcher(None, source, caption, autojunk=False).ratio() < .45
    ):
        return True
    return False


def translation_quality_flags(japanese: str, english: str) -> frozenset[str]:
    """Flag only testable contradictions, not merely unusual fluent English.

    Missing nuance, audio hallucination and the correct sense of ambiguous
    Japanese cannot be reliably inferred from the two strings alone.
    """
    ja = unicodedata.normalize("NFKC", japanese)
    en = unicodedata.normalize("NFKC", english)
    flags: set[str] = set()
    # Japanese 万/億/千 multipliers alter numeric magnitude; a bare
    # numeric-token comparison would falsely flag correct "1000万" -> "10 million".
    simple_number_source = not re.search(r"[0-9]+(?:[.,][0-9]+)*[千万億百]", ja)
    jp_numbers = [v.replace(",", "") for v in _JP_NUM.findall(ja)]
    en_numbers = [v.replace(",", "") for v in _EN_NUM.findall(en)]
    if simple_number_source and jp_numbers and en_numbers and jp_numbers != en_numbers:
        flags.add("numbers")
    # Restrict to single-clause, unambiguous endings; complex / indirect
    # negative sentences are not safely classified by a single regex.
    # "すみません" normally means "sorry", not a literal negative.
    polite_apologies = ("すみません", "すいません", "すみませんでした",
                        "申し訳ありません", "かまいません", "構いません")
    is_formula = any(ja.strip("。！？!?") == phrase for phrase in polite_apologies)
    if not is_formula and _SIMPLE_NEGATIVE.fullmatch(ja.strip()) and not _EN_NEGATIVE.search(en):
        flags.add("negation")
    if "だっこ" in ja or "抱っこ" in ja:
        if re.search(r"\bon (?:my|your|his|her|their|our) shoulders?\b", en, re.I):
            flags.add("carry_location")
    if "踏み倒" in ja:
        if re.search(r"\b(?:devastat\w*|destroy\w*|trampled?|crushed?)\b", en, re.I):
            flags.add("nonpayment")
    return frozenset(flags)


def trusted_merge_candidate(
    first_text: str,
    next_text: str,
    *,
    first_key: int | None,
    next_key: int | None,
    first_end_s: float,
    next_start_s: float,
    combined_duration_s: float,
    next_handoff: bool,
    max_gap_s: float,
) -> bool:
    """Only join a known continuation of the SAME verified voice.

    No merge based on words or scene alone: different voices often overlap
    in Hololive GTA conversations. A 500ms wait is only for incomplete turns.
    """
    from .stream_context import is_incomplete_fragment

    gap = next_start_s - first_end_s
    return bool(
        first_key is not None and next_key == first_key
        and not next_handoff and 0 <= gap <= max_gap_s
        and combined_duration_s <= 7.0
        and len(first_text) + len(next_text) <= 110
        and is_incomplete_fragment(first_text)
        and next_text.strip()
    )
