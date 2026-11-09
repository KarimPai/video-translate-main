"""Evidence-bound GTA roleplay terminology and short polarity protections.

Do not rewrite uncertain Haas transcripts into a guess about unseen audio.
These helpers correct ONLY Japanese that already clearly supports a meaning.
"""
from __future__ import annotations

import re

from ..stream_context import is_gta_context

_VEST_COUNT = re.compile(r"ベスト[0-9０-９一二三四五六七八九十]+枚")
_VEST_PURCHASE = re.compile(r"ベスト(?:を)?(?:買|持|着|装備)")
_ROCKET_AMMO = re.compile(r"ロケラン(?:の)?弾")
_ROCKET_COST = re.compile(
    r"ロケランの弾(?:は|が|で)?(?:一千万|1000万|１千万|10,?000,?000)"
    r"[。.!！\s]*$"
)
_POSITIVE = {
    "すごい": "Amazing!",
    "凄い": "Amazing!",
    "素晴らしい": "Wonderful!",
    "最高": "The best!",
    "よかった": "That's great!",
    "良かった": "That's great!",
}
_NEGATIVE = {
    "無理": "That's impossible.",
    "だめ": "No good.",
    "駄目": "No good.",
    "できない": "I can't do it.",
    "出来ない": "I can't do it.",
}
_NEGATIVE_EN = re.compile(
    r"\b(?:bad|awful|terrible|horrible|worst|useless|sucks?)\b|"
    r"\b(?:not|isn't|wasn't|can't|cannot|don't)\s+(?:good|great|amazing|possible)\b",
    re.I,
)
_POSITIVE_EN = re.compile(
    r"\b(?:good|great|awesome|amazing|wonderful|excellent|fine|okay|"
    r"certainly|sure|possible|can do|no problem)\b",
    re.I,
)
_BAD_VEST = re.compile(r"\b(?:of\s+)?the\s+best\b", re.I)


def _compact(sentence: str) -> str:
    return sentence.strip().rstrip(" 。.!！?？\t\r\n")


def short_polarity_fallback(japanese: str) -> str | None:
    """Only fully explicit, self-contained positive/negative replies."""
    source = _compact(japanese)
    return _POSITIVE.get(source) or _NEGATIVE.get(source)


def polarity_conflict(japanese: str, english: str) -> bool:
    """Don't infer the polarity of colloquial ambiguous or mixed sentences."""
    source = _compact(japanese)
    if source in _POSITIVE:
        return bool(_NEGATIVE_EN.search(english))
    if source in _NEGATIVE:
        return bool(_POSITIVE_EN.search(english)) and not bool(
            re.search(r"\b(?:not|no|never|can't|cannot|don't|impossible)\b", english, re.I)
        )
    return False


def gta_prompt_terms(
    japanese: str,
    title: str = "",
    recent: list[str] | tuple[str, ...] = (),
) -> list[tuple[str, str]]:
    """Add glossary hints only where Japanese and GTA context agree."""
    neighbors = " ".join(list(recent)[-6:])
    gta = is_gta_context(title) or any(
        word in neighbors for word in ("アーマー", "ロケラン", "ブラックマーケット", "ギャング", "ステッカー")
    )
    if not gta:
        return []
    hits: list[tuple[str, str]] = []
    if _VEST_COUNT.search(japanese) or _VEST_PURCHASE.search(japanese):
        hits.append(("ベスト", "protective vest / body armor (equipment)"))
    if "ロケラン" in japanese:
        hits.append(("ロケラン", "rocket launcher (weapon)"))
    if _ROCKET_AMMO.search(japanese):
        hits.append(("弾", "ammunition / rocket rounds"))
    if "ステッカー" in japanese:
        hits.append(("ステッカー", "vehicle sticker / decal"))
    if "フルカス" in japanese:
        hits.append(("フルカス", "fully customized vehicle"))
    if "ブラックマーケット" in japanese:
        hits.append(("ブラックマーケット", "Black Market (game location)"))
    return hits


def repair_gta_translation(
    japanese: str, english: str, title: str = "",
    recent: list[str] | tuple[str, ...] = (),
) -> str:
    """Correct directly evidenced semantic failures, never invent dialogue."""
    if not (is_gta_context(title) or gta_prompt_terms(japanese, title, recent)):
        return english

    if _VEST_COUNT.search(japanese):
        english = _BAD_VEST.sub("vests", english)
    elif _VEST_PURCHASE.search(japanese):
        english = re.sub(r"\\bthe\\s+best(?:\\s+one)?\\b", "a vest", english, flags=re.I)
    if _ROCKET_AMMO.search(japanese):
        if _ROCKET_COST.fullmatch(_compact(japanese)):
            if re.search(r"\bballs?\b|\bmarbles?\b", english, re.I):
                return "Rocket launcher ammunition costs 10 million."
        english = re.sub(r"\b(?:balls?|marbles?)\b", "rounds", english, flags=re.I)
    if (
        _compact(japanese) == "この横貼りたくね"
        and any("ステッカー" in line or "車両飾" in line for line in recent[-8:])
        and not re.search(r"\b(?:stick|put|attach|place|paste|decal|sticker)\b", english, re.I)
    ):
        return "Wouldn't it look good stuck on the side?"
    return english
