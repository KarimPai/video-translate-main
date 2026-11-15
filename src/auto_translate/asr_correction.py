"""High-precision Japanese ASR typo repairs supported by video context.

Only exact, reviewed HoloGTA/Hololive variants are corrected. This is not a
generic fuzzy spellchecker: never infer unheard dialogue, names or negation.
"""
import re

from .glossary import nickname_boundary
from .asr.selection import normalize

_HOLO = re.compile(r"ホロ|holo", re.IGNORECASE)
_GTA_ALIASES = re.compile(
    r"(?:コロナイブ|コロライブ|ホロライフ)\s*(?:ＧＴＡ|GTA)",
    re.IGNORECASE,
)
_HOLOMEM_ALIAS = re.compile(r"フォロメン")


_RAP_SOUND = re.compile(r"ラップ|らっぷ|(?<![A-Za-z])(?:rapping|rap)(?![A-Za-z])", re.I)
_LAP_SOUND = re.compile(r"ラプ|らぷ")
_LITERAL_RAP = re.compile(r"音楽|ラッパー|ヒップホップ|ビート|リズム|歌|曲|韻|上手|得意|包[むんみ]|包装|サラン|食品")
_ADDRESS = re.compile(r"(?:様|ちゃん|さん)")


def lap_spelling_from_peer(text: str, peer: str, *, aligned=False, recent=()) -> str:
    """Admit one bounded sound substitution only when the entire peer line aligns."""
    if not peer:
        return text
    for heard, supported in ((_RAP_SOUND, _LAP_SOUND), (_LAP_SOUND, _RAP_SOUND)):
        reverse = heard is _LAP_SOUND
        if reverse and not aligned:
            continue
        if not reverse and _LITERAL_RAP.search(text):
            continue
        for m in heard.finditer(text):
            suffix = _ADDRESS.match(text, m.end())
            end = suffix.end() if suffix else m.end()
            if not nickname_boundary(text, m.start(), end):
                continue
            if not (aligned or suffix or any(
                re.search(r"(?:ラプ|らぷ)(?:様|ちゃん|さん)|ラプラス・ダークネス", line)
                for line in recent[-2:]
            )):
                continue
            for form in supported.finditer(peer):
                candidate = text[:m.start()] + form.group() + text[m.end():]
                if normalize(candidate) == normalize(peer):
                    return candidate
    return text


def correct_known_asr_terms(
    japanese: str, *, video_context: str = "", caption: str = "",
) -> str:
    """Repair limited audio-confusable names when Holo context is confirmed.

    Do not contradict an aligned caption that explicitly agrees with the
    apparent variant; this rule operates before PLaMo glossary lookup.
    """
    updated = lap_spelling_from_peer(japanese, caption, aligned=True)
    if not japanese or not _HOLO.search(video_context):
        return updated
    if not _GTA_ALIASES.search(caption):
        updated = _GTA_ALIASES.sub("ホロライブGTA", updated)
    if "フォロメン" not in caption:
        updated = _HOLOMEM_ALIAS.sub("ホロメン", updated)
    # In Hololive videos these precise phonetic variants are established
    # nicknames, not free-form fuzzy matching. A conflicting aligned caption wins.
    if not any(term in caption for term in ("はやちゃま", "はーちゃま", "はぁちゃま", "ハーチャマ")):
        for alias in ("はやちゃま", "はーちゃま", "はぁちゃま", "ハーチャマ"):
            updated = updated.replace(alias, "はあちゃま")
    return updated
