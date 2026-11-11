"""High-precision Japanese ASR typo repairs supported by video context.

Only exact, reviewed HoloGTA/Hololive variants are corrected. This is not a
generic fuzzy spellchecker: never infer unheard dialogue, names or negation.
"""
import re

_HOLO = re.compile(r"ホロ|holo", re.IGNORECASE)
_GTA_ALIASES = re.compile(
    r"(?:コロナイブ|コロライブ|ホロライフ)\s*(?:ＧＴＡ|GTA)",
    re.IGNORECASE,
)
_HOLOMEM_ALIAS = re.compile(r"フォロメン")


def correct_known_asr_terms(
    japanese: str, *, video_context: str = "", caption: str = "",
) -> str:
    """Repair limited audio-confusable names when Holo context is confirmed.

    Do not contradict an aligned caption that explicitly agrees with the
    apparent variant; this rule operates before PLaMo glossary lookup.
    """
    if not japanese or not _HOLO.search(video_context):
        return japanese
    updated = japanese
    if not _GTA_ALIASES.search(caption):
        updated = _GTA_ALIASES.sub("ホロライブGTA", updated)
    if "フォロメン" not in caption:
        updated = _HOLOMEM_ALIAS.sub("ホロメン", updated)
    return updated
