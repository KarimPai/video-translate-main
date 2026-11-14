"""Short, context-gated Hololive name hints and safe English spelling repairs.

These only affect known names in Hololive streams. Do not use them for
general Japanese or reconstruct words absent from the audio.
"""
import re

_HOLO = re.compile(r"ホロ|holo", re.I)
_HOLOGTA = re.compile(r"ホロ(?:ライブ)?GTA|ホロスサントス|holo.?gta", re.I)
_HAACHAMA_JA = ("はあちゃま", "はぁちゃま", "はーちゃま", "はやちゃま", "ハーチャマ")
_LAPLUS_JA = ("ラプラス", "らぷらす")


def whisper_name_prompt(video_title: str, *, enabled: bool = True) -> str | None:
    """Bias Whisper only for a known HoloGTA scene, with few proper nouns.

    No free-form instruction is fed to Whisper; these are Japanese spellings.
    Its no-speech and language filters remain unchanged.
    """
    if not enabled or not _HOLOGTA.search(video_title):
        return None
    return "はあちゃま。ラプラス・ダークネス。さくらみこ。"


def canonicalize_english_names(japanese: str, english: str, video_title: str = "") -> str:
    """Correct only strongly evidenced, known English name misspellings."""
    if not _HOLO.search(video_title) or not japanese or not english:
        return english
    if any(name in japanese for name in _HAACHAMA_JA):
        english = re.sub(
            r"\b(?:Ha+[- ]?chama|Hasa[- ]?chama|Haya[- ]?chama)\b",
            "Haachama", english, flags=re.I,
        )
    if any(name in japanese for name in _LAPLUS_JA):
        english = re.sub(r"\bLaplace\b", "Laplus", english, flags=re.I)
    return english
