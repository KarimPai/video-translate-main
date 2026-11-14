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
    """Legacy compatibility hook. Never seed Whisper with unspoken names.

    Existing configs can still contain whisper_name_hints; the dual-ASR
    pipeline no longer calls this hook or supplies an initial name prompt.
    """
    return None


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
        # Prefer the short spoken name in English; do not expand it to a
        # full stage name, and do not introduce names absent from the JP.
        english = re.sub(
            r"(?<![A-Za-z])(?:La\s*\+|La\s+plus|Laplus|Laplace)"
            r"(?:[\s·・-]+Darkness+)?(?![A-Za-z])",
            "Laplus", english, flags=re.I,
        )
    return english
