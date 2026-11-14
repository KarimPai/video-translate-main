"""Conservative evidence-based guard for clearly broken ASR hypotheses.

Do not reject legitimate short Japanese reactions or infer spoken language from
script alone. Uncertain disagreements are suppressed only when both models are
usable, materially incompatible, and the selector abstains.
"""
import re
import unicodedata
from .selection import normalize
from ..text_filter import rejection_reason

_LATIN = re.compile(r"[A-Za-z]")
_JA = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_WORD = re.compile(r"[A-Za-z]{2,}")
_PHANTOM = re.compile(r"^(?:thank you for watching|subtitles? by .+|please subscribe)(?:[.! ]*)$", re.I)


def transcript_problem(text: str, duration_s: float) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return "empty"
    reason = rejection_reason(text)
    if reason:
        return reason
    normalized = unicodedata.normalize("NFKC", text).strip()
    if _PHANTOM.fullmatch(normalized):
        return "phantom_outro"
    if len(normalized) > 600 and duration_s < 8:
        return "implausible_length"
    # Do not treat names, borrowed words, or short code-switching as errors.
    words = _WORD.findall(normalized)
    if not _JA.search(normalized) and len(words) >= 6 and duration_s < 1:
        return "implausible_speech_rate"
    return None


def disagreement_unresolved(primary, secondary, selected, similarity, flags, selected_by):
    if selected_by == "uncertain":
        return similarity < 0.45 and bool(flags)
    # A deterministic fallback is not independent evidence for a transcript.
    if selected_by in ("haas", "whisper_turbo_rescue", "haas_secondary_unusable"):
        return False
    return False


_FULL_LAPLUS_ONLY = re.compile(r"ラプラス[・･]?ダークネス[。！？!?]*")
_LAPLUS_MENTION = re.compile(r"ラプラス|らぷらす|ラプ|らぷ")


def uncorroborated_laplus_full_name(chosen: str, other: str) -> bool:
    """Flag a standalone formal name that the independent ASR did not hear.

    Never reject sentences merely mentioning Laplus, or one model saying the
    short name while the other says the long name. This is not audio proof.
    """
    return (bool(_FULL_LAPLUS_ONLY.fullmatch(chosen.strip()))
            and not _LAPLUS_MENTION.search(other))


def suspicious_phonetics(text):
    """Only a review signal on its own: katakana can be a real name."""
    return re.fullmatch(r"[ァ-ヺー]{6,}[。！？!?]?", text.strip()) is not None
