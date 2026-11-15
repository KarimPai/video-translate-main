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
    # Deterministic fallbacks cannot turn weak agreement into an abstention.
    return selected_by == "uncertain" and similarity < 0.45 and bool(flags)


def uncorroborated_full_name(chosen: str, other: str, name_guards=()) -> bool:
    """Flag a standalone long name only if the peer has no short-name evidence.

    Patterns are configured in glossary.toml; ordinary name mentions are safe.
    """
    return any(full.fullmatch(chosen.strip()) and not mention.search(other)
               for full, mention in name_guards)


def suspicious_phonetics(text):
    """Only a review signal on its own: katakana can be a real name."""
    return re.fullmatch(r"[ァ-ヺー]{6,}[。！？!?]?", text.strip()) is not None

_JAPANESE_OUTROS = ("ご視聴ありがとうございました", "チャンネル登録よろしくお願いします")


def weak_outro(candidate, peer, duration, vad_confidence, audio):
    """Flag isolated complete outros only on very short, faint, weak-VAD audio."""
    if normalize(candidate) not in _JAPANESE_OUTROS or normalize(candidate) in normalize(peer):
        return False
    if duration > 1.3 or vad_confidence is None or vad_confidence >= 0.55:
        return False
    import numpy as np
    return bool(len(audio) and np.mean(np.square(audio)) < 0.0001)
