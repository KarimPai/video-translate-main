"""Conservative evidence-based guard for clearly broken ASR hypotheses.

Do not reject legitimate short Japanese reactions or infer spoken language from
script alone. Uncertain disagreements are suppressed only when both models are
usable, materially incompatible, and the selector abstains.
"""
import re
import unicodedata
from difflib import SequenceMatcher
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



def caption_supported_hypothesis(primary: str, secondary: str, caption_match):
    """Identify a hypothesis independently corroborated by aligned JP captions.

    Strong similarity AND a clear margin over the other candidate are needed;
    captions are timing-dependent and can themselves be incorrect or rolling.
    Return (label, score, competing_score) or None. Never generate new text.
    """
    if (caption_match is None or not getattr(caption_match, "reliable", False)
            or getattr(caption_match, "coverage", 0.0) < 0.80):
        return None
    caption = getattr(caption_match, "text", "")
    if not _JA.search(caption):
        return None
    anchor = normalize(caption)
    if len(anchor) < 4 or len(anchor) > 240:
        return None
    if (transcript_problem(primary, 15.0) is not None
            or transcript_problem(secondary, 15.0) is not None):
        return None
    a = SequenceMatcher(None, normalize(primary), anchor, autojunk=False).ratio()
    b = SequenceMatcher(None, normalize(secondary), anchor, autojunk=False).ratio()
    if a >= 0.78 and a - b >= 0.20:
        return ("haas", a, b)
    if b >= 0.78 and b - a >= 0.20:
        return ("parakeet", b, a)
    return None

def disagreement_unresolved(primary, secondary, selected, similarity, flags, selected_by):
    if selected_by == "uncertain":
        return similarity < 0.45 and bool(flags)
    # A deterministic fallback is not independent evidence for a transcript.
    if selected_by in ("haas", "parakeet_rescue", "haas_secondary_unusable"):
        return False
    return False
