"""Conservative guard for clearly broken ASR hypotheses.

Do not reject legitimate short Japanese reactions. Reject explicit nonspeech,
decoder loops, phantom outros and physically implausible output.
"""
import re
import unicodedata

from ..text_filter import rejection_reason

_JA = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_WORD = re.compile(r"[A-Za-z]{2,}")
_PHANTOM = re.compile(r"^(?:thank you for watching|subtitles? by .+|please subscribe|ご視聴ありがとうございました)(?:[.!。！ ]*)$", re.I)


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
    # Japanese speech peaks near 10 morae/s; allow generous slack for VAD padding.
    if len(_JA.findall(normalized)) > 24 + duration_s * 14:
        return "implausible_speech_rate"
    words = _WORD.findall(normalized)
    if not _JA.search(normalized) and len(words) >= 6 and duration_s < 1:
        return "implausible_speech_rate"
    return None
