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



def foreign_audio_transcript_problem(text: str, decision, caption_match=None) -> str | None:
    """Reject obvious Japanese-forced foreign phonetics only with audio evidence.

    Japanese-only ASR can transliterate English/Indonesian as meaningless
    katakana or mixed Latin fragments. Script alone is NOT language ID:
    require independent CPU ECAPA votes for EN/ID, no strong Japanese vote,
    and a conspicuously corrupted transcript. Preserve short replies and
    ordinary Japanese, including genuine loanwords and proper names.
    """
    if decision is None or not text or getattr(decision, "drop", False):
        return None
    windows = getattr(decision, "windows", ())
    if not windows:
        return None
    if (caption_match is not None
            and getattr(caption_match, "reliable", False)
            and getattr(caption_match, "coverage", 0.0) >= 0.80
            and _JA.search(getattr(caption_match, "text", ""))):
        return None

    # A confident Japanese speech window protects the entire mixed cue.
    if any(lang == "ja" and ja_prob >= 0.25
           for lang, _, ja_prob, _ in windows):
        return None
    if any(ja_prob >= 0.12 for _, _, ja_prob, _ in windows):
        return None

    foreign_votes = sum(
        lang in ("en", "id") and probability >= 0.75
        and ja_prob <= 0.08 and margin >= 0.30
        for lang, probability, ja_prob, margin in windows
    )
    # A two-window cue must be foreign-leaning at both ends. Three windows
    # tolerate one uncertain window, but never a Japanese-positive window.
    required = 2 if len(windows) == 2 else (
        2 if len(windows) >= 3 else 1
    )
    if foreign_votes < required:
        return None

    normalized = unicodedata.normalize("NFKC", text)
    hiragana_kanji = re.findall(r"[\u3040-\u309f\u3400-\u9fff]", normalized)
    katakana = re.findall(r"[\u30a0-\u30ff]", normalized)
    latin = re.findall(r"[A-Za-z]", normalized)
    jp_count = len(hiragana_kanji) + len(katakana)
    # Many continuous or phrase-long katakana syllables can be a forced
    # phonetic rendering of foreign speech. Never drop from this alone.
    if (len(katakana) >= 16 and len(hiragana_kanji) <= 2
            and len(katakana) / max(jp_count, 1) >= 0.85):
        return "foreign_audio_katakana_soup"
    # Latin word fragments mixed with only a couple of Japanese particles
    # are unlikely to be a meaningful Japanese transcript.
    if (len(latin) >= 12 and len(_WORD.findall(normalized)) >= 3
            and len(hiragana_kanji) <= 5):
        return "foreign_audio_latin_fragments"
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
        # The selector explicitly abstained. Flagged material disagreements
        # are unsafe, and very low agreement between two substantive long
        # hypotheses is unsafe even when neither contains numbers/negation.
        return ((similarity < 0.45 and bool(flags))
                or (similarity < 0.25
                    and min(len(normalize(primary)), len(normalize(secondary))) >= 10))
    # A deterministic fallback is not independent evidence for a transcript.
    if selected_by in ("haas", "parakeet_rescue", "haas_secondary_unusable"):
        return False
    return False
