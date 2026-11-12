"""Conservative spoken-language routing for Nemotron's detected locale.

Only Nemotron's explicit English locale + meaningful Latin transcript may bypass
Japanese ASR/translation. Missing, mixed or uncertain results use Japanese.
No classifier is trained on the output text alone.
"""
import re

from ..text_filter import is_usable_transcript

_JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_WORD = re.compile(r"[A-Za-z]+(?:['\u2019-][A-Za-z]+)*")
_NON_SPEECH = re.compile(
    r"^\s*[\[(]?\s*(?:music|instrumental|laughter|applause|silence|noise|"
    r"background music|sound effects?)\s*[\])]?[.!?\s]*$", re.I
)


def detected_locale(value: str | None) -> str:
    if not isinstance(value, str):
        return "unknown"
    label = value.strip().lower().replace("_", "-")
    if label in ("english", "eng") or label == "en" or label.startswith("en-"):
        return "en"
    if label in ("japanese", "jpn") or label == "ja" or label.startswith("ja-"):
        return "ja"
    return "unknown"


def classify_language(
    language: str | None, text: str, duration_s: float, *,
    previous_english: bool = False,
) -> str:
    """Return 'english', 'japanese' or 'uncertain', never guess from script alone."""
    locale = detected_locale(language)
    if locale == "ja":
        return "japanese"
    if locale != "en" or not isinstance(text, str):
        return "uncertain"
    if _JAPANESE.search(text) or _NON_SPEECH.fullmatch(text):
        return "uncertain"
    if not is_usable_transcript(text):
        return "uncertain"
    words = _WORD.findall(text)
    if not words or not any(len(word) >= 3 for word in words):
        return "uncertain"
    # Short one-word exclamations are unreliable language-ID examples.
    # Permit a brief English reply after another nearby English subtitle.
    if previous_english and duration_s >= 0.35:
        return "english"
    if duration_s >= 0.75 and len(words) >= 2:
        return "english"
    if duration_s >= 1.7 and len(words) == 1 and len(words[0]) >= 4:
        return "english"
    return "uncertain"
