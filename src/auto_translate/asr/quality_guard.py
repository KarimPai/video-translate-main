"""Conservative evidence-based guard for clearly broken ASR hypotheses.

Do not reject legitimate short Japanese reactions or infer spoken language from
script alone. Reject clear decoder loops and implausible output while preserving
uncertain but meaningful speech.
"""
import re
import unicodedata
from functools import lru_cache

from ..text_filter import rejection_reason

_JA = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_WORD = re.compile(r"[A-Za-z]{2,}")
_PHANTOM = re.compile(r"^(?:thank you for watching|subtitles? by .+|please subscribe)(?:[.! ]*)$", re.I)
_ROMAJI_REACTIONS = frozenset({
    "hai", "un", "uun", "ee", "eh", "etto", "eto", "ano", "nani", "sugoi",
    "kawaii", "kowai", "yabai", "yatta", "arigatou", "arigato", "daijoubu",
    "daijobu", "sou", "soudane", "honto", "hontou", "uso", "kusa", "otsukare",
})


def _romanized_phrase(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).strip().casefold()
    value = value.strip(' \t\r\n.!?…、。！？"“”「」')
    value = re.sub(r"[\s-]+", " ", value)
    # A narrow exception for names and romanized reactions, not a foreign
    # sentence classifier. Full sentences still require Japanese script.
    if len(value) > 64 or len(value.split()) > 4 or not re.fullmatch(r"[a-z]+(?: [a-z]+)*", value):
        return ""
    return value


@lru_cache(maxsize=1)
def _romanized_names() -> frozenset[str]:
    from ..proper_names import reviewed_spoken_name_forms
    return frozenset(_romanized_phrase(form) for form in reviewed_spoken_name_forms()) - {""}


def _romanized_name(phrase: str) -> bool:
    return bool(phrase) and phrase in _romanized_names()


def potential_japanese_transcript(text: str) -> bool:
    """Keep a narrow Latin candidate for the pipeline's original-audio gate.

    The backend has neither a verified language flag nor conversation metadata.
    A retained Latin candidate is not yet authorized for translation.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    phrase = _romanized_phrase(text)
    return bool(_JA.search(text) or phrase in _ROMAJI_REACTIONS or _romanized_name(phrase))


def supports_japanese_transcript(
    text: str, *, language_decision=None, video_context: str = "", confirmed_participants=(),
) -> bool:
    """Validate Latin exceptions using actual Japanese audio or a named person.

    A generic Hololive title cannot authorize Latin reactions. A name-only cue
    can use an exact reviewed name in participant metadata; this never changes
    the recognized words or expands a nickname.
    """
    if getattr(language_decision, "drop", False) is True or not potential_japanese_transcript(text):
        return False
    if _JA.search(text):
        return True
    probability = getattr(language_decision, "probability", 0.)
    japanese = getattr(language_decision, "japanese_probability", 0.)
    if (getattr(language_decision, "language", None) == "ja"
            and isinstance(probability, (int, float)) and probability >= .75
            and isinstance(japanese, (int, float)) and japanese >= .5):
        return True
    phrase = _romanized_phrase(text)
    if not _romanized_name(phrase):
        return False
    if isinstance(confirmed_participants, str):
        confirmed_participants = (confirmed_participants,)
    contexts = [str(video_context or ""), *(str(value) for value in confirmed_participants)]
    # Japanese metadata contributes only explicitly resolved, spoken forms.
    from ..proper_names import PERSON, resolve_names
    contexts += [mention.target for context in contexts[:]
                 for mention in resolve_names(context)
                 if mention.classification == PERSON and mention.target]
    base = re.sub(r" (?:chan|san|sama|senpai|kun)$", "", phrase)
    return any(re.search(r"(?<![a-z])" + re.escape(base) + r"(?![a-z])",
                         unicodedata.normalize("NFKC", context).casefold())
               for context in contexts)


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
