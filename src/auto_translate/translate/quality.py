"""Cheap, conservative signals for selective PLaMo translation review.

Do not classify unfamiliar romanized words as gibberish: they may be VTuber
names or game terminology. Only skip Japanese with clear syllable loops and
only trigger a second model call for suspicious *outputs*.
"""

import re
import unicodedata

SKIP_TOKEN = "[SKIP]"

_KANJI = re.compile(r"[\u3400-\u9fff]")
_KANA = re.compile(r"[\u3040-\u30ff]")
_JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_NON_LETTERS = re.compile(r"[\s、。，．・!?！？…〜～.,:;:「」『』（）()\[\]【】ー]+")
_KANA_LOOP = re.compile(r"([ぁ-ん])\1{3,}")
_KANA_PHRASE_LOOP = re.compile(r"([ぁ-ん]{1,3})\1{2,}")
_ROMANIZED_LOOP = re.compile(
    r"(?i)(?<![a-z])([a-z]{1,4})(?:[\s-]*\1){2,}(?![a-z])"
)
_HYPHENATED_PHONETIC = re.compile(
    r"(?i)^(?:[a-z]{2,8}-){2,}[a-z]{2,8}[!?.,]*$"
)
_EXPLICIT_NEGATION = re.compile(
    r"じゃない|ではない|できない|出来ない|してない|しない|"
    r"てない|くない|なかった|ません|無理|禁止|駄目|だめ"
)
_EN_NEGATION = re.compile(
    r"\b(?:not|no|never|can't|cannot|don't|doesn't|didn't|"
    r"isn't|aren't|wasn't|weren't|won't|wouldn't|couldn't|"
    r"shouldn't|mustn't|unable|impossible|forbidden|prohibited|"
    r"bad|wrong|without|neither|nor|nothing|nobody|nowhere)\b",
    re.IGNORECASE,
)
# These are ordinary English repetitions, not romanization artifacts.
_MEANINGFUL_REPETITIONS = frozenset(
    {"go", "no", "yes", "stop", "wait", "come", "run", "nice", "hey", "wow",
     "okay", "fire", "help", "look", "here", "that", "what", "this", "yeah",
     "good", "turn", "left", "back", "move", "roll"}
)


def is_skip_token(text: str) -> bool:
    """Allow only an entire exact sentinel, never a substring of dialogue."""
    return text.strip().upper() == SKIP_TOKEN


def is_obvious_nonverbal_japanese(text: str) -> bool:
    """Detect strong phonetic loops, not general short or unusual Japanese."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = "".join(
        chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c
        for c in normalized
    )
    compact = _NON_LETTERS.sub("", normalized)
    if (
        len(compact) < 5
        or not _KANA.search(compact)
        or _KANJI.search(compact)
        or any(c.isascii() and c.isalnum() for c in compact)
    ):
        return False
    single_loop = _KANA_LOOP.search(compact)
    if single_loop:
        # "ここここ" and "そそそそ" can be meaningful deictic speech.
        return single_loop.group(1) not in {"こ", "そ"}
    matched = _KANA_PHRASE_LOOP.search(compact)
    if matched is None or len(matched.group()) < 0.65 * len(compact):
        return False
    # Repeated lexical words are meaningful speech, not nonsensical syllables.
    unit = matched.group(1)
    meaningful = {
        "ここ", "そこ", "そう", "はい", "うん", "いい", "いや",
        "いけ", "こい", "まて", "やめ", "だめ", "すご", "よし",
    }
    return len(unit) <= 2 and unit not in meaningful


def translation_review_reason(japanese: str, english: str) -> str | None:
    """Return a high-signal issue worth ONE bounded retry, or None."""
    if is_skip_token(english):
        return "unexpected_skip"
    if not english.strip():
        return "empty_english"
    if _JAPANESE.search(english):
        return "untranslated_japanese"
    loop = _ROMANIZED_LOOP.search(english)
    if loop and loop.group(1).lower() not in _MEANINGFUL_REPETITIONS:
        return "romanized_syllable_loop"
    if (
        _HYPHENATED_PHONETIC.fullmatch(english.strip())
        and not _KANJI.search(japanese)
    ):
        return "phonetic_transliteration"
    if _EXPLICIT_NEGATION.search(japanese) and not _EN_NEGATION.search(english):
        # Obligations such as "寝ないといけない" should not be forced negative.
        if not re.search(r"ないと(?:いけない|ならない|ダメ|だめ)", japanese):
            return "possible_missing_negation"
    return None
