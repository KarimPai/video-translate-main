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
    r"てない|くない|なかった|ません|無理|禁止|駄目|だめ|"
    r"(?:行か|行け|食べ|飲ま|見え|聞こえ|分から|わから|勝て|負け)ない"
)
_EN_NEGATION = re.compile(
    r"\b(?:not|no|never|can't|cannot|don't|doesn't|didn't|"
    r"isn't|aren't|wasn't|weren't|won't|wouldn't|couldn't|"
    r"shouldn't|mustn't|unable|impossible|forbidden|prohibited|"
    r"bad|wrong|without|neither|nor|nothing|nobody|nowhere)\b",
    re.IGNORECASE,
)
# Anchor to a complete favorable appraisal. Praise can also modify something
# negative (すごく悪い), and やばい alone does not establish polarity.
_PRAISE = re.compile(
    r"(?:これ|それ|あれ)?(?:は|って)?(?:本当に|ほんとに|めっちゃ|めちゃくちゃ)?"
    r"(?:すごい|すごかった|凄い|凄かった|最高|素晴らしい|上手い|うまい|楽しかった)"
    r"(?:だった|だ|です)?(?:ね|よ|な|わ|よね)?"
)
_CRITICISM = re.compile(r"\b(?:bad|terrible|awful|horrible|lousy|hated?|sucks?)\b", re.I)
_POSITIVE_IDIOM = re.compile(
    r"\b(?:not bad|no way|no joke|nothing short of|can.t believe|couldn.t be better)\b",
    re.I,
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
    english = english.replace("’", "'")
    source = unicodedata.normalize("NFKC", japanese)
    appraisal = source.strip(" !?。、.…")
    if (_PRAISE.fullmatch(appraisal) and "?" not in source
            and not _POSITIVE_IDIOM.search(english)
            and (_CRITICISM.search(english) or re.search(
                r"\b(?:not|isn't|wasn't)\s+(?:that |very )?(?:good|great|fun|impressive|amazing)\b",
                english, re.I,
            ))):
        return "positive_appraisal_reversed"
    if (re.fullmatch(r"(?:できる|出来る|勝てる|行ける|見える|聞こえる)(?:よ|ね|よね)?", appraisal)
            and "?" not in source and re.search(r"\b(?:can't|cannot|unable to)\b", english, re.I)):
        return "affirmative_ability_reversed"
    # Context-specific meaning reversals seen in live ARK subtitles.
    if (re.search(r"読み込み.{0,10}軽くな|軽くなっちま|ロード.{0,8}軽くな", source)
            and re.search(r"\b(?:slowed?|slower|sluggish|heavier)\b", english, re.I)):
        return "loading_performance_reversed"
    if (re.search(r"(?:のん[、\s]?){2,}|(?:ノン[、\s]?){2,}", source)
            and re.search(r"\b(?:fine|okay|all right|all good|no problem)\b", english, re.I)
            and not re.search(r"\b(?:no|not|never)\b", english, re.I)):
        return "rejection_reversed"
    if _EXPLICIT_NEGATION.search(japanese) and not _EN_NEGATION.search(english):
        # Obligations such as "寝ないといけない" should not be forced negative.
        if not re.search(r"ないと(?:いけない|ならない|ダメ|だめ)|なければならない|なくちゃ|なくても(?:いい|大丈夫)|かもしれない", japanese):
            return "possible_missing_negation"
    return None

def repair_high_confidence_meaning(japanese: str, english: str) -> str:
    """Fix two tightly bounded reversals found in real Japanese stream output.

    Applies AFTER optional PLaMo quality review, so a rejected or timed-out
    model retry cannot leave these known meaning inversions on screen.
    Does not attempt to repair arbitrary garbled ASR sentences.
    """
    source = unicodedata.normalize("NFKC", japanese)
    if re.search(r"読み込み.{0,8}軽くな(?:っ|った|る|い)", source):
        # Japanese says loading became lighter, *not* that it slowed down.
        english = re.sub(
            r"\b(?:the\s+)?loading\s+has\s+slowed\s+down\b",
            "The load is lighter now",
            english, count=1, flags=re.I,
        )
    if (
        re.fullmatch(r"\s*(?:レベル)?95は(?:のん){2,}(?:よ)?[。!?！。\s]*", source)
        and re.fullmatch(r"\s*95\s+is\s+(?:totally\s+)?fine[.!?]?\s*", english, re.I)
    ):
        return "Level 95? No, no, no."
    return english
