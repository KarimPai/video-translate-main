"""Conservative post-ASR filtering for non-linguistic Japanese output."""

import re
import unicodedata

# Common Japanese laughter spellings and chat-style laughter.
_LAUGHTER = re.compile(
    r"(?:あ?は){2,}|あは+|わは+|ふふ+|へへ+|ほほ+|くく+|"
    r"ｗ{2,}|w{3,}|笑+|草+",
    re.IGNORECASE,
)
_NON_TEXT = re.compile(r"[\s、。，．・！？!?…〜～.,:;()（）「」『』【】\[\]{}<>]+")
_JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_KANA = re.compile(r"[\u3040-\u30ff]+")


def _compact(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _NON_TEXT.sub("", normalized)



# Remove caption/ASR sound-effect labels, not actual spoken words.
_SOUND_WORDS = (
    r"applause|clapping|claps?|cheering|crowd cheering|"
    r"music|background music|bgm|instrumental|singing|"
    r"sound effects?|sfx|laughter|laughing|laughs|"
    r"拍手|拍手喝采|歓声|声援|音楽|背景音楽|効果音|笑い声|笑い|爆笑"
)
_SOUND_BRACKET = re.compile(
    rf"[\[\(（【［]\s*(?:{_SOUND_WORDS})\s*[\]\)）】］]",
    re.IGNORECASE,
)
_SOUND_ONLY = re.compile(rf"^(?:{_SOUND_WORDS})$", re.IGNORECASE)
_MUSIC_MARKS = re.compile(r"[♪♫♬♩]+")
_SPACES = re.compile(r"\s+")


def strip_nonverbal_labels(text: str) -> str:
    """Remove explicit sound-event labels while preserving adjacent speech."""
    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = _SOUND_BRACKET.sub(" ", cleaned)
    cleaned = _MUSIC_MARKS.sub(" ", cleaned)
    cleaned = _SPACES.sub(" ", cleaned).strip()
    if _SOUND_ONLY.fullmatch(cleaned):
        return ""
    return cleaned

def rejection_reason(text: str) -> str | None:
    """Return why an ASR result should be dropped, or None if it looks usable.

    Deliberately conservative: short reactions and speech containing laughter
    are retained. Only laughter-only output and obvious repeated-character
    loops are rejected.
    """
    compact = _compact(text)
    if not compact:
        return "empty_after_normalization"

    label = unicodedata.normalize("NFKC", text).strip().casefold()
    if re.fullmatch(r"(?:\[|\(|【)\s*(?:笑い声|笑い|爆笑|laughter|laughs)\s*(?:\]|\)|】)", label):
        return "laughter_only"

    # Haas may spell the same laugh in hiragana or katakana.
    laugh_kana = "".join(chr(ord(ch) - 0x60) if "ァ" <= ch <= "ヶ" else ch for ch in compact)
    without_laughter = _LAUGHTER.sub("", laugh_kana)
    if compact and not without_laughter:
        return "laughter_only"

    # A single kana repeated many times is usually a vocalization or decoder
    # loop, not useful lexical content. Keep short reactions such as 「ええ」.
    if len(compact) >= 6 and len(set(compact)) == 1:
        return "repeated_character"

    # Detect obvious 1-3 character kana loops (e.g. あいあいあいあい).
    # Require at least four repetitions to avoid rejecting normal emphasis.
    if len(compact) >= 8 and _KANA.fullmatch(compact):
        for width in (1, 2, 3):
            unit = compact[:width]
            if len(compact) % width == 0 and compact == unit * (len(compact) // width):
                if unit in ("はい", "うん"):
                    return None
                return "repeated_kana_loop"

    # Filter long outputs containing no Japanese text only when they are
    # mostly punctuation/symbol noise. Legitimate English names are retained.
    visible = [ch for ch in compact if ch.isalnum()]
    if len(visible) >= 8 and not _JAPANESE.search(compact):
        if sum(ch.isalpha() for ch in visible) / len(visible) < 0.5:
            return "symbol_noise"

    return None


def is_usable_transcript(text: str) -> bool:
    return rejection_reason(text) is None


# Aggressive suppression is applied in run_asr *after* caption reconciliation.
# Restrict it to whole utterances so speech such as "うわー、これすごい" survives.
_REACTION_SEPARATORS = re.compile(
    r"[\s、。，．・!?！？…〜～,.;:：~「」『』（）()\[\]【】]+"
)
_REACTION_RUN = re.compile(
    r"(?:"
    r"あ+[ぁー]*っ?|は+[ぁー]*っ?|え+[ぇー]*っ?|"
    r"お+[ぉー]*っ?|う+[ぅー]*っ?|うわ[あぁー]*っ?|"
    r"わ[あぁー]+|ふ[うぅー]+|へ[えぇー]+|ほ[おぉー]+|"
    r"う[うぅー]*ん+|ん+[ー]?|"
    r"よいしょ|よっこいしょ|え[ー]*っと|ええっと|あの[ー]*|はー+い|"
    r"すっ?ご[いぃっー]+|すっ?げ[えぇー]+|すすげ[えぇー]+|"
    r"やば[いぁっー]+|はい+|わお+|くっ|ぐっ|うぐ|"
    r"ah+|oh+|ooh+|uh+|um+|hm+|hmm+|huh+|"
    r"whoa+|woah+|wow+|hey+|yeah+|yep+|yup+|ugh+|sigh+"
    r")+"
)


def is_isolated_reaction(text: str) -> bool:
    """Return True only when the entire utterance is filler/interjections.

    Aggressive mode intentionally drops isolated acknowledgements ("うん",
    "はい") and praise ("すごい"). Speech containing any other words is
    preserved in full, including when an interjection begins the sentence.
    """
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = "".join(
        chr(ord(char) - 0x60) if "ァ" <= char <= "ヶ" else char
        for char in normalized
    )
    tokens = [part for part in _REACTION_SEPARATORS.split(normalized) if part]
    return bool(tokens) and all(_REACTION_RUN.fullmatch(part) for part in tokens)


def is_whole_utterance_blocked(text: str, blocklist: list[str]) -> bool:
    """Match a custom Haas blocklist item exactly, ignoring punctuation.

    Do not reject an otherwise meaningful sentence merely because it contains
    a blocked filler word.
    """
    normalized = _compact(text)
    return bool(normalized) and any(
        normalized == _compact(phrase) for phrase in blocklist if phrase.strip()
    )
