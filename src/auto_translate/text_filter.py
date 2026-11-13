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


def rejection_reason(text: str) -> str | None:
    """Return why an ASR result should be dropped, or None if it looks usable.

    Deliberately conservative: short reactions and speech containing laughter
    are retained. Explicit nonspeech labels, chat markers and obvious decoder
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
        # Spoken laughter is evidence of a reaction. Keep explicit sound
        # labels and text-only laughter markers out of spoken subtitles.
        return "laughter_only" if re.fullmatch(r"[w笑草]+", compact) else None

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

    # Whisper Turbo can loop a complete kana phrase throughout a cue.
    # Require six identical copies with no other words to preserve emphasis.
    if len(compact) >= 24 and re.fullmatch(r"([ぁ-ヿ]{4,16})\1{5,}", compact):
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
