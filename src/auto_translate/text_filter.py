"""Conservative post-ASR filtering for non-linguistic Japanese output."""

import re
import unicodedata

KATA_TO_HIRA = {code: code - 0x60 for code in range(ord("ァ"), ord("ヶ") + 1)}

# Common Japanese laughter spellings and chat-style laughter.
_LAUGHTER = re.compile(
    r"(?:あ?は){2,}|あは+|わは+|ふふ+|へへ+|ほほ+|くく+|"
    r"ｗ{2,}|w{3,}|笑+|草+",
    re.IGNORECASE,
)
_NON_TEXT = re.compile(r"[\s、。，．・！？!?…〜～.,:;()（）「」『』【】\[\]{}<>]+")
_JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_KANA = re.compile(r"[\u3040-\u30ff]+")


_TAIL = re.compile(r"([ぁ-ヺー一-鿿]{1,12}?)\1{7,}[。！？!?]*$")


def trim_repetition_tail(text: str) -> str:
    """Trim only a long contiguous decoder loop after independent dialogue."""
    match = _TAIL.search(text)
    if not match or len(match[0].rstrip("。！？!?")) < 18:
        return text
    prefix = text[:match.start()].rstrip()
    if len(_JAPANESE.findall(prefix)) < 3 or match[1] in ("はい", "うん", "待って", "いや", "だめ"):
        return text
    return prefix


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

    # A single spoken 草 may be the noun “grass”, including a one-word answer.
    if compact == "草":
        return None

    # The decoder may spell the same laugh in hiragana or katakana.
    laugh_kana = compact.translate(KATA_TO_HIRA)
    without_laughter = _LAUGHTER.sub("", laugh_kana)
    if compact and not without_laughter:
        # Entire utterance is laughter — no translatable content.
        # Text-only markers (w, 笑, 草) and spoken laughter (あははは, わはは, etc.)
        # are both omitted because they carry no semantic meaning for subtitles.
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

    # An ASR decoder can loop a complete kana phrase throughout a cue.
    # Require six identical copies with no other words to preserve emphasis.
    if len(compact) >= 24 and re.fullmatch(r"([ぁ-ヿ]{4,16})\1{5,}", compact):
        return "repeated_kana_loop"

    # Filter long outputs containing no Japanese text only when they are
    # mostly punctuation/symbol noise. Legitimate English names are retained.
    visible = [ch for ch in compact if ch.isalnum()]
    if len(visible) >= 8 and not _JAPANESE.search(compact) and sum(ch.isalpha() for ch in visible) / len(visible) < 0.5:
        return "symbol_noise"

    return None


def is_usable_transcript(text: str) -> bool:
    return rejection_reason(text) is None


# ---------------------------------------------------------------------------
# Spoken-language filter: distinguishes Japanese dialogue from non-Japanese
# ---------------------------------------------------------------------------

# Character ranges for Japanese scripts
_JAPANESE_CHARS = set(range(0x3040, 0x3100)) | set(range(0x30A0, 0x3100))  # Hiragana + Katakana
_JAPANESE_CHARS |= set(range(0x4E00, 0x9FFF)) | set(range(0x3400, 0x4DBF))  # CJK ideographs
_JAPANESE_CHARS |= {0x3002, 0x3001, 0xFF01, 0xFF0C, 0xFF0E, 0xFF0F}  # Japanese punctuation


def _japanese_ratio(text: str) -> float:
    """Return the fraction of Japanese-script characters in *text*.

    Uses Unicode block analysis — no external model required.
    """
    if not text:
        return 0.0
    jp_count = sum(1 for ch in text if ord(ch) in _JAPANESE_CHARS)
    visible = sum(1 for ch in text if not unicodedata.category(ch).startswith("C"))
    if visible == 0:
        return 0.0
    return jp_count / visible


class LanguageFilter:
    """CPU-only spoken-language gate that drops non-Japanese utterances.

    Uses Unicode character analysis to distinguish Japanese dialogue from
    English or Indonesian speech. Does not require external models.

    When the user enables language filtering but the ASR output contains
    sufficient Japanese script, the subtitle passes. Otherwise it is dropped.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.min_duration_s = cfg.min_duration_s
        self.max_window_s = cfg.max_window_s
        self.drop_probability = cfg.drop_probability
        self.max_japanese_probability = cfg.max_japanese_probability
        self.min_top_margin = cfg.min_top_margin

    def should_keep(self, text: str, duration_s: float = 0.0) -> bool:
        """Return True if the utterance should be kept (is Japanese dialogue).

        Short utterances below min_duration_s are always kept to avoid
        dropping brief Japanese reactions. Longer utterances are classified
        by their Japanese character ratio.
        """
        # Short utterances: keep them (likely genuine reactions)
        if duration_s < self.min_duration_s:
            return True

        ratio = _japanese_ratio(text)
        # Keep if Japanese ratio is above threshold; drop non-Japanese speech.
        return ratio >= (1.0 - self.max_japanese_probability)

    def filter(self, text: str, duration_s: float = 0.0) -> str | None:
        """Return *text* if it should be kept, or None if it should be dropped."""
        if self.should_keep(text, duration_s):
            return text
        return None
