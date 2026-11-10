"""Conservative display-only quality checks for Anime Whisper JP captions.

Do not replace, paraphrase or infer Japanese names from a speculative lexicon.
Only skip pure laughter or a repeated exact stock outro; spoken dialogue is kept.
"""
from __future__ import annotations

import re
import unicodedata


_PUNCTUATION = " \t\r\n、。．,.!?！？…ー〜~～・「」『』（）()♪♬"
_LAUGHTER = re.compile(
    r"(?:あ?は){2,}|(?:はっ){2,}は?|(?:ふ){2,}|(?:へ){2,}|(?:えへ){2,}|(?:[wｗ]){2,}",
    re.IGNORECASE,
)
_OUTROS = frozenset((
    "ご視聴ありがとうございました",
    "最後までご視聴ありがとうございました",
    "チャンネル登録よろしくお願いします",
))


def _plain(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    # Match katakana laughter as well as hiragana, not ordinary JP words.
    normalized = normalized.translate(str.maketrans("アハフヘッ", "あはふへっ"))
    return normalized.strip(_PUNCTUATION)


def is_standalone_laughter(text: str) -> bool:
    """Match only a complete isolated laugh, not words mixed with laughter."""
    return bool(_LAUGHTER.fullmatch(_plain(text)))


def formulaic_japanese_outro_key(text: str) -> str | None:
    """Return normalized exact stock phrase; punctuation variants compare equal."""
    value = _plain(text)
    return value if value in _OUTROS else None


def is_formulaic_japanese_outro(text: str) -> bool:
    """Recognize exact stock outros but never delete their first occurrence."""
    return formulaic_japanese_outro_key(text) is not None
