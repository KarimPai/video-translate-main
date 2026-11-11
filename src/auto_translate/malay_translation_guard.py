"""Conservative direct Japanese-to-Standard-Malay output validation."""
import re
import unicodedata
from collections import Counter

from .subtitle_clean import has_generation_loop, looks_like_prompt_echo

_NUM = re.compile(r"\d+(?:\.\d+)?")
_JA_NEGATIVE = re.compile(
    r"できない|出来ない|知らない|分からない|わからない|"
    r"いらない|要らない|じゃない|ではない|行かない|"
    r"欲しくない|無理"
)
_MS_NEGATIVE = re.compile(
    r"\b(?:tidak|tak|bukan|belum|jangan|tiada|tanpa|mustahil|"
    r"enggan|gagal|takkan|susah)\b", re.I,
)
_GUARDED_NAMES = frozenset({"hololive", "watame", "pekora", "miko", "korone", "subaru"})


def suspicious_malay_translation(
    japanese: str, malay: str, *, names: list[tuple[str, str]] | None = None,
) -> str | None:
    """Return one clear reason to retry, or None. Avoid English-only heuristics."""
    text = malay.strip()
    if not text:
        return "empty_malay"
    if has_generation_loop(text) or looks_like_prompt_echo(text):
        return "prompt_echo_or_loop"
    jp = unicodedata.normalize("NFKC", japanese).strip()
    ms = unicodedata.normalize("NFKC", text)
    if len(jp) <= 24 and _JA_NEGATIVE.search(jp) and not _MS_NEGATIVE.search(ms):
        return "possible_negation_loss"
    src_numbers = _NUM.findall(jp)
    out_numbers = _NUM.findall(ms)
    if src_numbers and out_numbers and Counter(src_numbers) != Counter(out_numbers):
        return "numeric_mismatch"
    for source, target in names or []:
        if target.casefold() in _GUARDED_NAMES and source in jp:
            if not re.search(r"(?<![a-z])" + re.escape(target) + r"(?![a-z])", ms, re.I):
                return "proper_name_mismatch"
    return None
