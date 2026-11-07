"""Conservative corrections for the Watame/Watanabe name confusion."""
import re

_WATAME = re.compile(r"わため|ワタメ|\bWatame\b", re.I)
_WATANABE = re.compile(r"渡辺|渡邊|渡邉|渡部|わたなべ|ワタナベ|\bWatanabe\b", re.I)
_EN_SURNAME = re.compile(r"\bWatanabe\b", re.I)


def reconcile_caption(asr: str, caption: str) -> str:
    """Accept the caption spelling only if the rest of the utterance agrees."""
    if not _WATANABE.search(asr) or _WATANABE.search(caption):
        return asr
    for match in _WATAME.finditer(caption):
        candidate = _WATANABE.sub(match.group(), asr, count=1)
        if normalized(candidate) == normalized(caption):
            return candidate
    return asr


def normalized(text: str) -> str:
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def correct_translation(source: str, english: str) -> str:
    if _WATAME.search(source) and not _WATANABE.search(source):
        if len(_EN_SURNAME.findall(english)) == 1 and not _WATAME.search(english):
            return _EN_SURNAME.sub("Watame", english)
    return english
