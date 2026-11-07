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
        # The same name may occur several times in one short utterance.
        candidate = _WATANABE.sub(match.group(), asr)
        if normalized(candidate) == normalized(caption):
            return candidate
    return asr


def correct_asr_name(asr: str, *, vod_title: str = "", caption: str = "") -> str:
    """Correct likely Watame homophones when the VOD title identifies her.

    Automatic Japanese captions can repeat the same kanji surname mistake.
    A caption explicitly spelling Watanabe phonetically, however, wins.
    Outside Watame-titled videos, retain the ASR name unless a matching
    caption already reconciled it.
    """
    if not _WATANABE.search(asr) or not _WATAME.search(vod_title):
        return asr
    if re.search(r"わたなべ|ワタナベ|\\bWatanabe\\b", caption, re.I):
        return asr
    return _WATANABE.sub("わため", asr)


def normalized(text: str) -> str:
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def correct_translation(source: str, english: str) -> str:
    if _WATAME.search(source) and not _WATANABE.search(source):
        if len(_EN_SURNAME.findall(english)) == 1 and not _WATAME.search(english):
            return _EN_SURNAME.sub("Watame", english)
    return english
