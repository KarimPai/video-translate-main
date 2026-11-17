"""Conservative corrections for the Watame/Watanabe name confusion."""
import re

_WATAME = re.compile(r"わため|ワタメ|\bWatame\b", re.I)
_WATANABE = re.compile(r"渡辺|渡邊|渡邉|渡部|わたなべ|ワタナベ|\bWatanabe\b", re.I)
_EN_SURNAME = re.compile(r"\bWatanabe\b", re.I)
_NAME = re.compile(f"(?:{_WATAME.pattern})|(?:{_WATANABE.pattern})", re.I)


def reconcile_caption(asr: str, caption: str) -> str:
    """Repair individual mentions only when the entire aligned line agrees.

    The caller must supply a reliable, time-aligned caption. Keep surname
    occurrences the caption also hears as a surname, including mixed lines
    with both Watame and a real Watanabe. Never replace every occurrence just
    because another occurrence has a matching Watame spelling.
    """
    if not _WATANABE.search(asr) or not _WATAME.search(caption):
        return asr
    heard, supported = list(_NAME.finditer(asr)), list(_NAME.finditer(caption))
    if len(heard) != len(supported) or len(heard) > 16:
        return asr
    left, right = 0, 0
    replacements = []
    for source, peer in zip(heard, supported):
        if normalized(asr[left:source.start()]) != normalized(caption[right:peer.start()]):
            return asr
        source_person = _WATAME.fullmatch(source.group()) is not None
        peer_person = _WATAME.fullmatch(peer.group()) is not None
        if source_person and not peer_person:
            return asr  # Contradictory name evidence: abstain from further repairs.
        if not source_person and peer_person:
            replacements.append((source.start(), source.end(), peer.group()))
        left, right = source.end(), peer.end()
    if normalized(asr[left:]) != normalized(caption[right:]):
        return asr
    for start, end, spelling in reversed(replacements):
        asr = asr[:start] + spelling + asr[end:]
    return asr


def correct_asr_name(asr: str, *, vod_title: str = "", caption: str = "") -> str:
    """A title cannot establish a spoken name; require aligned caption support.

    ``vod_title`` remains accepted for existing callers. Pipeline callers
    already withhold captions that fail their reliability/alignment checks.
    """
    return reconcile_caption(asr, caption)


def normalized(text: str) -> str:
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def correct_translation(source: str, english: str) -> str:
    if _WATAME.search(source) and not _WATANABE.search(source):
        if _EN_SURNAME.search(english) and not _WATAME.search(english):
            return _EN_SURNAME.sub("Watame", english)
    return english
