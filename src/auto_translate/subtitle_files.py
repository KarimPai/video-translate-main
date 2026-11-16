"""Explicit subtitle import/export; importing this module never writes a session.

Install the optional ``subtitles`` extra to read or write SRT, WebVTT and ASS.
Corrections match existing cue timestamps rather than replacing source timing.
The caller owns revision history and decides whether to publish a correction.
"""
import math
import os
import tempfile
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .types import Subtitle

Language = Literal["japanese", "english", "bilingual"]
_FORMATS = {".srt": "srt", ".vtt": "vtt", ".ass": "ass"}


@dataclass(frozen=True)
class SubtitleRevision:
    """A proposed correction with independent before/after snapshots."""

    original: Subtitle
    revised: Subtitle
    source: str


def _pysubs2():
    try:
        import pysubs2
    except ImportError as exc:
        raise RuntimeError(
            "Subtitle files require pysubs2; install the 'subtitles' extra "
            "(uv sync --extra subtitles)."
        ) from exc
    return pysubs2


def _format(path: Path) -> str:
    try:
        return _FORMATS[path.suffix.lower()]
    except KeyError as exc:
        raise ValueError("Choose an .srt, .vtt or .ass subtitle file") from exc


def _language(language: str) -> Language:
    if language not in ("japanese", "english", "bilingual"):
        raise ValueError("language must be japanese, english or bilingual")
    return language  # type: ignore[return-value]


def validate_subtitles(subtitles: Iterable[Subtitle]) -> list[Subtitle]:
    """Take independent snapshots and reject invalid or ambiguous cue timing."""
    snapshots = deepcopy(list(subtitles))
    seen = set()
    for cue in snapshots:
        if not isinstance(cue, Subtitle):
            raise TypeError("Expected Subtitle objects")
        if not (math.isfinite(cue.start_s) and math.isfinite(cue.end_s)):
            raise ValueError("Subtitle timestamps must be finite")
        if cue.start_s < 0 or cue.end_s <= cue.start_s:
            raise ValueError("Subtitle timestamps must satisfy 0 <= start < end")
        if not isinstance(cue.ja, str) or not isinstance(cue.en, str):
            raise TypeError("Subtitle text must be a string")
        if "\x00" in cue.ja or "\x00" in cue.en:
            raise ValueError("Subtitle text cannot contain NUL characters")
        key = _key(cue.start_s, cue.end_s)
        if key[1] <= key[0]:
            raise ValueError("Subtitle is shorter than file timestamp precision")
        if key in seen:
            raise ValueError("Duplicate subtitle timestamps are ambiguous")
        seen.add(key)
    return sorted(snapshots, key=lambda cue: (cue.start_s, cue.end_s))


def _key(start: float, end: float) -> tuple[int, int]:
    return round(start * 1000), round(end * 1000)


def _file_key(start: float, end: float, format_: str) -> tuple[int, int]:
    a, b = _key(start, end)
    # pysubs2 follows Aegisub's half-up centisecond rounding for ASS.
    return ((a + 5) // 10, (b + 5) // 10) if format_ == "ass" else (a, b)


def _text(cue: Subtitle, language: Language) -> str:
    if language == "japanese":
        text = cue.ja
    elif language == "english":
        text = cue.en
    else:
        if not cue.ja.strip() or "\n" in cue.ja.strip() or "\r" in cue.ja.strip():
            raise ValueError("Bilingual files require one nonempty Japanese line per cue; export languages separately otherwise")
        text = "\n".join(part.strip() for part in (cue.ja, cue.en) if part.strip())
    # ASS interprets braces and backslashes as commands, including when pysubs2
    # converts other formats. Match the player's plain-text escaping convention.
    return (text.strip().replace("\r\n", "\n").replace("\r", "\n")
            .replace("\\", "＼").replace("{", "｛").replace("}", "｝"))


def export_subtitles(
    path: str | Path,
    subtitles: Iterable[Subtitle],
    language: Language = "bilingual",
) -> int:
    """Atomically export a snapshot, returning the number of nonempty cues.

    ASS has centisecond precision; SRT and WebVTT have millisecond precision.
    Plain-text braces/backslashes are exported as fullwidth characters, just as
    in the player, so transcript text cannot inject ASS styling or drawings.
    The destination directory must already exist. Existing files are replaced
    only after the complete new file has been serialized, flushed and fsynced.
    """
    path = Path(path)
    format_ = _format(path)
    language = _language(language)
    cues = validate_subtitles(subtitles)
    pysubs2 = _pysubs2()
    document = pysubs2.SSAFile()
    file_keys = set()
    for cue in cues:
        text = _text(cue, language)
        if not text:
            continue
        start, end = _key(cue.start_s, cue.end_s)
        key = _file_key(cue.start_s, cue.end_s, format_)
        if key[0] >= key[1]:
            raise ValueError("Subtitle is shorter than file timestamp precision")
        if key in file_keys:
            raise ValueError("Cue timestamps collide at file timestamp precision")
        file_keys.add(key)
        limit_ms = 36_000_000 - 10 if format_ == "ass" else 360_000_000 - 1
        if end > limit_ms:
            raise ValueError("Subtitle timestamp exceeds this file format's supported duration")
        event = pysubs2.SSAEvent(start=start, end=end)
        event.plaintext = text
        document.events.append(event)
    if not document.events:
        raise ValueError("There are no nonempty subtitles in the selected language")
    contents = document.to_string(format_)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", delete=False,
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return len(document.events)


def _read(path: Path, language: Language) -> list[Subtitle]:
    pysubs2 = _pysubs2()
    document = pysubs2.load(str(path), encoding="utf-8-sig", format_=_format(path))
    cues = []
    for event in document.events:
        if event.is_comment or event.is_drawing:
            continue
        text = event.plaintext.strip()
        if not text:
            continue
        if language == "bilingual":
            # Explicit contract: bilingual correction files have one Japanese
            # line followed by optional English lines. Multi-line Japanese
            # should be imported separately to avoid guessing a language split.
            ja, _separator, en = text.partition("\n")
        else:
            ja, en = (text, "") if language == "japanese" else ("", text)
        cues.append(Subtitle(event.start / 1000, event.end / 1000, ja, en))
    if not cues:
        raise ValueError("The subtitle file contains no usable text cues")
    return validate_subtitles(cues)


def import_corrections(
    path: str | Path,
    originals: Iterable[Subtitle],
    language: Language = "english",
) -> list[SubtitleRevision]:
    """Propose corrections without changing originals or their timing/metadata.

    Imported files may contain a subset of cues, but every imported cue must
    match an existing cue. ASS matches centiseconds; SRT/VTT milliseconds.
    Ambiguous timestamp matches or retimed files are rejected as a whole.
    Revisions contain only changed cues, in chronological order.
    """
    path = Path(path)
    language = _language(language)
    snapshots = validate_subtitles(originals)
    imported = _read(path, language)
    format_ = _format(path)
    by_time = {}
    for cue in snapshots:
        key = _file_key(cue.start_s, cue.end_s, format_)
        if key in by_time:
            raise ValueError("Original cue timestamps are ambiguous at file precision")
        by_time[key] = cue
    revisions = []
    for correction in imported:
        key = _file_key(correction.start_s, correction.end_s, format_)
        if key not in by_time:
            raise ValueError("Correction timestamps do not match existing subtitles; timing was not changed")
        original = by_time[key]
        revised = deepcopy(original)
        if language in ("japanese", "bilingual"):
            revised.ja = correction.ja
        if language in ("english", "bilingual"):
            revised.en = correction.en
        if revised.ja != original.ja or revised.en != original.en:
            revisions.append(SubtitleRevision(deepcopy(original), revised, str(path)))
    return revisions


def import_subtitles(
    path: str | Path,
    originals: Iterable[Subtitle] | None = None,
    language: Language = "english",
) -> list[Subtitle]:
    """Read standalone cues, or return a corrected independent full snapshot.

    Use :func:`import_corrections` when the UI needs explicit before/after
    revisions. Nothing is applied to a live pipeline or written automatically.
    """
    path = Path(path)
    language = _language(language)
    if originals is None:
        return _read(path, language)
    snapshots = validate_subtitles(originals)
    revisions = import_corrections(path, snapshots, language)
    changes = {_key(r.original.start_s, r.original.end_s): r.revised for r in revisions}
    return [changes.get(_key(cue.start_s, cue.end_s), cue) for cue in snapshots]
