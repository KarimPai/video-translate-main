"""Japanese YouTube auto-caption download, parsing and timestamp matching."""

import html
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .auth import cookie_args
from .config import IngestConfig

log = logging.getLogger(__name__)

_TIMESTAMP = re.compile(
    r"(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})"
)
_TIMING_LINE = re.compile(r"^\s*(\S+)\s+-->\s+(\S+)")
_TAG = re.compile(r"<[^>]*>")
_SPACE = re.compile(r"\s+")


def _seconds(value: str) -> float | None:
    match = _TIMESTAMP.fullmatch(value.strip())
    if not match:
        return None
    hours, minutes, seconds, millis = match.groups()
    return (
        int(hours or 0) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis) / 1000
    )


def _clean_caption(lines: list[str]) -> str:
    value = " ".join(lines)
    value = _TAG.sub("", value)
    value = html.unescape(value)
    value = value.replace("\u200b", "").replace("\ufeff", "")
    return _SPACE.sub(" ", value).strip()


@dataclass(frozen=True)
class CaptionCue:
    start_s: float
    end_s: float
    text: str


@dataclass(frozen=True)
class CaptionMatch:
    text: str
    overlap_s: float
    coverage: float
    reliable: bool


class CaptionIndex:
    """Small immutable VTT index matched against source-timeline timestamps."""

    def __init__(self, cues: list[CaptionCue]) -> None:
        self.cues = sorted(cues, key=lambda cue: (cue.start_s, cue.end_s))

    @classmethod
    def from_vtt(cls, content: str) -> "CaptionIndex":
        cues: list[CaptionCue] = []
        for block in re.split(r"\n\s*\n", content.replace("\r\n", "\n").replace("\r", "\n")):
            lines = block.splitlines()
            timing_index = next(
                (i for i, line in enumerate(lines) if "-->" in line),
                None,
            )
            if timing_index is None:
                continue
            timing = _TIMING_LINE.match(lines[timing_index])
            if timing is None:
                continue
            start = _seconds(timing.group(1))
            end = _seconds(timing.group(2))
            if start is None or end is None or end <= start:
                continue
            text = _clean_caption(lines[timing_index + 1:])
            if not text:
                continue
            cues.append(CaptionCue(start, end, text))
        return cls(cues)

    @classmethod
    def from_file(cls, path: Path) -> "CaptionIndex":
        return cls.from_vtt(path.read_text(encoding="utf-8-sig"))

    def match(
        self,
        start_s: float,
        end_s: float,
        tolerance_s: float = 0.15,
    ) -> CaptionMatch | None:
        duration = max(0.01, end_s - start_s)
        matched: list[tuple[CaptionCue, float]] = []
        for cue in self.cues:
            if cue.end_s < start_s - tolerance_s:
                continue
            if cue.start_s > end_s + tolerance_s:
                break
            overlap = max(
                0.0,
                min(end_s, cue.end_s) - max(start_s, cue.start_s),
            )
            if overlap > 0:
                matched.append((cue, overlap))

        if not matched:
            return None

        # Avoid repeating rolling-caption text when adjacent cues overlap.
        texts: list[str] = []
        seen: set[str] = set()
        for cue, _ in matched:
            key = _SPACE.sub("", cue.text).casefold()
            if key and key not in seen:
                seen.add(key)
                texts.append(cue.text)

        if not texts:
            return None

        overlap_s = min(duration, sum(overlap for _, overlap in matched))
        coverage = overlap_s / duration
        strongest = max(overlap for _, overlap in matched)
        reliable = (
            (coverage >= 0.30 and strongest >= 0.25)
            or (overlap_s >= 0.75 and strongest >= 0.40)
        )
        return CaptionMatch(
            text=" ".join(texts),
            overlap_s=overlap_s,
            coverage=coverage,
            reliable=reliable,
        )


def _caption_cache_dir() -> Path:
    root = os.environ.get("LOCALAPPDATA")
    base = Path(root) if root else Path.home() / ".cache"
    return base / "auto-translate" / "timelines"


def download_vod_captions(
    video_id: str,
    url: str,
    cfg: IngestConfig,
) -> CaptionIndex | None:
    """Download/cache Japanese auto-captions; never raises for unavailable captions."""
    cache_dir = _caption_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{video_id}.ja.vtt"

    if cached.is_file():
        try:
            index = CaptionIndex.from_file(cached)
            if index.cues:
                log.info(
                    "Japanese YouTube captions loaded from cache (%d cues)",
                    len(index.cues),
                )
                return index
        except (OSError, UnicodeError):
            log.warning("could not read cached captions: %s", cached)

    command = [
        "yt-dlp",
        "--no-warnings",
        "--no-call-home",
        "--skip-download",
        "--write-auto-subs",
        "--sub-langs", "ja.*,ja",
        "--sub-format", "vtt/best",
        "--output", str(cache_dir / f"{video_id}.%(ext)s"),
    ]
    command += cookie_args(cfg)
    command.append(url)

    try:
        with tempfile.TemporaryDirectory(
            prefix=f"captions-{video_id}-",
            dir=str(cache_dir),
        ) as temp_dir:
            command[command.index("--output") + 1] = str(
                Path(temp_dir) / f"{video_id}.%(ext)s"
            )
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=90,
                check=False,
            )
            candidates = sorted(Path(temp_dir).glob("*.vtt"))
            if result.returncode != 0 and not candidates:
                detail = (result.stderr or result.stdout).strip().splitlines()
                log.info(
                    "Japanese YouTube captions unavailable: %s",
                    detail[-1] if detail else f"yt-dlp exit code {result.returncode}",
                )
                return None
            if not candidates:
                log.info("this VOD has no downloadable Japanese auto-captions")
                return None

            # Prefer the Japanese track, but accept a Japanese regional variant.
            candidates.sort(
                key=lambda path: (
                    0 if ".ja." in path.name else 1,
                    path.name,
                )
            )
            shutil.copy2(candidates[0], cached)

    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("Japanese caption download failed; continuing with Haas: %s", exc)
        return None

    try:
        index = CaptionIndex.from_file(cached)
    except (OSError, UnicodeError) as exc:
        log.warning("could not parse downloaded captions; continuing with Haas: %s", exc)
        return None

    if not index.cues:
        log.info("downloaded Japanese caption file contained no usable cues")
        return None

    log.info("Japanese YouTube captions ready (%d cues)", len(index.cues))
    return index
