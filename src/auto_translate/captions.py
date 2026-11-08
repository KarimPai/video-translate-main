"""Japanese YouTube auto-caption download, parsing and timestamp matching."""

import html
import logging
import os
import re
import shutil
import subprocess
import tempfile
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path

from .config import IngestConfig
from .text_filter import strip_nonverbal_labels

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
        self._starts = [cue.start_s for cue in self.cues]
        # Ends can be out of order when a long cue overlaps shorter ones.
        self._latest_ends = list(accumulate((cue.end_s for cue in self.cues), max))

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
            payload = lines[timing_index + 1:]
            # YouTube rolling VTT can retain the previous line above a new
            # timestamped line. Discard it only when it matches the last cue.
            if (cues and len(payload) > 1 and cues[-1].end_s >= start - 0.05
                    and re.search(r"<(?:\d+:)?\d{2}:\d{2}[.,]\d{3}>", " ".join(payload[1:]))
                    and _clean_caption(payload[:1]) == cues[-1].text):
                payload = payload[1:]
            text = _clean_caption(payload)
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
        first = bisect_left(self._latest_ends, start_s - tolerance_s)
        stop = bisect_right(self._starts, end_s + tolerance_s)
        for index in range(first, stop):
            cue = self.cues[index]
            if cue.end_s < start_s - tolerance_s:
                continue
            overlap = max(
                0.0,
                min(end_s, cue.end_s) - max(start_s, cue.start_s),
            )
            if overlap > 0:
                matched.append((cue, overlap))

        if not matched:
            return None

        # Stage directions cannot lend their overlap to a sliver of dialogue.
        cleaned = [(cue, overlap, strip_nonverbal_labels(cue.text)) for cue, overlap in matched]
        speech = [entry for entry in cleaned if entry[2]]
        evidence = speech or cleaned
        texts: list[str] = []
        covered_to = start_s
        overlap_s = 0.0
        well_aligned = True
        for cue, overlap, line in evidence:
            left, right = max(start_s, cue.start_s), min(end_s, cue.end_s)
            overlap_s += max(0.0, right - max(left, covered_to))
            covered_to = max(covered_to, right)
            # A short segment inside a long cue cannot verify the whole line.
            if overlap < 0.40 * (cue.end_s - cue.start_s):
                well_aligned = False
            if not line:
                continue
            if texts and line.startswith(texts[-1]):
                texts[-1] = line
            elif not texts or texts[-1] != line:
                texts.append(line)

        coverage = overlap_s / duration
        return CaptionMatch(
            text=" ".join(texts),
            overlap_s=overlap_s,
            coverage=coverage,
            reliable=well_aligned and coverage >= 0.60 and overlap_s >= 0.25,
        )


class PendingCaptions:
    """Publish a complete index when the background VTT download finishes."""

    def __init__(self) -> None:
        self.index: CaptionIndex | None = None

    def match(self, start_s: float, end_s: float) -> CaptionMatch | None:
        index = self.index
        return index.match(start_s, end_s) if index is not None else None


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
    if cfg.cookies_from_browser:
        command += ["--cookies-from-browser", cfg.cookies_from_browser]
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
        log.warning("Japanese caption download failed; continuing with Qwen ASR: %s", exc)
        return None

    try:
        index = CaptionIndex.from_file(cached)
    except (OSError, UnicodeError) as exc:
        log.warning("could not parse downloaded captions; continuing with Qwen ASR: %s", exc)
        return None

    if not index.cues:
        log.info("downloaded Japanese caption file contained no usable cues")
        return None

    log.info("Japanese YouTube captions ready (%d cues)", len(index.cues))
    return index
