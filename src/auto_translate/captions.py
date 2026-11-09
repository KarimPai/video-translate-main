"""Japanese YouTube caption download, parsing and timestamp matching."""

import html
import json
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

    def __init__(self, cues: list[CaptionCue], *, source_format: str = "unknown") -> None:
        self.source_format = source_format
        self.source_origin = "provided"
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
            # YouTube's rolling VTT may use an earlier parent cue start for
            # text that first appears at an inline timestamp. Prefer that
            # explicit timestamp when the carried-over line was removed.
            inline_times = [
                _seconds(value)
                for line in payload
                for value in re.findall(
                    r"<((?:(?:\d+):)?\d{2}:\d{2}[.,]\d{3})>", line
                )
            ]
            valid_inline = [value for value in inline_times if value is not None]
            inline_payload = " ".join(payload)
            first_marker = re.search(
                r"<(?:(?:\d+):)?\d{2}:\d{2}[.,]\d{3}>",
                inline_payload,
            )
            # A cue such as "一緒に<00:00:03.000>遊びましょう" contains
            # speech before the marker. Never shift that earlier speech
            # to the timestamp of the later word.
            has_text_before_marker = bool(
                first_marker
                and _clean_caption([inline_payload[:first_marker.start()]])
            )
            if (valid_inline and not has_text_before_marker
                    and start <= min(valid_inline) < end):
                start = max(start, min(valid_inline))
            text = _clean_caption(payload)
            if not text or end <= start:
                continue
            cues.append(CaptionCue(start, end, text))
        return cls(cues, source_format="vtt")

    @classmethod
    def from_json3(cls, content: str) -> "CaptionIndex":
        """Read YouTube JSON3 events on the original media timeline.

        JSON3 preserves original event millisecond timestamps and segmented
        text. Join each event's pieces without inserting Japanese spaces;
        sentence reconstruction then occurs over the full downloaded track.
        Blank rolling/clear events are not speech.
        """
        document = json.loads(content)
        events = document.get("events", [])
        if not isinstance(events, list):
            raise ValueError("JSON3 subtitle events must be a list")
        cues: list[CaptionCue] = []
        for event_index, event in enumerate(events):
            if not isinstance(event, dict) or not isinstance(event.get("segs"), list):
                continue
            try:
                start_ms = float(event["tStartMs"])
            except (KeyError, ValueError, TypeError):
                continue
            if start_ms < 0:
                continue
            segments = [part for part in event["segs"] if isinstance(part, dict)]
            text = _clean_caption([
                "".join(str(part.get("utf8") or "") for part in segments)
            ])
            if not text:
                continue
            offsets = [
                max(0.0, float(part.get("tOffsetMs", 0)))
                for part in segments
                if str(part.get("utf8") or "").strip()
                and isinstance(part.get("tOffsetMs", 0), (int, float))
            ]
            # A caption can begin with an empty placeholder event segment.
            # Use the first audible text onset, not the placeholder timestamp.
            onset_ms = start_ms + (min(offsets) if offsets else 0.0)
            duration_ms = event.get("dDurationMs")
            try:
                duration_ms = float(duration_ms)
            except (ValueError, TypeError):
                duration_ms = 0.0
            end_ms = start_ms + max(0.0, duration_ms)
            if end_ms <= onset_ms:
                next_start = next(
                    (
                        float(item["tStartMs"])
                        for item in events[event_index + 1:]
                        if isinstance(item, dict)
                        and isinstance(item.get("tStartMs"), (int, float))
                        and item["tStartMs"] > onset_ms
                    ),
                    onset_ms + 2000.0,
                )
                end_ms = min(next_start, onset_ms + 4000.0)
            if end_ms > onset_ms:
                cues.append(CaptionCue(onset_ms / 1000.0, end_ms / 1000.0, text))
        return cls(cues, source_format="json3")

    @classmethod
    def from_file(cls, path: Path) -> "CaptionIndex":
        content = path.read_text(encoding="utf-8-sig")
        return cls.from_json3(content) if path.suffix.lower() == ".json3" else cls.from_vtt(content)

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
    """Download the whole Japanese track; prefer JSON3, fall back to VTT.

    Original VTT cached by older branches remains usable, but an available
    JSON3 track is preferred to retain structured caption event timings.
    """
    cache_dir = _caption_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    json3_cache = cache_dir / f"{video_id}.ja.json3"
    vtt_cache = cache_dir / f"{video_id}.ja.vtt"

    if json3_cache.is_file():
        try:
            index = CaptionIndex.from_file(json3_cache)
            if index.cues:
                index.source_origin = "cache"
                log.info(
                    "Japanese JSON3 captions loaded from cache (%d cues)",
                    len(index.cues),
                )
                return index
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            log.warning("corrupt cached JSON3: %s; re-downloading", json3_cache)

    command = [
        "yt-dlp",
        "--no-warnings",
        "--no-call-home",
        "--skip-download",
        "--write-auto-subs",
        "--write-subs",
        "--sub-langs", "ja.*,ja",
        "--sub-format", "json3/vtt/best",
        "--output", "",  # replaced by temp output path
    ]
    if cfg.cookies_from_browser:
        command += ["--cookies-from-browser", cfg.cookies_from_browser]
    command.append(url)

    selected: Path | None = None
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
            candidates = list(Path(temp_dir).glob("*.json3"))
            candidates += list(Path(temp_dir).glob("*.vtt"))
            if candidates:
                # Prefer structured JSON3 and an exact ja track over regional
                # variants. Only files belonging to the requested video ID.
                candidates = [
                    p for p in candidates if p.name.startswith(video_id + ".")
                ]
                candidates.sort(key=lambda p: (
                    0 if p.suffix.lower() == ".json3" else 1,
                    0 if ".ja." in p.name else 1,
                    p.name,
                ))
            for candidate in candidates:
                try:
                    potential = CaptionIndex.from_file(candidate)
                    if potential.cues:
                        destination = (
                            json3_cache if candidate.suffix == ".json3" else vtt_cache
                        )
                        shutil.copy2(candidate, destination)
                        selected = destination
                        index = potential
                        index.source_origin = "download"
                        break
                except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
                    log.warning("Invalid downloaded subtitle candidate: %s", candidate.name)
            if selected is None:
                detail = (result.stderr or result.stdout).strip().splitlines()
                log.info(
                    "No usable Japanese captions downloaded: %s",
                    detail[-1] if detail else f"yt-dlp exit code {result.returncode}",
                )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("Japanese captions download failed: %s", exc)

    if selected is None:
        # JSON3 unavailable or yt-dlp temporary failure: older cached VTT
        # is still better than losing all captions for an archived VOD.
        if vtt_cache.is_file():
            try:
                index = CaptionIndex.from_file(vtt_cache)
                if index.cues:
                    index.source_origin = "cached fallback"
                    log.info("Falling back to cached Japanese VTT (%d cues)", len(index.cues))
                    return index
            except (OSError, UnicodeError, ValueError):
                log.warning("Cached VTT is unreadable: %s", vtt_cache)
        return None
    log.info(
        "Japanese YouTube %s captions ready (%d cues, entire track)",
        selected.suffix.lstrip(".").upper(), len(index.cues),
    )
    return index
