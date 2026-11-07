"""VOD metadata and reference timeline persistence.

The downloaded yt-dlp metadata supplies the immutable VOD duration and chapter
boundaries. Runtime MPV session anchors are measured separately; they must not
be guessed from audio PTS because YouTube audio and video streams can have
different timestamp origins.
"""

import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import IngestConfig

log = logging.getLogger(__name__)

_FORMAT_FIELDS = (
    "format_id", "protocol", "ext", "width", "height", "fps",
    "vcodec", "acodec", "tbr", "format_note",
)


def _cookie_args(cfg: IngestConfig) -> list[str]:
    return (
        ["--cookies-from-browser", cfg.cookies_from_browser]
        if cfg.cookies_from_browser else []
    )


def _cache_dir() -> Path:
    root = os.environ.get("LOCALAPPDATA")
    base = Path(root) if root else Path.home() / ".cache"
    return base / "auto-translate" / "timelines"


def load_vod_metadata(url: str, cfg: IngestConfig) -> dict[str, Any]:
    """Fetch VOD metadata, persist a safe summary, and return its timeline."""
    cmd = [
        "yt-dlp",
        "--no-warnings",
        "--no-call-home",
        "--skip-download",
        "--dump-single-json",
    ]
    cmd += _cookie_args(cfg) + [url]

    raw = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    data = json.loads(raw)

    duration = data.get("duration")
    live_status = str(data.get("live_status") or "not_live")
    if duration is None and live_status not in ("is_live", "is_upcoming"):
        raise ValueError("yt-dlp did not provide a VOD duration")

    chapters = []
    for chapter in data.get("chapters") or []:
        start = chapter.get("start_time")
        end = chapter.get("end_time")
        if start is None or end is None:
            continue
        chapters.append({
            "start_s": float(start),
            "end_s": float(end),
            "title": str(chapter.get("title") or ""),
        })

    formats = []
    for item in data.get("requested_formats") or data.get("formats") or []:
        formats.append({
            key: item[key]
            for key in _FORMAT_FIELDS
            if item.get(key) is not None
        })

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "video_id": str(data.get("id") or ""),
        "title": str(data.get("title") or ""),
        # Channel/uploader identifies the talent even when their name is
        # absent from a gaming VOD's title.
        "channel": str(data.get("channel") or data.get("uploader") or ""),
        "extractor": str(data.get("extractor_key") or data.get("extractor") or ""),
        "duration_s": float(duration) if duration is not None else None,
        "upload_timestamp": data.get("release_timestamp") or data.get("timestamp"),
        "live_status": live_status,
        "chapters": chapters,
        "formats": formats,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }

    video_id = metadata["video_id"]
    if not video_id:
        raise ValueError("yt-dlp metadata did not contain a video ID")

    directory = _cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{video_id}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)

    metadata["cache_path"] = str(path)
    log.info(
        "VOD timeline metadata: id=%s duration=%s chapters=%d formats=%d cache=%s",
        video_id,
        _format_duration(float(duration)) if duration is not None else "LIVE/unknown",
        len(chapters),
        len(formats),
        path,
    )
    return metadata


def _format_duration(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


def record_session_anchor(
    timeline: dict[str, Any] | None,
    source_start_s: float,
    mpv_time_pos_s: float,
    reason: str,
) -> None:
    """Persist a measured MPV-to-source clock anchor in the VOD metadata."""
    if not timeline or not timeline.get("cache_path"):
        return

    path = Path(timeline["cache_path"])
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        sessions = stored.setdefault("session_anchors", [])
        sessions.append({
            "source_start_s": round(float(source_start_s), 6),
            "mpv_time_pos_s": round(float(mpv_time_pos_s), 6),
            "reason": str(reason),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        })
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(stored, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except (OSError, ValueError, json.JSONDecodeError):
        log.warning("could not persist VOD clock anchor to %s", path, exc_info=True)
