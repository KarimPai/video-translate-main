"""Optional Holodex lookup of a VTuber video's channel and tagged collab cast."""
from __future__ import annotations

import logging
import re

import httpx

log = logging.getLogger(__name__)

API = "https://holodex.net/api/v2/videos/{video_id}"
_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")


def _names(entry: object) -> list[str]:
    if not isinstance(entry, dict):
        return []
    return [str(entry[key]) for key in ("english_name", "name") if isinstance(entry.get(key), str) and entry[key]]


def fetch_cast(video_id: str, api_key: str, timeout_s: float = 5.0) -> dict[str, list[str]]:
    """Return {"channel": names, "mentions": names}; empty on any failure."""
    if not api_key or not _VIDEO_ID.fullmatch(video_id or ""):
        return {}
    try:
        response = httpx.get(API.format(video_id=video_id), headers={"X-APIKEY": api_key},
                             timeout=timeout_s, follow_redirects=False)
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        # Never include the request (and its key header) in the log.
        log.warning("Holodex lookup failed for %s: %s", video_id, type(exc).__name__)
        return {}
    if not isinstance(data, dict):
        return {}
    mentions = data.get("mentions") if isinstance(data.get("mentions"), list) else []
    return {
        "channel": _names(data.get("channel")),
        "mentions": [name for entry in mentions for name in _names(entry)],
    }
