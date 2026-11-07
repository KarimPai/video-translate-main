"""Shared yt-dlp authentication options for metadata, audio, captions and MPV.

The user's cookie files remain local and git-ignored. Never persist cookie
contents, and never include authentication values in diagnostic messages.
"""
from pathlib import Path

from .config import IngestConfig


def cookie_args(cfg: IngestConfig, directory: Path | None = None) -> list[str]:
    """Prefer explicitly configured browser cookies, then a local cookies.txt."""
    if cfg.cookies_from_browser:
        return ["--cookies-from-browser", cfg.cookies_from_browser]

    directory = directory if directory is not None else Path.cwd()
    for name in ("cookies.txt", "youtube-cookies.txt"):
        path = directory / name
        if path.is_file():
            return ["--cookies", str(path.resolve())]

    return []
