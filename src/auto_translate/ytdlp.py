"""Consistent bounded YouTube extraction for metadata, media and captions."""
import hashlib
import re
import subprocess
from pathlib import PureWindowsPath

from .windows_subprocess import hidden_console_flags


def cache_key(video_id: str) -> str:
    if (re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", video_id)
            and not PureWindowsPath(video_id).is_reserved()):
        return video_id
    return "id." + hashlib.sha256(video_id.encode()).hexdigest()


def extract(cfg, url, options, *, check=True):
    command = ["yt-dlp", "--no-warnings", "--js-runtimes", "deno", *options]
    if cfg.cookies_from_browser:
        command += ["--cookies-from-browser", cfg.cookies_from_browser]
    result = subprocess.run(
        [*command, "--", url], stdin=subprocess.DEVNULL, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=90,
        check=False,
        creationflags=hidden_console_flags(),
    )
    if check and result.returncode:
        detail = (result.stderr or result.stdout or "").strip()[-1600:]
        raise RuntimeError(f"yt-dlp failed (exit {result.returncode}): {detail}")
    return result
