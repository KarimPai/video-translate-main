import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

from auto_translate import ytdlp
from auto_translate.output.mpv import _youtube_hook_options


def test_extract_uses_project_python_yt_dlp(monkeypatch):
    python = r"C:\project\.venv\Scripts\python.exe"
    url = "https://www.youtube.com/watch?v=example"
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    run = Mock(return_value=completed)
    monkeypatch.setattr(sys, "executable", python)
    monkeypatch.setattr(ytdlp.subprocess, "run", run)

    result = ytdlp.extract(SimpleNamespace(cookies_from_browser=""), url, ["--dump-json"])

    assert result is completed
    assert run.call_args.args[0] == [
        python, "-m", "yt_dlp", "--no-warnings", "--js-runtimes", "deno",
        "--dump-json", "--", url,
    ]


def test_mpv_hook_uses_project_managed_yt_dlp(monkeypatch, tmp_path):
    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    windows = sys.platform == "win32"
    python = scripts / ("python.exe" if windows else "python")
    executable = scripts / ("yt-dlp.exe" if windows else "yt-dlp")
    executable.touch()
    monkeypatch.setattr(sys, "executable", str(python))

    options = _youtube_hook_options(SimpleNamespace(ytdlp_format="best"), "https://youtube.com/watch?v=example")

    assert options == [
        f"--script-opts=ytdl_hook-ytdl_path={executable}",
        "--ytdl-format=best",
    ]


def test_mpv_hook_does_not_add_ytdlp_options_for_local_media(tmp_path):
    media = tmp_path / "clip.mp4"
    media.touch()

    assert _youtube_hook_options(SimpleNamespace(ytdlp_format="best"), str(media)) == []