import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

from auto_translate import ytdlp


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