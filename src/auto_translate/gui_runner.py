"""Qt-independent Windows GUI launch helpers (also unit-testable on Linux).

A packaged GUI is a frontend for the established project environment. Models,
MPV, speech.cpp and llama-server remain externally installed/shared.
"""
import os
import shutil
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


READY_MARKER = "VIDEO_TRANSLATE_GUI_READY"
QUALITY_OPTIONS = (360, 480, 720, 1080)


def valid_youtube_url(value: str) -> bool:
    """Accept direct YouTube VOD/live URLs, never arbitrary commands or hosts."""
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        url = urlsplit(value.strip())
        host = (url.hostname or "").lower()
        if url.scheme not in ("http", "https") or url.username or url.password or url.port:
            return False
        if host == "youtu.be":
            return bool(url.path.strip("/"))
        if host not in ("youtube.com", "www.youtube.com", "m.youtube.com",
                        "music.youtube.com", "www.youtube-nocookie.com"):
            return False
        segments = url.path.strip("/").split("/")
        if segments[0] == "watch":
            return bool(parse_qs(url.query).get("v"))
        return segments[0] in ("live", "shorts", "embed") and len(segments) > 1 and bool(segments[1])
    except ValueError:
        return False


def discover_project_root(executable: str | None = None, cwd: Path | None = None) -> Path:
    """Find project beside the source checkout or a dist/VideoTranslate.exe."""
    sources = [Path(cwd or Path.cwd()), Path(executable or sys.executable).resolve().parent,
               Path(__file__).resolve().parent]
    for source in sources:
        for candidate in (source, *source.parents):
            if ((candidate / "pyproject.toml").is_file() and
                    (candidate / "src" / "auto_translate" / "__main__.py").is_file()):
                return candidate
    raise FileNotFoundError("Project folder not found. Keep VideoTranslate.exe inside the project/dist folder.")


def backend_python(root: Path, frozen: bool = False, executable: str | None = None) -> Path:
    venv = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if venv.is_file():
        return venv
    if not frozen:
        path = Path(executable or sys.executable).resolve()
        if path.exists():
            return path
    raise FileNotFoundError("Python environment missing. Run Setup.bat / uv sync in this folder first.")


def build_backend_command(root: Path, url: str, quality: int, hwnd: int,
                          *, frozen: bool = False, executable: str | None = None):
    if not valid_youtube_url(url):
        raise ValueError("Enter a valid YouTube video or livestream URL.")
    if quality not in QUALITY_OPTIONS:
        raise ValueError("Unsupported video quality.")
    if not isinstance(hwnd, int) or hwnd <= 0:
        raise ValueError("Embedded MPV requires a valid native window handle.")
    python = backend_python(root, frozen=frozen, executable=executable)
    args = ["-u", "-m", "auto_translate", url.strip(),
            "--mode", "auto", "--mpv", "--shared-gpu-playback",
            "--no-web", "--hwdec", "auto-safe",
            "--video-quality", str(quality), "--mpv-wid", str(hwnd),
            "--simple-terminal"]
    return str(python), args


def prepare_project(root: Path):
    """Ensure config/log folders exist without resetting existing settings."""
    cfg = root / "config.toml"
    if not cfg.exists():
        shutil.copyfile(root / "config.dual-asr.example.toml", cfg)
    (root / "logs").mkdir(exist_ok=True)


def subprocess_environment(root: Path, original: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if original is None else original)
    env["PYTHONPATH"] = str(root / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    extra = []
    if os.name == "nt":
        local = Path(env.get("LOCALAPPDATA", ""))
        marker = local / "HaasPLaMo" / "dual-llama-bin-path.txt"
        if marker.is_file():
            path = Path(marker.read_text(encoding="utf-8-sig").strip())
            if (path / "llama-server.exe").is_file():
                extra.append(str(path))
        speech = local / "Programs" / "speech.cpp" / "current"
        if (speech / "speech.exe").is_file():
            extra.append(str(speech))
    if extra:
        env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    return env
