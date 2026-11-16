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
GUI_CONTROL_PROTOCOL = "1"


def validate_gui_launch(args: list[str]) -> None:
    """Check the complete frontend command before wrapping/spawning it.

    The generic command builder deliberately has no listener credentials yet.
    Every real GUI launch must append exactly one authenticated endpoint.
    """
    if args.count("--gui-control") != 1:
        raise ValueError("GUI launch must enable authenticated player controls")
    values = {}
    for name in ("--gui-control-port", "--gui-control-token"):
        if args.count(name) != 1:
            raise ValueError("GUI/backend mismatch: control port/token missing or duplicated. Restart the updated GUI.")
        index = args.index(name) + 1
        if index >= len(args):
            raise ValueError("GUI/backend mismatch: incomplete control connection")
        values[name] = args[index]
    try:
        port = int(values["--gui-control-port"])
    except (TypeError, ValueError):
        raise ValueError("GUI control port must be an integer") from None
    token = values["--gui-control-token"]
    if not 0 < port <= 65535 or not isinstance(token, str) or not token or token.startswith("--"):
        raise ValueError("GUI/backend mismatch: invalid control connection")


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
    sources = [Path(executable or sys.executable).resolve().parent, Path(cwd or Path.cwd()),
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
                          *, rtx_vsr: bool = False, frozen: bool = False,
                          executable: str | None = None):
    if not valid_youtube_url(url):
        raise ValueError("Enter a valid YouTube video or livestream URL.")
    if quality not in QUALITY_OPTIONS:
        raise ValueError("Unsupported video quality.")
    if not isinstance(rtx_vsr, bool):
        raise ValueError("RTX VSR selection must be a boolean.")
    if not isinstance(hwnd, int) or hwnd <= 0:
        raise ValueError("Embedded MPV requires a valid native window handle.")
    python = backend_python(root, frozen=frozen, executable=executable)
    if os.name == "nt":
        # pythonw.exe is a GUI-subsystem interpreter: no transient CMD window
        # while QProcess still owns captured stdout/stderr pipes.
        python = python.with_name("pythonw.exe")
        if not python.is_file():
            raise FileNotFoundError("Windowless Python runtime missing from .venv/Scripts/pythonw.exe")
    args = ["-u", "-m", "auto_translate", url.strip(),
            "--mode", "auto", "--mpv", "--shared-gpu-playback",
            "--no-web", "--hwdec", "auto-safe",
            "--video-quality", str(quality), "--mpv-wid", str(hwnd),
            "--rtx-vsr" if rtx_vsr else "--no-rtx-vsr",
            "--gui-control", "--simple-terminal"]
    return str(python), args


def prepare_project(root: Path):
    """Ensure config/log folders exist without resetting existing settings."""
    cfg = root / "config.toml"
    if not cfg.exists():
        shutil.copyfile(root / "config.example.toml", cfg)
    (root / "logs").mkdir(exist_ok=True)


def subprocess_environment(root: Path, original: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if original is None else original)
    env["PYTHONPATH"] = str(root / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    env["VIDEO_TRANSLATE_GUI"] = "1"
    env["VIDEO_TRANSLATE_GUI_PROTOCOL"] = GUI_CONTROL_PROTOCOL
    env["VIDEO_TRANSLATE_FRONTEND"] = str(Path(sys.executable).resolve())
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


def session_launcher(python, args):
    """Use the base interpreter so the venv redirector itself is in the job."""
    config = Path(python).parent.parent / "pyvenv.cfg"
    base = Path(python)
    if config.is_file():
        entries = dict(line.split("=", 1) for line in config.read_text(encoding="utf8").splitlines()
                       if "=" in line)
        home = next((v.strip() for k, v in entries.items() if k.strip() == "home"), None)
        if not home:
            raise ValueError("Missing base interpreter home in pyvenv.cfg")
        base = Path(home) / "pythonw.exe"
    if not base.is_file():
        raise FileNotFoundError(f"Missing base interpreter: {base}")
    return str(base), ["-S", "-m", "auto_translate.gui_session", python, *args]
