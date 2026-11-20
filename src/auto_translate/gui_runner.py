"""Qt-independent Windows GUI launch helpers (also unit-testable on Linux).

A packaged GUI is a frontend for the established project environment. Models,
MPV and llama-server remain externally installed/shared.
"""
import math
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


def _is_project(path: Path) -> bool:
    return ((path / "pyproject.toml").is_file() and
            (path / "src" / "auto_translate" / "__main__.py").is_file())


def _venv_python(root: Path) -> Path:
    return root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _has_gui_runtime(root: Path) -> bool:
    python = _venv_python(root)
    return python.is_file() and (os.name != "nt" or python.with_name("pythonw.exe").is_file())


def discover_project_root(executable: str | None = None, cwd: Path | None = None) -> Path:
    """Find the source project or a standalone or checkout-local GUI package."""
    entry = Path(executable or sys.executable).resolve()
    package = entry.parent
    if (package.parent.name.lower() == "dist" and package.name.lower() == "videotranslate"
            and entry.name.lower() == "videotranslate.exe" and _is_project(package)):
        checkout = package.parent.parent
        if _has_gui_runtime(package):
            return package
        if _is_project(checkout) and _has_gui_runtime(checkout):
            return checkout
    sources = [Path(cwd or Path.cwd()), entry.parent, Path(__file__).resolve().parent]
    for source in sources:
        for candidate in (source, *source.parents):
            if _is_project(candidate):
                return candidate
    raise FileNotFoundError("Project folder not found. Keep VideoTranslate.exe inside the project/dist folder.")


def backend_python(root: Path, frozen: bool = False, executable: str | None = None) -> Path:
    venv = _venv_python(root)
    if venv.is_file():
        return venv
    if not frozen:
        path = Path(executable or sys.executable).resolve()
        if path.exists():
            return path
    raise FileNotFoundError("Python environment missing. Run scripts/Setup-Windows.ps1 or uv sync in this folder first.")


def build_backend_command(root: Path, url: str, quality: int, hwnd: int,
                          *, frozen: bool = False, context_lines: int = 6,
                          asr_model: str | None = None, mt_model: str | None = None,
                          asr_backend: str | None = None,
                          translate_backend: str | None = None,
                          safe_playback: bool = False, diagnostics: bool = True,
                          executable: str | None = None):
    if not valid_youtube_url(url):
        raise ValueError("Enter a valid YouTube video or livestream URL.")
    if quality not in QUALITY_OPTIONS:
        raise ValueError("Unsupported video quality.")
    for name, value in (("Safe Playback", safe_playback),
                        ("Diagnostics", diagnostics)):
        if type(value) is not bool:
            raise ValueError(f"{name} selection must be a boolean.")
    if type(context_lines) is not int or not 0 <= context_lines <= 20:
        raise ValueError("Translation context must be between 0 and 20 lines.")
    for name, value in (("ASR model", asr_model), ("Translation model", mt_model)):
        if value is not None and (not isinstance(value, str) or not value.strip() or "\x00" in value):
            raise ValueError(f"{name} path must be nonempty text.")
    for name, value in (("ASR backend", asr_backend), ("Translation backend", translate_backend)):
        if value is not None and (not isinstance(value, str) or not value.strip() or "\x00" in value):
            raise ValueError(f"{name} must be nonempty text.")
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
            "--gui-control", "--simple-terminal"]
    args.extend(["--context-lines", str(context_lines)])
    for flag, value in (("--asr-model", asr_model), ("--mt-model", mt_model)):
        if value is not None:
            args.extend([flag, value.strip()])
    for flag, value in (("--asr-backend", asr_backend), ("--translate-backend", translate_backend)):
        if value is not None:
            args.extend([flag, value.strip()])
    for enabled, flag in ((safe_playback, "safe-playback"),
                          (diagnostics, "diagnostics")):
        args.append(f"--{flag}" if enabled else f"--no-{flag}")
    return str(python), args


def diagnostic_rows(event: dict) -> dict[str, str]:
    """Format cached observations without turning requests into confirmations."""
    def section(name):
        value = event.get(name)
        return value if isinstance(value, dict) else {}

    def number(value, suffix="", precision=1):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            return "unknown"
        return f"{value:.{precision}f}{suffix}"

    def state(value):
        return "yes" if value is True else "no" if value is False else "unknown"

    def description(value):
        if value is None or value == "":
            return "unknown"
        if isinstance(value, bool):
            return state(value)
        return str(value).replace("\n", " ")[:160]

    gpu, stages, queues = section("gpu"), section("stages"), section("queues")
    counters, playback, workers = section("counters"), section("playback"), section("workers")
    gpu_text = (f"VRAM {number(gpu.get('used_gb'), ' GB')} used / "
                f"{number(gpu.get('total_gb'), ' GB')} total; "
                f"{number(gpu.get('free_gb'), ' GB')} available")
    gpu_activity = (f"Utilization {number(gpu.get('utilization_pct', gpu.get('utilization_percent')), '%')}; "
                    f"temperature {number(gpu.get('temperature_c'), ' °C')}; "
                    f"power {number(gpu.get('power_w'), ' W')}")
    stage_text = "; ".join(f"{label} {number(stages.get(key), ' s', 2)}" for label, key in (
        ("ASR", "asr_s"), ("Translation", "mt_s")))
    pending = counters.get("pending_subtitles", queues.get("subtitles", queues.get("subtitle")))
    subtitle_text = (f"Pending {number(pending, precision=0)}; "
                     f"latency {number(counters.get('latency_s'), ' s', 2)}; "
                     f"dropped {number(counters.get('dropped'), precision=0)}; "
                     f"expired {number(counters.get('expired'), precision=0)}")
    queue_text = "; ".join(f"{description(name)}: {number(value, precision=0)}"
                            for name, value in list(queues.items())[:8]) or "unknown"
    playback_text = (f"Resolution {description(playback.get('resolution'))}; "
                     f"decoder {description(playback.get('decoder'))}; "
                     f"hardware decoding {description(playback.get('hwdec'))}")
    worker_parts = []
    threads = workers.get("threads")
    if isinstance(threads, dict):
        thread_status = [f"{description(name)} {'running' if active is True else 'stopped' if active is False else 'unknown'}"
                         for name, active in list(threads.items())[:8]]
        worker_parts.append("Threads: " + (", ".join(thread_status) or "none"))
    processes = workers.get("processes")
    if isinstance(processes, list):
        process_status = []
        for process in processes[:12]:
            if not isinstance(process, dict):
                continue
            status = "running" if process.get("running") is True else (
                f"exited ({description(process.get('exit_code'))})" if process.get("running") is False
                else "unknown")
            process_status.append(f"{description(process.get('name'))} (PID {description(process.get('pid'))}): {status}")
        worker_parts.append("Processes: " + (", ".join(process_status) or "none"))
    worker_parts.extend(f"{description(name)}: {description(value)}"
                        for name, value in list(workers.items())[:12]
                        if name not in ("threads", "processes"))
    worker_text = "; ".join(worker_parts) or "unknown"
    return {"gpu": gpu_text, "activity": gpu_activity, "stages": stage_text,
            "subtitles": subtitle_text, "queues": queue_text,
            "playback": playback_text, "workers": worker_text}


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
    if os.name == "nt":
        extra = []
        for marker_name, executable in (("llama-bin-path.txt", "llama-server.exe"),):
            marker = root / ".cache" / marker_name
            if marker.is_file():
                directory = Path(marker.read_text(encoding="utf-8-sig").strip())
                if (directory / executable).is_file():
                    extra.append(str(directory))
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
