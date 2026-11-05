
"""Translation-synchronised MPV playback.

mpv listens on a local TCP port (paused); ingest's ffmpeg streams video+audio into it.
VOD playback starts as soon as translation is 10 seconds ahead, then pauses
whenever translated coverage falls to that lead. No deliberate video prebuffer.
For live streams, delay_s controls how far behind live playback starts.

A scheduler polls mpv's time-pos over JSON IPC (~10x/s) and draws subtitles
using osd-overlay.
"""
import ctypes
import json
import logging
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import MpvConfig
from ..ingest import Ingest, fmt_hms
from ..types import Subtitle

log = logging.getLogger(__name__)

IPC_PATH = (
    r"\\.\pipe\auto-translate-mpv"
    if sys.platform == "win32"
    else "/tmp/auto-translate-mpv.sock"
)
WINDOWS_DEFAULT = Path(r"C:\Program Files\MPV Player\mpv.exe")
OVERLAY_ID = 1
INPUT_CONF = Path(__file__).with_name("mpv_input.conf")

KEYS_HINT = (
    "←/→: jump 30s · shift+←/→: 5 min · space: pause · m: mute · "
    "↑/↓: volume · f: fullscreen · q: quit"
)
JUMP_MESSAGE = "auto-translate-jump"


def _create_kill_job(proc: subprocess.Popen) -> Any:
    """Automatically terminate MPV when the Python process exits."""
    if sys.platform != "win32":
        return None

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    kernel32.CreateJobObjectW.argtypes = [
        ctypes.c_void_p,
        wintypes.LPCWSTR,
    ]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE

    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL

    kernel32.AssignProcessToJobObject.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
    ]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL

    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    job = kernel32.CreateJobObjectW(None, None)

    if not job:
        raise ctypes.WinError(ctypes.get_last_error())

    info = EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = 0x2000

    if not kernel32.SetInformationJobObject(
        job,
        9,
        ctypes.byref(info),
        ctypes.sizeof(info),
    ):
        error = ctypes.get_last_error()
        kernel32.CloseHandle(job)
        raise ctypes.WinError(error)

    if not kernel32.AssignProcessToJobObject(
        job,
        wintypes.HANDLE(int(proc._handle)),
    ):
        error = ctypes.get_last_error()
        kernel32.CloseHandle(job)
        raise ctypes.WinError(error)

    return job


def _close_job(job: Any) -> None:
    """Close the Windows Job Object handle."""
    if job is None or sys.platform != "win32":
        return

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle(job)


class MpvError(Exception):
    pass


class MpvIpc:
    """mpv's JSON IPC: one JSON object per line."""

    def __init__(self, path: str, timeout_s: float = 15.0) -> None:
        deadline = time.monotonic() + timeout_s

        while True:
            try:
                if sys.platform == "win32":
                    self._f = open(path, "r+b", buffering=0)
                else:
                    sock = socket.socket(socket.AF_UNIX)
                    sock.connect(path)
                    self._f = sock.makefile("rwb", buffering=0)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.1)

        self._lock = threading.Lock()
        self._next_id = 0
        self.events: queue.Queue[dict] = queue.Queue()

    def command(self, *args: Any, **named: Any) -> Any:
        """Send one command and wait for its response."""
        with self._lock:
            self._next_id += 1
            rid = self._next_id

            cmd = {"name": args[0], **named} if named else list(args)

            self._f.write(
                (
                    json.dumps(
                        {"command": cmd, "request_id": rid}
                    ) + "\n"
                ).encode()
            )

            while True:
                line = self._f.readline()

                if not line:
                    raise ConnectionError("mpv closed the IPC connection")

                msg = json.loads(line)

                if msg.get("request_id") == rid:
                    if msg.get("error") != "success":
                        raise MpvError(msg.get("error"))
                    return msg.get("data")

                if "event" in msg:
                    self.events.put(msg)

    def get(self, prop: str) -> Any:
        """Return a property's value, or None if unavailable."""
        try:
            return self.command("get_property", prop)
        except MpvError:
            return None

    def close(self) -> None:
        self._f.close()


@dataclass
class _Line:
    sub: Subtitle
    show_from: float
    show_until: float


def find_mpv(configured: str) -> str:
    if configured:
        return configured

    if found := shutil.which("mpv"):
        return found

    if sys.platform == "win32" and WINDOWS_DEFAULT.exists():
        return str(WINDOWS_DEFAULT)

    raise SystemExit(
        "mpv not found: install it (winget install shinchiro.mpv) "
        "or set [mpv] path"
    )


class MpvOutput:
    # Keep playback this far behind completed translation.
    PAUSE_LEAD_SECONDS = 10.0

    def __init__(
        self,
        cfg: MpvConfig,
        ingest: Ingest,
        stop: threading.Event,
    ) -> None:
        self.cfg = cfg
        self.ingest = ingest
        self.stop = stop

        self.listen_url = f"tcp://127.0.0.1:{cfg.port}"

        self._lines: list[_Line] = []
        self._lock = threading.Lock()
        self._now: float | None = None
        self._shown = ""
        self._lags: list[float] = []
        self._session_start = ingest.source.start_s
        self._buffer_condition = threading.Condition()
        self._translated_until = ingest.source.start_s
        self._source_finished = False
        self._job = None
        self._closed = False

        self._proc = subprocess.Popen([
            find_mpv(cfg.path),
            f"{self.listen_url}?listen=1",

            "--pause",
            f"--input-ipc-server={IPC_PATH}",

            "--force-window=yes",
            "--title=auto-translate",
            "--keep-open=no",
            "--idle=yes",
            "--rebase-start-time=no",

            # Keep a modest forward cache to absorb brief FFmpeg/YouTube
            # delivery interruptions. VOD seeking still restarts ingest.
            "--cache=yes",
            "--demuxer-readahead-secs=12",
            "--demuxer-max-bytes=128MiB",
            "--demuxer-max-back-bytes=32MiB",
            "--input-default-bindings=no",

            # Disable unsafe on-screen controls.
            "--osc=no",
            "--no-input-default-bindings",
            f"--input-conf={INPUT_CONF}",

            # Hardware video decoding.
            f"--hwdec={cfg.hwdec}",

            "--msg-level=all=warn",
        ])

        if sys.platform == "win32":
            try:
                self._job = _create_kill_job(self._proc)
                log.info("mpv automatic process cleanup enabled")
            except OSError:
                log.exception(
                    "Could not enable automatic mpv process cleanup"
                )

        try:
            self.ipc = MpvIpc(IPC_PATH)

            # Wait for mpv to open its listening socket.
            deadline = time.monotonic() + 10

            while (
                self.ipc.get("path") is None
                and time.monotonic() < deadline
            ):
                if self._proc.poll() is not None:
                    raise RuntimeError("mpv exited during startup")
                time.sleep(0.1)

            time.sleep(0.5)

        except Exception:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait()

            _close_job(self._job)
            self._job = None
            raise

        log.info(
            "mpv ready on %s (live delay %.1fs, VOD translation lead %.1fs)",
            self.listen_url,
            cfg.delay_s,
            self.PAUSE_LEAD_SECONDS,
        )

        self._thread = threading.Thread(
            target=self._run,
            name="mpv",
            daemon=True,
        )
        self._thread.start()

    # ---- Output interface ----

    def show(self, sub: Subtitle) -> None:
        cfg = self.cfg

        if sub.end_s < self._session_start:
            return

        until = (
            max(sub.end_s, sub.start_s + cfg.min_display_s)
            + cfg.hold_s
        )

        line = _Line(sub, sub.start_s, until)
        now = self._now

        if now is not None and now > until:
            log.debug(
                "late subtitle (%.1fs): %s",
                now - sub.end_s,
                sub.ja,
            )
            line = _Line(
                sub,
                now,
                now + cfg.min_display_s + cfg.hold_s,
            )

        with self._lock:
            self._lines.append(line)

    def advance(self, progress_s: float) -> None:
        """Record the latest audio position safely processed for translation."""
        with self._buffer_condition:
            self._translated_until = max(
                self._translated_until,
                progress_s,
            )
            self._buffer_condition.notify_all()

    def reset_buffer(self, progress_s: float) -> None:
        """Reset translated coverage after a VOD seek."""
        with self._buffer_condition:
            self._translated_until = progress_s
            self._source_finished = False
            self._buffer_condition.notify_all()

    def finish(self) -> None:
        """Signal that no more translated audio will arrive."""
        with self._buffer_condition:
            self._source_finished = True
            self._buffer_condition.notify_all()

    def drain(self) -> None:
        """Let MPV finish playback before closing."""
        while not self.stop.is_set():
            if self._proc.poll() is not None:
                break

            if self.ipc.get("idle-active"):
                break

            time.sleep(0.5)

    def close(self) -> None:
        """Cleanly stop MPV and release its automatic cleanup protection."""
        if self._closed:
            return

        self._closed = True
        self.stop.set()

        with self._buffer_condition:
            self._buffer_condition.notify_all()

        if self._lags:
            on_time = sum(abs(x) <= 0.5 for x in self._lags)

            log.info(
                "mpv sync: %d lines shown, %d within 0.5s of speech, "
                "avg %+.2fs, worst %+.2fs",
                len(self._lags),
                on_time,
                sum(self._lags) / len(self._lags),
                max(self._lags, key=abs),
            )

        if hasattr(self, "_thread") and self._thread.is_alive():
            self._thread.join(timeout=2)

        try:
            if self._proc.poll() is None:
                self.ipc.command("quit")
        except (ConnectionError, OSError, MpvError):
            pass

        try:
            self._proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                log.error("Could not stop mpv process")

        finally:
            try:
                self.ipc.close()
            except (AttributeError, OSError):
                pass

            _close_job(self._job)
            self._job = None

    # ---- Scheduler ----

    def _run(self) -> None:
        try:
            first = True

            while not self.stop.is_set():
                self._play_session(show_hint=first)
                first = False

        except (
            ConnectionError,
            OSError,
            json.JSONDecodeError,
            MpvError,
        ):
            if not self.stop.is_set():
                log.exception("mpv playback/IPC error; stopping")
                self.stop.set()

    def _play_session(self, show_hint: bool) -> None:
        """Play only while translated subtitles remain sufficiently ahead."""

        lead = self.PAUSE_LEAD_SECONDS

        while not self.ingest.first_pts_ready.wait(0.2):
            if self.stop.is_set():
                return

        assert self.ingest.first_pts is not None

        offset = (
            self.ingest.first_pts
            - self.ingest.source.start_s
        )

        self._session_start = self.ingest.source.start_s

        if self.ingest.source.live:
            unpause_at = (
                self.ingest.first_pts_at
                + self.cfg.delay_s
            )
            log.info(
                "live playback delay: %.1fs",
                self.cfg.delay_s,
            )
        else:
            # Start the stream immediately, but keep MPV paused until the
            # translator has built a small lead. This is translation waiting,
            # not video prebuffering.
            self._now = self._session_start
            self.ingest.start_video()
            unpause_at = None

            log.info(
                "waiting for translation to reach %s (+%.0fs)",
                fmt_hms(self._session_start + lead),
                lead,
            )

        if self.stop.is_set():
            return

        paused = True

        while not self.stop.is_set():

            if (
                not self.ingest.source.live
                and not paused
                and self._now is not None
            ):
                with self._buffer_condition:
                    translated_lead = self._translated_until - self._now
                    should_pause = (
                        not self._source_finished
                        and translated_lead <= lead
                    )

                if should_pause:
                    self.ipc.command(
                        "set_property",
                        "pause",
                        True,
                    )
                    paused = True
                    unpause_at = None

                    log.info(
                        "playback paused: translation lead %.1fs (target %.1fs)",
                        translated_lead,
                        lead,
                    )

            should_unpause = (
                unpause_at is not None
                and time.monotonic() >= unpause_at
            )

            if paused and unpause_at is None:
                with self._buffer_condition:
                    should_unpause = (
                        self._source_finished
                        or (
                            self._now is not None
                            and self._translated_until >= self._now + lead
                        )
                    )

            if paused and should_unpause:
                if unpause_at is not None and show_hint:
                    if self.ingest.first_pts > 0.05:
                        self.ipc.command(
                            "seek",
                            self.ingest.first_pts,
                            "absolute+exact",
                        )

                self.ipc.command(
                    "set_property",
                    "pause",
                    False,
                )

                if show_hint:
                    self.ipc.command(
                        "show-text",
                        KEYS_HINT,
                        6000,
                    )

                paused = False
                unpause_at = None

                log.info(
                    "mpv playing from %s",
                    fmt_hms(self._session_start),
                )

            if (pos := self.ipc.get("time-pos")) is not None:
                self._now = pos - offset
                self._render(self._now)

            if (delta := self._jump_request()) and self._jump(delta):
                return

            time.sleep(0.1)

    def _jump_request(self) -> float:
        """Collect jump-key requests from mpv."""
        delta = 0.0

        while not self.ipc.events.empty():
            ev = self.ipc.events.get()
            args = ev.get("args") or []

            if (
                ev.get("event") == "client-message"
                and args[:1] == [JUMP_MESSAGE]
            ):
                delta += float(args[1])

        return delta

    def _jump(self, delta: float) -> bool:
        """Restart ingest and mpv at the requested VOD position."""
        src = self.ingest.source

        if src.live:
            self.ipc.command(
                "show-text",
                "Jumping only works for VODs",
                2000,
            )
            return False

        if self._now is None:
            return False

        target = max(0.0, self._now + delta)

        if src.duration:
            target = min(
                target,
                max(0.0, src.duration - 5),
            )

        log.info(
            "jump %+.0fs → %s",
            delta,
            fmt_hms(target),
        )

        self.ipc.command("set_property", "pause", True)

        self._now = None
        self._shown = ""

        self.ipc.command(
            "osd-overlay",
            id=OVERLAY_ID,
            format="none",
            data="",
        )

        with self._lock:
            self._lines.clear()

        self.ingest.request_restart(target)

        self.ipc.command(
            "loadfile",
            f"{self.listen_url}?listen=1",
        )

        self._wait_for_event("start-file")

        time.sleep(0.5)

        self.ipc.command(
            "show-text",
            f"Jumping to {fmt_hms(target)}…",
            4000,
        )

        self.ingest.go()
        return True

    def _wait_for_event(
        self,
        name: str,
        timeout_s: float = 5.0,
    ) -> None:
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            self.ipc.get("pause")

            while not self.ipc.events.empty():
                if self.ipc.events.get().get("event") == name:
                    return

            time.sleep(0.05)

    def _render(self, now: float) -> None:
        with self._lock:
            self._lines = [
                ln
                for ln in self._lines
                if ln.show_until > now - 60
            ]

            active = [
                ln
                for ln in self._lines
                if ln.show_from <= now < ln.show_until
            ]

        active.sort(key=lambda ln: ln.show_from)

        selected: list[_Line] = []
        used_rows = 0

        for ln in reversed(active):
            row_count = (
                (1 if self.cfg.ja_size > 0 else 0)
                + bool(ln.sub.en)
            )

            if (
                selected
                and used_rows + row_count
                > max(1, self.cfg.max_subtitle_rows)
            ):
                break

            selected.append(ln)
            used_rows += row_count

        active = list(reversed(selected))

        text = (
            _ass_many(
                [ln.sub for ln in active],
                self.cfg,
            )
            if active
            else ""
        )

        if text != self._shown:
            self._shown = text

            for ln in active:
                if abs(now - ln.show_from) < 0.15:
                    self._lags.append(
                        now - ln.sub.start_s
                    )

            self.ipc.command(
                "osd-overlay",
                id=OVERLAY_ID,
                format="ass-events" if text else "none",
                data=text,
                res_x=0,
                res_y=720,
            )


def _ass_many(
    subtitles: list[Subtitle],
    cfg: MpvConfig,
) -> str:
    """Render several active utterances as stacked mpv ASS overlay rows."""

    def esc(value: str) -> str:
        return (
            value
            .replace("\\", "＼")
            .replace("{", "｛")
            .replace("}", "｝")
            .replace("\n", " ")
        )

    blocks: list[str] = []

    for sub in subtitles:
        head = r"{\an2\bord2.5\shad0\3c&H000000&}"
        rows: list[str] = []

        if cfg.ja_size > 0 and sub.ja:
            rows.append(
                rf"{{\fs{cfg.ja_size}\1c&HC8C8C8&}}"
                + esc(sub.ja)
            )

        if sub.en:
            rows.append(
                rf"{{\fs{cfg.en_size}\1c&HFFFFFF&}}"
                + esc(sub.en)
            )

        if not rows:
            continue

        blocks.append(
            head + r"\N".join(rows)
        )

    return r"\N".join(blocks)


def _ass(sub: Subtitle, cfg: MpvConfig) -> str:
    """Backward-compatible single-subtitle renderer."""
    return _ass_many([sub], cfg)
