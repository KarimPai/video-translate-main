"""Bounded, nonblocking application diagnostics and privacy-scoped export.

GPU reads belong to GpuBudget's sampler. This module only stores cached
measurements, and moves diagnostic disk writes off inference/playback threads.
No application logger can run after a complete operating-system freeze.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import logging
import math
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import queue
import re
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile

log = logging.getLogger(__name__)
_ACTIVE = None
_ACTIVE_LOCK = threading.Lock()
_MAX_RECORD = 16_384
_MAX_EXPORT_FILES = 32
_EXPORT_FILE_BYTES = 512 * 1024
_MAX_EXPORT_BYTES = 8 * 1024 * 1024
_LOG_NAME = re.compile(
    r"(?:diagnostics\.jsonl|run\.log|gui-lifecycle\.log|gui-startup\.log|"
    r"mpv\.log|plamo-server\.log|voxtral-server\.log)(?:\.[0-9]+)?\Z"
)
_SECRETS = re.compile(
    r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?|(?:token|password|cookie|api[_-]?key)"
    r"[\"']?\s*[:=]\s*[\"']?)[^\s,;\"'}]+"
)


def _redact(text: str) -> str:
    text = _SECRETS.sub(r"\1[redacted]", text)
    text = re.sub(r"(?i)([A-Z]:[\\/]+Users[\\/]+)[^\\/\s\"]+", r"\1[user]", text)
    return re.sub(r"(/(?:home|Users)/)[^/\s\"]+", r"\1[user]", text)


def _json_value(value):
    """Keep telemetry JSON-compatible and avoid objects retaining model buffers."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in list(value.items())[:64]}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value[:64]]
    return str(value)[:256]


class _Capture(logging.Handler):
    def __init__(self, owner):
        super().__init__(logging.INFO)
        self.owner = owner

    def emit(self, record):
        try:
            # Application logs only: HTTP libraries may carry request secrets.
            if not record.name.startswith("auto_translate"):
                return
            data = dict(level=record.levelname, logger=record.name,
                        message=_redact(record.getMessage())[:4096])
            if record.exc_info and record.exc_info[0]:
                data["exception"] = record.exc_info[0].__name__
            self.owner.record("application_log", **data)
        except Exception:
            # Logging failure must never affect the operation being diagnosed.
            pass


class RuntimeDiagnostics:
    def __init__(self, cfg, log_dir: Path = Path("logs")):
        self.enabled = bool(getattr(cfg, "enabled", True))
        self.log_dir = Path(log_dir)
        self.session = uuid.uuid4().hex
        self._lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False
        self._metrics = {"gpu": {}, "stages": {}, "queues": {},
                         "counters": {"dropped": 0, "expired": 0, "pending_subtitles": 0, "latency_s": None},
                         "playback": {}, "workers": {}}
        self._queue = queue.Queue(maxsize=512)
        self._stop = threading.Event()
        self._thread = None
        self._handler = None
        self._sink = None
        self._dropped = 0
        if not self.enabled:
            return
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            previous = self._previous_session()
            self._sink = RotatingFileHandler(
                self.log_dir / "diagnostics.jsonl", mode="a", encoding="utf-8",
                maxBytes=max(65536, int(getattr(cfg, "max_bytes", 2_000_000))),
                backupCount=max(1, min(10, int(getattr(cfg, "backups", 3)))),
            )
            self._sink.setFormatter(logging.Formatter("%(message)s"))
            self._thread = threading.Thread(target=self._write, name="runtime-diagnostics", daemon=True)
            self._thread.start()
            self._handler = _Capture(self)
            logging.getLogger().addHandler(self._handler)
            self.record("session_start", pid=os.getpid(),
                        os_freeze_logging="unavailable after complete operating-system freeze")
            if previous is not None:
                self.record("unclean_shutdown_suspected", previous_session=previous,
                            reason="previous diagnostics ended without a shutdown record; cause unknown")
        except (OSError, ValueError, RuntimeError) as exc:
            self.enabled = False
            if self._sink is not None:
                self._sink.close()
            log.warning("Application diagnostics unavailable: %s", exc)

    def _previous_session(self):
        """A missing final marker is evidence of interruption, never its cause."""
        try:
            path = self.log_dir / "diagnostics.jsonl"
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 65536))
                lines = stream.read(65536).decode("utf-8", errors="replace").splitlines()
            for line in reversed(lines):
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict) and record.get("session"):
                    return None if record.get("event") == "session_shutdown" else record["session"]
        except OSError:
            pass
        return None

    def record(self, event: str, **data):
        if not self.enabled or self._closed:
            return
        try:
            item = {"time_utc": datetime.now(timezone.utc).isoformat(),
                    "monotonic_s": round(time.monotonic(), 6),
                    "session": self.session, "event": str(event)[:96],
                    **_json_value(data)}
            line = _redact(json.dumps(item, ensure_ascii=False, default=str))
            if len(line.encode("utf-8")) > _MAX_RECORD:
                # Preserve valid JSON and the reason rather than splitting it.
                line = json.dumps({"event": str(event)[:96], "session": self.session,
                                   "truncated": True, "data": line[:3500]}, ensure_ascii=False)
            self._queue.put_nowait(line)
        except queue.Full:
            with self._lock:
                self._dropped += 1
        except (TypeError, ValueError, RuntimeError):
            pass

    def update(self, **metrics):
        values = _json_value(metrics)
        with self._lock:
            for key, value in values.items():
                if isinstance(value, dict) and isinstance(self._metrics.get(key), dict):
                    self._metrics[key].update(value)
                else:
                    self._metrics[key] = value

    def increment(self, name: str, amount: int = 1):
        with self._lock:
            counters = self._metrics.setdefault("counters", {})
            counters[name] = counters.get(name, 0) + amount

    def snapshot(self):
        with self._lock:
            result = copy.deepcopy(self._metrics)
            result["counters"]["diagnostic_records_dropped"] = self._dropped
        return result

    def _write(self):
        try:
            while True:
                try:
                    item = self._queue.get(timeout=0.2)
                except queue.Empty:
                    if self._stop.is_set():
                        return
                    continue
                try:
                    if item is None:
                        return
                    self._sink.emit(logging.LogRecord("diagnostics", logging.INFO, "", 0,
                                                      item, (), None))
                except Exception:
                    # A failed disk remains a diagnostic failure, not a pipeline failure.
                    with self._lock:
                        self._dropped += 1
                finally:
                    self._queue.task_done()
        finally:
            self._sink.close()

    def close(self):
        with self._close_lock:
            self._close()

    def _close(self):
        with self._lock:
            if self._closed:
                return
        self.record("session_shutdown", diagnostic_records_dropped=self._dropped)
        with self._lock:
            self._closed = True
        if self._handler is not None:
            logging.getLogger().removeHandler(self._handler)
            self._handler.close()
        if self._thread is not None:
            self._stop.set()
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                # The writer exits after draining even when no sentinel fits.
                pass
            self._thread.join(timeout=2)


def configure_diagnostics(cfg, log_dir: Path = Path("logs")) -> RuntimeDiagnostics:
    global _ACTIVE
    with _ACTIVE_LOCK:
        if _ACTIVE is not None:
            _ACTIVE.close()
        _ACTIVE = RuntimeDiagnostics(cfg, log_dir)
        return _ACTIVE


def record_event(event: str, **data):
    active = _ACTIVE
    if active is not None:
        active.record(event, **data)


def update_metrics(**metrics):
    active = _ACTIVE
    if active is not None:
        active.update(**metrics)


def increment_counter(name: str, amount: int = 1):
    active = _ACTIVE
    if active is not None:
        active.increment(name, amount)


def rotate_log(path: Path, max_bytes=2_000_000, backups=3):
    """Rotate an owned helper's raw log *before* opening its inherited handle.

    Rotating an open inherited Windows file handle is unsafe. Diagnostic JSONL
    rotates continuously; helper logs rotate between process starts.
    """
    path = Path(path)
    try:
        if not path.exists() or path.stat().st_size < max_bytes:
            return
        for index in range(backups, 0, -1):
            source = path if index == 1 else Path(f"{path}.{index - 1}")
            target = Path(f"{path}.{index}")
            if source.exists():
                source.replace(target)
    except OSError as exc:
        log.warning("Raw log rotation unavailable for %s: %s", path.name, exc)


class _RotatingStream:
    """A text write/flush/close interface for GUI-owned rotating log files."""
    def __init__(self, path, max_bytes, backups):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handler = RotatingFileHandler(path, maxBytes=max_bytes,
                                          backupCount=backups, encoding="utf-8")
        self.handler.setFormatter(logging.Formatter("%(message)s"))
        self.handler.terminator = ""
        self.closed = False

    def write(self, text):
        if self.closed:
            raise ValueError("write to closed diagnostics stream")
        # Bound each record too: a single giant Qt output chunk must not evade
        # the rotating handler's file limit.
        for offset in range(0, len(text), 8192):
            record = logging.LogRecord("diagnostics", logging.INFO, "", 0,
                                       text[offset:offset + 8192], (), None)
            self.handler.handle(record)
        return len(text)

    def flush(self):
        if not self.closed:
            self.handler.flush()

    def close(self):
        if not self.closed:
            self.closed = True
            self.handler.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def open_rotating_log(path: Path, max_bytes=2_000_000, backups=3):
    return _RotatingStream(path, max(65536, int(max_bytes)), max(1, min(10, int(backups))))


def _windows_event_metadata():
    if os.name != "nt":
        return {"available": False, "reason": "Windows event logs require Windows"}
    # Fixed queries only. Message/XML content is used locally for filtering and
    # never exported; identity, machine, raw EventData and rendered text omitted.
    script = r"""
$ErrorActionPreference='SilentlyContinue'
$since=(Get-Date).AddDays(-7)
$display=Get-WinEvent -FilterHashtable @{LogName='System';Id=4101;StartTime=$since} -MaxEvents 100 | Where-Object {$_.ProviderName -eq 'Display'}
$dx=Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Microsoft-Windows-DxgKrnl';StartTime=$since} -MaxEvents 100
$wer=Get-WinEvent -FilterHashtable @{LogName='Application';ProviderName='Windows Error Reporting';Id=1001;StartTime=$since} -MaxEvents 100 | Where-Object {$_.ToXml() -match 'LiveKernelEvent|WATCHDOG|0x141'}
@(@($display)+@($dx)+@($wer) | Select-Object -First 100 | ForEach-Object {[pscustomobject]@{event_id=$_.Id;provider=$_.ProviderName;time_utc=$_.TimeCreated.ToUniversalTime().ToString('o');record_id=$_.RecordId;level=$_.Level}}) | ConvertTo-Json -Compress
"""
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if completed.returncode:
            return {"available": False, "reason": "event query unavailable"}
        records = json.loads(completed.stdout) if completed.stdout.strip() else []
        if isinstance(records, dict):
            records = [records]
        allowed = {"event_id", "provider", "time_utc", "record_id", "level"}
        safe = [{key: item[key] for key in allowed if key in item}
                for item in records[:100] if isinstance(item, dict)]
        return {"available": True, "lookback_days": 7, "events": safe}
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        return {"available": False, "reason": "event query unavailable"}


def export_diagnostics(destination: Path, log_dir: Path = Path("logs"), *,
                       include_windows_events: bool = False) -> Path:
    """Export only allowlisted application log tails and optional GPU event IDs.

    No config, cookies, browser data, chat identities, dumps, machine identifiers,
    process command lines or arbitrary Windows event messages are gathered.
    """
    destination, log_dir = Path(destination), Path(log_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "files": [],
                "windows_events_requested": bool(include_windows_events),
                "limits": {"files": _MAX_EXPORT_FILES, "bytes_per_file": _EXPORT_FILE_BYTES,
                           "total_log_bytes": _MAX_EXPORT_BYTES},
                "note": "Application logs may include spoken subtitles. A complete OS freeze cannot be logged after it occurs."}
    candidates = []
    if log_dir.is_dir():
        candidates = [item for item in log_dir.iterdir()
                      if _LOG_NAME.fullmatch(item.name) and item.is_file() and not item.is_symlink()]
        ffmpeg = log_dir / "ffmpeg"
        if ffmpeg.is_dir() and not ffmpeg.is_symlink():
            candidates.extend(item for item in ffmpeg.glob("session-*-ffmpeg-*.log")
                              if item.is_file() and not item.is_symlink())
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".zip", delete=False) as stream:
            temporary = Path(stream.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            consumed = 0
            ranked = []
            for path in candidates:
                try:
                    ranked.append((path.stat().st_mtime, path))
                except OSError:
                    # Raw logs can rotate while the independent exporter runs.
                    continue
            for _, path in sorted(ranked, key=lambda item: item[0], reverse=True)[:_MAX_EXPORT_FILES]:
                allowance = min(_EXPORT_FILE_BYTES, _MAX_EXPORT_BYTES - consumed)
                if allowance <= 0:
                    break
                try:
                    with path.open("rb") as stream:
                        stream.seek(max(0, path.stat().st_size - allowance))
                        content = _redact(stream.read(allowance).decode("utf-8", errors="replace"))
                except OSError:
                    continue
                name = "logs/" + path.relative_to(log_dir).as_posix()
                encoded = content.encode("utf-8")[:allowance]
                archive.writestr(name, encoded)
                consumed += len(encoded)
                manifest["files"].append(name)
            if include_windows_events:
                archive.writestr("windows-event-metadata.json", json.dumps(_windows_event_metadata(), indent=2))
            archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        temporary.replace(destination)
        return destination
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
