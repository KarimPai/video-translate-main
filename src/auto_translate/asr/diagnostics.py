"""Nonblocking, best-effort JSONL diagnostics for Japanese ASR comparisons.

The latency-sensitive ASR/translation threads enqueue records only. A single
background writer batches filesystem writes and rotates at 20 MiB.
"""
import atexit
import json
import logging
import math
import queue
import threading
from pathlib import Path

log = logging.getLogger(__name__)
_MAX_FILE_BYTES = 20 * 1024 * 1024
_QUEUE = queue.Queue(maxsize=4096)
_START_LOCK = threading.Lock()
_STOPPING = False
_WRITER = None
_DROPPED = 0


def _bounded_record(record):
    """Bound memory per queued cue and never serialize raw audio or history."""
    remaining = [12000, 512]
    excluded = {"audio", "pcm", "audio_buffer", "history", "recent_ja", "messages"}

    def visit(value, depth=0):
        remaining[1] -= 1
        if remaining[1] < 0 or depth > 7:
            return "[omitted]"
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, (float, int)):
            return value if not isinstance(value, float) or math.isfinite(value) else None
        if isinstance(value, str):
            size = min(len(value), 1600, max(0, remaining[0]))
            remaining[0] -= size
            return value[:size] + ("…" if size < len(value) else "")
        if isinstance(value, dict):
            return {str(key)[:64]: visit(item, depth + 1)
                    for key, item in list(value.items())[:64]
                    if str(key).lower() not in excluded}
        if isinstance(value, (tuple, list)):
            return [visit(item, depth + 1) for item in value[:32]]
        # In particular, don't call str() on NumPy arrays, request objects,
        # exceptions containing request data, or arbitrary model objects.
        return "[unsupported value]"

    return visit(record)


def _write_batch(batch):
    """Write consecutive queued records in one disk operation per path."""
    by_path = {}
    for path, line in batch:
        by_path.setdefault(path, []).append(line)
    for path, lines in by_path.items():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            chunk = "".join(lines)
            size = path.stat().st_size if path.exists() else 0
            if size and size + len(chunk.encode("utf-8")) > _MAX_FILE_BYTES:
                path.replace(path.with_suffix(".previous.jsonl"))
            with path.open("a", encoding="utf-8") as stream:
                stream.write(chunk)
        except OSError as exc:
            log.warning("ASR comparison logging unavailable: %s", exc)


def _writer_loop():
    while True:
        item = _QUEUE.get()
        batch = []
        while item[0] is not None:
            batch.append(item)
            if len(batch) == 64:
                break
            try:
                item = _QUEUE.get(timeout=0.05)
            except queue.Empty:
                break
        try:
            if batch:
                _write_batch(batch)
        finally:
            for _ in batch:
                _QUEUE.task_done()
        if item[0] is None:
            _QUEUE.task_done()
            if item[1] is None:
                return
            item[1].set()


def append_comparison(record: dict, path: Path = Path("logs/asr-comparison.jsonl")) -> None:
    """Enqueue JSONL without blocking ASR on disk or Windows AV scans."""
    global _DROPPED, _WRITER
    try:
        line = json.dumps(_bounded_record(record), ensure_ascii=False,
                          separators=(",", ":")) + "\n"
        with _START_LOCK:
            if _STOPPING:
                return
            if _WRITER is None:
                _WRITER = threading.Thread(target=_writer_loop, daemon=True,
                                           name="asr-comparison-writer")
                _WRITER.start()
            _QUEUE.put_nowait((Path(path), line))
    except queue.Full:
        _DROPPED += 1
        if _DROPPED == 1 or _DROPPED % 1000 == 0:
            log.warning("ASR diagnostics queue full; dropped %d records", _DROPPED)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        log.warning("ASR comparison logging unavailable: %s", exc)


def flush_comparisons(timeout=3.0):
    """Wait for records queued before this call; useful for tests and shutdown."""
    done = threading.Event()
    with _START_LOCK:
        if _WRITER is None:
            return True
        if _STOPPING:
            return False
        try:
            _QUEUE.put((None, done), timeout=timeout)
        except queue.Full:
            return False
    return done.wait(timeout)


def _shutdown_writer(timeout=3.0):
    global _STOPPING
    with _START_LOCK:
        if _WRITER is None:
            return
        if not _STOPPING:
            try:
                _QUEUE.put((None, None), timeout=timeout)
            except queue.Full:
                log.warning("Timed out queuing diagnostics shutdown")
                return
            _STOPPING = True
    _WRITER.join(timeout)
    if _WRITER.is_alive():
        log.warning("Timed out flushing diagnostics on shutdown")


atexit.register(_shutdown_writer)
