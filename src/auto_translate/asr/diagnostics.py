"""Nonblocking, best-effort JSONL diagnostics for Japanese ASR comparisons.

The latency-sensitive ASR/translation threads enqueue records only. A single
background writer batches filesystem writes and rotates at 20 MiB.
"""
import atexit
import json
import logging
import queue
import threading
from pathlib import Path

log = logging.getLogger(__name__)
_MAX_FILE_BYTES = 20 * 1024 * 1024
_QUEUE = queue.Queue(maxsize=4096)
_START_LOCK = threading.Lock()
_STARTED = False
_WRITER = None
_DROPPED = 0


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
        if item[0] is None:
            item[1].set()
            _QUEUE.task_done()
            break
        batch = [item]
        end_after_batch = None
        # Bound per-write size and flush shortly after the first record.
        while len(batch) < 64:
            try:
                next_item = _QUEUE.get(timeout=0.05)
            except queue.Empty:
                break
            if next_item[0] is None:
                end_after_batch = next_item
                break
            batch.append(next_item)
        try:
            _write_batch(batch)
        finally:
            for _ in batch:
                _QUEUE.task_done()
        if end_after_batch is not None:
            end_after_batch[1].set()
            _QUEUE.task_done()
            break


def _start_writer():
    global _STARTED, _WRITER
    with _START_LOCK:
        if not _STARTED:
            _WRITER = threading.Thread(target=_writer_loop, daemon=True,
                                       name="asr-comparison-writer")
            _WRITER.start()
            _STARTED = True


def append_comparison(record: dict, path: Path = Path("logs/asr-comparison.jsonl")) -> None:
    """Enqueue JSONL without blocking ASR on disk or Windows AV scans."""
    global _DROPPED
    try:
        line = json.dumps(record, ensure_ascii=False, default=str,
                          separators=(",", ":")) + "\n"
        _start_writer()
        _QUEUE.put_nowait((Path(path), line))
    except queue.Full:
        _DROPPED += 1
        if _DROPPED == 1 or _DROPPED % 1000 == 0:
            log.warning("ASR diagnostics queue full; dropped %d records", _DROPPED)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        log.warning("ASR comparison logging unavailable: %s", exc)


def flush_comparisons(timeout=3.0):
    """Wait for records queued before this call; useful for tests and shutdown."""
    if not _STARTED:
        return True
    done = threading.Event()
    try:
        _QUEUE.put((None, done), timeout=timeout)
    except queue.Full:
        return False
    # The writer exits on the sentinel: only call when no further writes are
    # expected (tests and shutdown). New sessions start a new Python process.
    return done.wait(timeout)


atexit.register(flush_comparisons)
