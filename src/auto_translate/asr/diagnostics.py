"""Local, best-effort JSONL diagnostics for comparing Japanese ASR hypotheses.

No upload or external service. Diagnostics must never change inference output.
"""
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)
_MAX_FILE_BYTES = 20 * 1024 * 1024


def append_comparison(record: dict, path: Path = Path("logs/asr-comparison.jsonl")) -> None:
    """Record complete candidates, selection and timings; rotate at 20 MiB."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > _MAX_FILE_BYTES:
            path.replace(path.with_suffix(".previous.jsonl"))
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    except OSError as exc:
        log.warning("ASR comparison logging unavailable: %s", exc)
