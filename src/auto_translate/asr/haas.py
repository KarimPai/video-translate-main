"""Offline Japanese Haas-ja-v1.0 Q8_0 via a persistent speech.cpp worker.

The selected GGUF is speech.cpp-specific (not llama.cpp-compatible).
Only Haas Q8_0 is loaded: no secondary ASR backend.
"""
from __future__ import annotations

import base64
import json
import logging
import queue
import threading
import time
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np

from ..config import AsrConfig
from ..process_cleanup import _close_job, _create_kill_job
from ..text_filter import is_whole_utterance_blocked, rejection_reason

log = logging.getLogger(__name__)

def load_haas_keywords(filename: str) -> str:
    """Load a small, editable, newline-separated Haas keyword list from TOML.

    speech.cpp takes a *string*, not a JSON array, in worker requests.
    Each line is preferred written form, optionally =kana readings.
    """
    path = Path(filename)
    if not path.is_absolute():
        # Works from the launcher directory and from an installed package.
        from_cwd = Path.cwd() / path
        path = from_cwd if from_cwd.is_file() else Path(__file__).resolve().parents[3] / path
    try:
        with path.open("rb") as source:
            data = tomllib.load(source)
    except FileNotFoundError:
        log.warning("Haas vocabulary file not found: %s; continuing without keyword boosting", path)
        return ""

    entries = data.get("keywords")
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected a top-level keywords = [...] array")
    seen: set[str] = set()
    selected: list[str] = []
    for index, item in enumerate(entries, start=1):
        if not isinstance(item, str) or not item.strip() or any(c in item for c in "\r\n"):
            raise ValueError(f"{path}: invalid keyword at index {index}")
        keyword = item.strip()
        written, separator, readings = keyword.partition("=")
        if not written.strip() or (separator and not readings.strip()):
            raise ValueError(f"{path}: invalid keyword format at index {index}")
        if written in seen:
            log.warning("Ignoring duplicate Haas vocabulary entry: %s", written)
            continue
        seen.add(written)
        selected.append(keyword)
    return "\n".join(selected)


class HaasASR:
    """Japanese-only Haas recognition on the Vulkan GPU.

    It consumes the same float32 16 kHz PCM as the former recognizer, while
    retaining FireRedVAD, Sherpa turn segmentation, captions and MPV timing.
    """

    def __init__(self, cfg: AsrConfig) -> None:
        self.cfg = cfg
        self._counter = 0
        self._closed = False
        self.process: subprocess.Popen | None = None
        self._job = None
        self._messages: queue.Queue[str | BaseException | None] = queue.Queue()
        self._reader: threading.Thread | None = None
        self.keywords = (
            load_haas_keywords(cfg.haas_keywords_file)
            if cfg.haas_keywords_enabled else ""
        )
        exe = shutil.which("speech")
        if not exe:
            raise RuntimeError(
                "Haas ASR needs speech.cpp 0.10+ (Windows Vulkan). "
                "Install speech.exe and check 'speech --version' and 'speech devices'."
            )
        command = [
            exe, "worker", cfg.haas_model,
            "--device", cfg.haas_device,
            f"--decode-on-cpu={str(cfg.haas_decode_on_cpu).lower()}",
        ]
        try:
            self.process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=None, text=True, encoding="utf-8", bufsize=1,
            )
            if sys.platform == "win32":
                try:
                    self._job = _create_kill_job(self.process)
                except OSError:
                    log.warning("Haas ASR Windows job cleanup unavailable", exc_info=True)
            response = self.process.stdout.readline()
            if not response:
                raise RuntimeError(
                    "Haas ASR worker quit during startup. Inspect logs/run.log, "
                    "check model integrity and GPU VRAM, and run 'speech devices'."
                )
            try:
                ready = json.loads(response)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"speech.cpp Haas worker returned invalid startup data: {response[:240]!r}"
                ) from exc
            if ready.get("type") != "ready":
                raise RuntimeError(f"Haas ASR worker failed to load: {ready}")
            # Only this thread reads stdout. A worker can otherwise hang
            # inside readline() indefinitely while its process stays alive.
            self._reader = threading.Thread(
                target=self._read_worker,
                args=(self.process.stdout,),
                name="haas-asr-output",
                daemon=True,
            )
            self._reader.start()
        except Exception:
            self.close()
            raise
        log.info(
            "Haas-ja-v1.0 Q8_0 ready (%s, speech.cpp %s)",
            cfg.haas_device, ready.get("version", "?"),
        )
        if self.keywords:
            log.info("Haas vocabulary enabled: %d guarded terms (alpha=%.2f)",
                     len(self.keywords.splitlines()), cfg.haas_keywords_alpha)

    def _read_worker(self, stdout) -> None:
        """Feed all worker responses to a queue so requests can time out."""
        try:
            for line in stdout:
                self._messages.put(line)
        except (OSError, ValueError) as exc:
            self._messages.put(exc)
        finally:
            self._messages.put(None)

    def transcribe(self, audio: np.ndarray, probe: bool = False) -> str:
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Haas ASR worker is no longer running")
        samples = np.ascontiguousarray(audio, dtype=np.float32)
        pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
        pcm = base64.b64encode(pcm16.tobytes()).decode("ascii")
        self._counter += 1
        request_id = str(self._counter)
        chunk = {"type": "chunk", "id": request_id, "seq": 0, "pcm": pcm}
        request = {"type": "transcribe", "id": request_id, "sample_rate": 16000}
        stdin = self.process.stdin
        try:
            stdin.write(json.dumps(chunk) + "\n")
            stdin.write(json.dumps(request) + "\n")
            stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("Haas ASR worker stopped accepting audio") from exc

        # Progress messages may arrive before the terminal 'end' response.
        # Keep an absolute deadline, not a fresh timeout per progress line.
        deadline = time.monotonic() + max(1.0, float(self.cfg.response_timeout_s))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "Haas ASR response timeout: speech.cpp worker stopped "
                    "responding; check Vulkan memory and logs/run.log"
                )
            try:
                line = self._messages.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError(
                    "Haas ASR response timeout: speech.cpp worker stopped "
                    "responding; check Vulkan memory and logs/run.log"
                ) from exc
            if isinstance(line, BaseException):
                raise RuntimeError("Haas ASR worker stdout reader failed") from line
            if line is None:
                raise RuntimeError("Haas ASR worker closed the stdout pipe")
            try:
                answer = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Haas ASR worker returned invalid JSON: {line[:240]!r}"
                ) from exc
            if answer.get("type") == "fatal":
                raise RuntimeError(f"Haas ASR worker failed: {answer}")
            if answer.get("id") != request_id:
                continue
            if answer.get("type") in {"error", "cancelled"}:
                raise RuntimeError(f"Haas ASR recognition failed: {answer}")
            if answer.get("type") != "end":
                continue
            text = str(answer.get("text") or "").strip()
            if answer.get("stop") == "model_limit":
                log.warning("Haas ASR reached model token limit on one utterance")
            if not text:
                return ""
            if is_whole_utterance_blocked(text, self.cfg.blocklist):
                log.debug("Haas ASR blocked text: %s", text)
                return ""
            reason = rejection_reason(text)
            if reason is not None:
                log.debug("Haas ASR discarded nonverbal speech (%s): %s", reason, text)
                return ""
            return text

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        proc = self.process
        self.process = None
        try:
            if proc is not None:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=3)
                if proc.stdin is not None:
                    proc.stdin.close()
                if proc.stdout is not None:
                    proc.stdout.close()
        finally:
            _close_job(self._job)
            self._job = None
