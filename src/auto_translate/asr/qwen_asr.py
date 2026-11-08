"""Offline Japanese Qwen3-ASR-1.7B-JA via a persistent speech.cpp worker.

The selected GGUF is speech.cpp-specific (not llama.cpp-compatible).
Only the Japanese fine-tuned Qwen model is loaded: no Haas fallback.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import queue
import threading
import time
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from ..config import AsrConfig
from ..process_cleanup import _close_job, _create_kill_job
from ..text_filter import is_whole_utterance_blocked, rejection_reason

log = logging.getLogger(__name__)

MODEL_REPO = "kznkzn/Qwen3-ASR-1.7B-JA-speechcpp-GGUF"
MODEL_FILENAME = "Qwen3-ASR-1.7B-JA-Q8_0.gguf"
MODEL_SHA256 = "cfe17bfad6708fd4ea243ade250634b7191becb491579ae8f19f423fb2516676"
MODEL_BYTES = 2_176_109_280


def resolve_qwen_asr_model(cfg: AsrConfig) -> Path:
    """Download once using Hugging Face's cache; reject damaged weights.

    Local paths allow offline installs without modifying the source tree.
    The published model's SHA256 is checked on first discovery in this
    process. Repeated ASR requests reuse the same loaded worker.
    """
    if cfg.model_path:
        model = Path(cfg.model_path).expanduser().resolve()
        if not model.is_file():
            raise RuntimeError(f"Qwen ASR GGUF not found: {model}")
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("Qwen ASR needs huggingface-hub; run uv sync") from exc
        print("Qwen ASR: checking Japanese fine-tune (2.18 GB first download)...", flush=True)
        try:
            model = Path(hf_hub_download(
                repo_id=MODEL_REPO, filename=MODEL_FILENAME,
            ))
        except Exception as exc:
            raise RuntimeError(
                "Qwen ASR Japanese model download failed. Check your network "
                f"or set [asr] model_path to a cached GGUF: {exc}"
            ) from exc
    # Do not silently run another Qwen model or a truncated cached file.
    if model.stat().st_size != MODEL_BYTES:
        raise RuntimeError(
            f"Qwen ASR model file is incomplete or unexpected: {model}. "
            f"Expected {MODEL_BYTES} bytes."
        )
    digest = hashlib.sha256()
    with model.open("rb") as stream:
        while block := stream.read(1024 * 1024 * 4):
            digest.update(block)
    if digest.hexdigest().lower() != MODEL_SHA256:
        raise RuntimeError(
            f"Qwen ASR model checksum mismatch: {model}. "
            "Delete/re-download the corrupted file."
        )
    return model


class QwenASR:
    """Japanese-only Qwen3-ASR recognition on the Vulkan GPU.

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
        exe = shutil.which("speech")
        if not exe:
            raise RuntimeError(
                "Qwen ASR needs speech.cpp 0.8.10+ (Windows Vulkan). "
                "Install speech.exe and check 'speech --version' and 'speech devices'."
            )
        model = resolve_qwen_asr_model(cfg)
        command = [exe, "worker", str(model), "--device", cfg.device,
                   "--language", "ja"]
        # Qwen accepts a short vocabulary hint, but an overlong/global
        # prompt can hallucinate unrelated words. Empty by default.
        if cfg.prompt.strip():
            command += ["--prompt", cfg.prompt.strip()[:200]]
        try:
            self.process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=None, text=True, encoding="utf-8", bufsize=1,
            )
            if sys.platform == "win32":
                try:
                    self._job = _create_kill_job(self.process)
                except OSError:
                    log.warning("Qwen ASR Windows job cleanup unavailable", exc_info=True)
            response = self.process.stdout.readline()
            if not response:
                raise RuntimeError(
                    "Qwen ASR worker quit during startup. Inspect logs/run.log, "
                    "check model integrity and GPU VRAM, and run 'speech devices'."
                )
            try:
                ready = json.loads(response)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"speech.cpp Qwen worker returned invalid startup data: {response[:240]!r}"
                ) from exc
            if ready.get("type") != "ready":
                raise RuntimeError(f"Qwen ASR worker failed to load: {ready}")
            # Only this thread reads stdout. A worker can otherwise hang
            # inside readline() indefinitely while its process stays alive.
            self._reader = threading.Thread(
                target=self._read_worker,
                args=(self.process.stdout,),
                name="qwen-asr-output",
                daemon=True,
            )
            self._reader.start()
        except Exception:
            self.close()
            raise
        log.info(
            "Qwen3-ASR-1.7B-JA ready (%s, language=ja, speech.cpp %s)",
            cfg.device, ready.get("version", "?"),
        )

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
            raise RuntimeError("Qwen ASR worker is no longer running")
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
            raise RuntimeError("Qwen ASR worker stopped accepting audio") from exc

        # Progress messages may arrive before the terminal 'end' response.
        # Keep an absolute deadline, not a fresh timeout per progress line.
        deadline = time.monotonic() + max(1.0, float(self.cfg.response_timeout_s))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "Qwen ASR response timeout: speech.cpp worker stopped "
                    "responding; check Vulkan memory and logs/run.log"
                )
            try:
                line = self._messages.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError(
                    "Qwen ASR response timeout: speech.cpp worker stopped "
                    "responding; check Vulkan memory and logs/run.log"
                ) from exc
            if isinstance(line, BaseException):
                raise RuntimeError("Qwen ASR worker stdout reader failed") from line
            if line is None:
                raise RuntimeError("Qwen ASR worker closed the stdout pipe")
            try:
                answer = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Qwen ASR worker returned invalid JSON: {line[:240]!r}"
                ) from exc
            if answer.get("type") == "fatal":
                raise RuntimeError(f"Qwen ASR worker failed: {answer}")
            if answer.get("id") != request_id:
                continue
            if answer.get("type") in {"error", "cancelled"}:
                raise RuntimeError(f"Qwen ASR recognition failed: {answer}")
            if answer.get("type") != "end":
                continue
            text = str(answer.get("text") or "").strip()
            if answer.get("stop") == "model_limit":
                log.warning("Qwen ASR reached model token limit on one utterance")
            if not text:
                return ""
            if is_whole_utterance_blocked(text, self.cfg.blocklist):
                log.debug("Qwen ASR blocked text: %s", text)
                return ""
            reason = rejection_reason(text)
            if reason is not None:
                log.debug("Qwen ASR discarded nonverbal speech (%s): %s", reason, text)
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
