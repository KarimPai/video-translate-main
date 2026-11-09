"""Isolated Reazon ASR process: keeps sherpa native ONNX Runtime away from CUDA VAD.

All requests are 16 kHz, mono, signed PCM16, transferred through line-delimited
JSON on stdio. stderr is inherited for launch diagnostics. PLaMo runs in parent.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import sys
import threading
from dataclasses import asdict

import numpy as np

from ..config import AsrConfig

log = logging.getLogger(__name__)


class ReazonWorkerASR:
    """Persistent CPU worker; native crash returns an actionable Python error."""

    def __init__(self, cfg: AsrConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._closed = False
        self._seq = 0
        cmd = [
            sys.executable, "-u", "-m", "auto_translate.asr.reazon_worker",
            json.dumps(asdict(cfg), ensure_ascii=False),
        ]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,    # user launcher redirects this to logs/run.log
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=flags,
        )
        try:
            ready = self._receive("initialization")
            if ready.get("type") != "ready":
                raise RuntimeError(
                    f"Reazon worker did not become ready: {ready!r}"
                )
            log.info(
                "ReazonSpeech ASR subprocess ready (pid=%s, %s)",
                self.process.pid,
                ready.get("detail", "sherpa-onnx"),
            )
        except Exception:
            self.close()
            raise

    def _receive(self, phase: str) -> dict:
        assert self.process.stdout is not None
        reply = self.process.stdout.readline()
        if not reply:
            code = self.process.poll()
            status = f"0x{(code & 0xffffffff):08X}" if code is not None else "unknown"
            raise RuntimeError(
                f"Reazon ASR subprocess terminated during {phase} "
                f"(exit={status}). Check logs/run.log above this line. "
                "If it mentions ONNX Runtime 1.17.1 / API 28, a stale Windows "
                "onnxruntime.dll was loaded. The worker must preload the DLL "
                "shipped with sherpa-onnx-core."
            )
        try:
            data = json.loads(reply)
        except ValueError as exc:
            raise RuntimeError(
                f"Reazon ASR subprocess returned invalid JSON: {reply[:300]!r}"
            ) from exc
        if data.get("type") == "error":
            raise RuntimeError(
                f"Reazon ASR subprocess failed during {phase}: "
                f"{data.get('message', 'unknown error')}"
            )
        return data

    def transcribe(self, audio: np.ndarray, probe: bool = False) -> str:
        if probe:
            return ""  # disable VAD-negative fallback for this experiment
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not samples.size or np.max(np.abs(samples)) < 1.0 / 32768:
            return ""
        if not np.isfinite(samples).all():
            raise ValueError("Nonfinite Reazon audio")
        if samples.size > self.cfg.max_audio_s * 16000:
            raise ValueError("Audio duration exceeds Reazon max_audio_s")
        pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
        with self._lock:
            if self._closed:
                raise RuntimeError("Reazon worker is closed")
            self._seq += 1
            assert self.process.stdin is not None
            request = {"type":"transcribe","id":self._seq,
                       "pcm":base64.b64encode(pcm.tobytes()).decode("ascii")}
            try:
                self.process.stdin.write(json.dumps(request) + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise RuntimeError(
                    f"Reazon ASR subprocess pipe closed: {exc}"
                ) from exc
            result = self._receive("transcription")
            if result.get("type") != "result" or result.get("id") != self._seq:
                raise RuntimeError(f"Unexpected Reazon ASR response: {result!r}")
            return str(result.get("text", "")).strip()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not hasattr(self, "process"):
            return
        if self.process.poll() is None:
            try:
                if self.process.stdin:
                    self.process.stdin.write('{"type":"shutdown"}\n')
                    self.process.stdin.flush()
            except (OSError, BrokenPipeError):
                pass
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
        if self.process.stdin:
            self.process.stdin.close()
        if self.process.stdout:
            self.process.stdout.close()
