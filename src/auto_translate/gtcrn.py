"""Streaming CPU GTCRN enhancement before FireRedVAD and Haas ASR.

Input and output use 16 kHz mono PCM. MPV continues playing original audio.
Each ingest/seek generation owns a fresh denoiser and bounded, cancellable queue.
"""
from __future__ import annotations

import logging
import os
import queue
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np

from .types import SAMPLE_RATE, AudioChunk, AudioProgress

log = logging.getLogger(__name__)

MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speech-enhancement-models/gtcrn_simple.onnx"
)
MODEL_FILE_SIZE = 535638  # official Sherpa ONNX release asset, not a guessed threshold
OUTPUT_CHUNK = 1600  # 100 ms of 16 kHz speech
MAX_QUEUED_MESSAGES = 16


def _model_file(custom_path: str = "") -> Path:
    if custom_path:
        path = Path(custom_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"GTCRN model not found: {path}")
        return path
    root = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".cache")))
    path = root / "auto-translate" / "gtcrn" / "gtcrn_simple.onnx"
    if path.is_file() and path.stat().st_size == MODEL_FILE_SIZE:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading GTCRN CPU model (about 523 KiB)")
    filename = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="gtcrn-", suffix=".onnx.part", dir=path.parent,
            delete=False,
        ) as temp:
            filename = Path(temp.name)
            with urllib.request.urlopen(
                urllib.request.Request(
                    MODEL_URL, headers={"User-Agent": "auto-translate-gtcrn/1.0"}
                ),
                timeout=45,
            ) as response:
                while chunk := response.read(65536):
                    temp.write(chunk)
        if filename.stat().st_size != MODEL_FILE_SIZE:
            raise RuntimeError(
                f"Invalid GTCRN download size: {filename.stat().st_size} bytes"
            )
        os.replace(filename, path)
        log.info("GTCRN model ready: %s", path)
    finally:
        if filename is not None:
            filename.unlink(missing_ok=True)
    return path


class GtcrnWorker:
    def __init__(
        self,
        out_q: "queue.Queue[AudioChunk | AudioProgress | None]",
        first_sample: int,
        generation: int,
        is_current: Callable[[], bool] | None = None,
        *,
        num_threads: int = 1,
        model_path: str = "",
    ) -> None:
        self.out_q = out_q
        self.first_sample = first_sample
        self.generation = generation
        self.num_threads = max(1, num_threads)
        self.model_path = model_path
        self.input_q: "queue.Queue[np.ndarray | str | None]" = queue.Queue(
            maxsize=MAX_QUEUED_MESSAGES
        )
        self.error: Exception | None = None
        self._closed = False
        self._cancel = threading.Event()
        self._is_current = is_current
        self.worker = threading.Thread(
            target=self._run, name="gtcrn", daemon=True,
        )
        self.worker.start()

    def _cancelled(self) -> bool:
        return self._cancel.is_set() or (
            self._is_current is not None and not self._is_current()
        )

    def _enqueue(self, item: np.ndarray | str | None) -> bool:
        while not self._cancelled():
            if self.error is not None:
                raise RuntimeError(f"GTCRN failed: {self.error}") from self.error
            if not self.worker.is_alive():
                raise RuntimeError("GTCRN worker stopped unexpectedly")
            try:
                self.input_q.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def put(self, pcm: np.ndarray) -> bool:
        if self._closed:
            return False
        return self._enqueue(pcm)

    def put_progress(self) -> bool:
        if self._closed:
            return False
        return self._enqueue("progress")

    def close(self, *, cancel: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        if cancel or self._cancelled():
            self._cancel.set()
            self.worker.join(timeout=2.0)
            if self.worker.is_alive():
                log.warning("Cancelled GTCRN worker is still finishing its current frame")
            return
        queued = self._enqueue(None)
        if queued:
            self.worker.join()
        if self.error is not None:
            raise RuntimeError(f"GTCRN failed: {self.error}") from self.error

    def _deliver(self, message: AudioChunk | AudioProgress) -> bool:
        while not self._cancelled():
            try:
                self.out_q.put(message, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        try:
            import sherpa_onnx

            path = _model_file(self.model_path)
            config = sherpa_onnx.OnlineSpeechDenoiserConfig(
                model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                    gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(
                        model=str(path),
                    ),
                    num_threads=self.num_threads,
                    provider="cpu",
                )
            )
            if not config.validate():
                raise RuntimeError("Invalid GTCRN model configuration")
            denoiser = sherpa_onnx.OnlineSpeechDenoiser(config)
            if denoiser.sample_rate != SAMPLE_RATE:
                raise RuntimeError(
                    f"GTCRN expects {denoiser.sample_rate} Hz, not {SAMPLE_RATE} Hz"
                )
            frame_shift = int(denoiser.frame_shift_in_samples)
            if frame_shift <= 0:
                raise RuntimeError("GTCRN returned an invalid frame shift")
            log.info("GTCRN ready (CPU %d threads, %d Hz, frame=%d)",
                     self.num_threads, SAMPLE_RATE, frame_shift)

            incoming = np.empty(0, dtype=np.float32)
            enhanced = np.empty(0, dtype=np.float32)

            def accept(samples: np.ndarray) -> bool:
                nonlocal enhanced
                samples = np.asarray(samples, dtype=np.float32)
                if samples.size:
                    if not np.all(np.isfinite(samples)):
                        raise RuntimeError("GTCRN emitted non-finite samples")
                    enhanced = np.concatenate((enhanced, samples))
                while len(enhanced) >= OUTPUT_CHUNK and not self._cancelled():
                    chunk = enhanced[:OUTPUT_CHUNK]
                    if not self._deliver(AudioChunk(
                        self.first_sample, chunk.copy(), time.monotonic(),
                        generation=self.generation,
                    )):
                        return False
                    self.first_sample += len(chunk)
                    enhanced = enhanced[OUTPUT_CHUNK:]
                return not self._cancelled()

            while not self._cancelled():
                try:
                    pcm = self.input_q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if pcm is None:
                    break
                if isinstance(pcm, str) and pcm == "progress":
                    if not self._deliver(AudioProgress(
                        self.first_sample, generation=self.generation,
                    )):
                        break
                    continue
                incoming = np.concatenate((
                    incoming, np.asarray(pcm, dtype=np.float32).reshape(-1),
                ))
                while len(incoming) >= frame_shift and not self._cancelled():
                    frame = np.ascontiguousarray(incoming[:frame_shift])
                    incoming = incoming[frame_shift:]
                    output = denoiser(frame, SAMPLE_RATE)
                    if output.sample_rate != SAMPLE_RATE:
                        raise RuntimeError("GTCRN changed the audio sample rate")
                    if not accept(output.samples):
                        break

            # A seek/close must never flush stale audio into the new generation.
            if not self._cancelled():
                if len(incoming):
                    if not accept(denoiser(
                        np.ascontiguousarray(incoming), SAMPLE_RATE
                    ).samples):
                        return
                if not accept(denoiser.flush().samples):
                    return
                if len(enhanced):
                    self._deliver(AudioChunk(
                        self.first_sample, enhanced.copy(), time.monotonic(),
                        generation=self.generation,
                    ))
                    self.first_sample += len(enhanced)
        except Exception as exc:
            if not self._cancelled():
                self.error = exc
                log.exception("GTCRN processing failed")
