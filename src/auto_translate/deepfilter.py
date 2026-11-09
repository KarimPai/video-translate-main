"""Realtime DeepFilterNet3 enhancement with bounded, seek-cancellable queues.

ASR audio alone is enhanced; MPV plays the original source. An uncached seek
must not wait for a fast-ingested VOD's old queued enhancement work to drain.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable

import numpy as np
from scipy.signal import firwin, lfilter

from .types import SAMPLE_RATE, AudioChunk, AudioProgress

log = logging.getLogger(__name__)

INPUT_RATE = 48000
OUTPUT_CHUNK = 1600  # 100 ms at 16 kHz
DECIMATION = INPUT_RATE // SAMPLE_RATE
MAX_QUEUED_MESSAGES = 16  # 8 * 4096 / 48000 ~= 0.7s of queued PCM

# Stateful low-pass filter for continuous 48 kHz -> 16 kHz conversion.
FILTER_TAPS = firwin(63, 1.0 / DECIMATION).astype(np.float32)


class DeepFilterWorker:
    def __init__(
        self,
        out_q: "queue.Queue[AudioChunk | AudioProgress | None]",
        first_sample: int,
        generation: int,
        is_current: Callable[[], bool] | None = None,
    ):
        self.out_q = out_q
        self.first_sample = first_sample
        self.generation = generation
        # Each input PCM chunk is followed by a progress marker; this limits
        # read-ahead while keeping VOD and live streaming memory bounded.
        self.input_q: queue.Queue[np.ndarray | str | None] = queue.Queue(
            maxsize=MAX_QUEUED_MESSAGES
        )
        self.error: Exception | None = None
        self._closed = False
        self._cancel = threading.Event()
        self._is_current = is_current

        self.worker = threading.Thread(
            target=self._run,
            name="deepfilter",
            daemon=True,
        )
        self.worker.start()

    def _cancelled(self) -> bool:
        return self._cancel.is_set() or (
            self._is_current is not None and not self._is_current()
        )

    def _enqueue(self, item: np.ndarray | str | None) -> bool:
        while not self._cancelled():
            if self.error is not None:
                raise RuntimeError(
                    f"DeepFilterNet failed: {self.error}"
                ) from self.error
            if not self.worker.is_alive():
                raise RuntimeError("DeepFilterNet worker stopped unexpectedly")
            try:
                self.input_q.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def put(self, pcm: np.ndarray) -> bool:
        """Return False instead of blocking on stale audio during a seek."""
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
            # A seek invalidates the generation: never drain queued PCM or
            # finalize the old stream before the new reader can start.
            self._cancel.set()
            self.worker.join(timeout=2.0)
            if self.worker.is_alive():
                log.warning("Old DeepFilterNet worker still finishing a cancelled chunk")
            return
        try:
            queued = self._enqueue(None)
            if queued:
                self.worker.join()
        finally:
            if self.error is not None:
                raise RuntimeError(
                    f"DeepFilterNet failed: {self.error}"
                ) from self.error

    def _deliver(self, message: AudioChunk | AudioProgress) -> bool:
        while not self._cancelled():
            try:
                self.out_q.put(message, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def _emit(self, audio: np.ndarray) -> bool:
        for offset in range(0, len(audio), OUTPUT_CHUNK):
            if self._cancelled():
                return False
            chunk = audio[offset:offset + OUTPUT_CHUNK]
            if len(chunk) == 0:
                continue
            if not self._deliver(
                AudioChunk(
                    self.first_sample,
                    chunk.astype(np.float32, copy=False),
                    time.monotonic(),
                    generation=self.generation,
                )
            ):
                return False
            self.first_sample += len(chunk)
        return True

    def _run(self) -> None:
        processor = None
        buffer = np.empty(0, dtype=np.float32)
        filter_state = np.zeros(len(FILTER_TAPS) - 1, dtype=np.float32)
        decimation_phase = 0

        def filter_and_buffer(enhanced: np.ndarray) -> bool:
            nonlocal filter_state, decimation_phase, buffer
            if len(enhanced) == 0 or self._cancelled():
                return not self._cancelled()
            filtered, filter_state = lfilter(
                FILTER_TAPS, [1.0], enhanced, zi=filter_state,
            )
            first_index = (-decimation_phase) % DECIMATION
            audio_16k = filtered[first_index::DECIMATION]
            decimation_phase = (decimation_phase + len(enhanced)) % DECIMATION
            if len(audio_16k):
                buffer = np.concatenate((buffer, audio_16k))
            while len(buffer) >= OUTPUT_CHUNK and not self._cancelled():
                if not self._emit(buffer[:OUTPUT_CHUNK]):
                    return False
                buffer = buffer[OUTPUT_CHUNK:]
            return not self._cancelled()

        try:
            from deepfilternet_rs import DeepFilterNetRealtime

            processor = DeepFilterNetRealtime()
            if processor.sample_rate != INPUT_RATE:
                raise RuntimeError(
                    f"DeepFilterNet expected {INPUT_RATE} Hz, got {processor.sample_rate}"
                )
            log.info("DeepFilterNet ready (%d Hz input)", processor.sample_rate)

            while not self._cancelled():
                try:
                    pcm = self.input_q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if self._cancelled():
                    break
                if pcm is None:
                    break
                if isinstance(pcm, str) and pcm == "progress":
                    if not self._deliver(
                        AudioProgress(self.first_sample, generation=self.generation)
                    ):
                        break
                    continue
                if not filter_and_buffer(processor.process_chunk(pcm)):
                    break

            # Finish the remaining clean audio only for a naturally completed
            # generation. On seek/stop the old audio must be discarded.
            if not self._cancelled():
                final = processor.finalize()
                processor = None
                if filter_and_buffer(final) and len(buffer):
                    self._emit(buffer)
        except Exception as exc:
            if not self._cancelled():
                self.error = exc
                log.exception("DeepFilterNet processing failed")
        finally:
            if processor is not None:
                try:
                    processor.finalize()
                except Exception:
                    if not self._cancelled():
                        log.warning("DeepFilterNet finalize failed", exc_info=True)
