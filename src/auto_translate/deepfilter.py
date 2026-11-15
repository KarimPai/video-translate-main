"""Realtime speech enhancement using DeepFilterNet3."""

import logging
import queue
import threading
import time

import numpy as np
from scipy.signal import firwin, lfilter

from .types import SAMPLE_RATE, AudioChunk, AudioProgress

log = logging.getLogger(__name__)

INPUT_RATE = 48000
OUTPUT_CHUNK = 1600  # 100 ms at 16 kHz
DECIMATION = INPUT_RATE // SAMPLE_RATE

# Stateful low-pass filter for continuous 48 kHz -> 16 kHz conversion.
FILTER_TAPS = firwin(63, 1.0 / DECIMATION).astype(np.float32)


class DeepFilterWorker:
    def __init__(
        self,
        out_q: "queue.Queue[AudioChunk | None]",
        first_sample: int,
        generation: int,
    ):
        self.out_q = out_q
        self.first_sample = first_sample
        self.generation = generation
        self.input_q: queue.Queue = queue.Queue()
        self.error: Exception | None = None
        self._closed = False
        self._close_lock = threading.Lock()

        self.worker = threading.Thread(
            target=self._run,
            name="deepfilter",
            daemon=True,
        )
        self.worker.start()

    def put(self, pcm: np.ndarray) -> None:
        self.input_q.put(pcm)
        
    def put_progress(self) -> None:
        self.input_q.put("progress")

    def close(self) -> None:
        with self._close_lock:
            if not self._closed:
                self._closed = True
                self.input_q.put(None)
        self.worker.join()

        if self.error:
            raise RuntimeError(
                f"DeepFilterNet failed: {self.error}"
            ) from self.error

    def _emit(self, audio: np.ndarray) -> None:
        for offset in range(0, len(audio), OUTPUT_CHUNK):
            chunk = audio[offset:offset + OUTPUT_CHUNK]

            if len(chunk) == 0:
                continue

            self.out_q.put(
                AudioChunk(
                    self.first_sample,
                    chunk.astype(np.float32, copy=False),
                    time.monotonic(),
                    generation=self.generation,
                )
            )
            
            self.first_sample += len(chunk)

    def _run(self) -> None:
        processor = None
        buffer = np.empty(0, dtype=np.float32)
        filter_state = np.zeros(
            len(FILTER_TAPS) - 1,
            dtype=np.float32,
        )
        decimation_phase = 0

        try:
            from deepfilternet_rs import DeepFilterNetRealtime

            processor = DeepFilterNetRealtime()

            log.info(
                "DeepFilterNet ready (%d Hz input)",
                processor.sample_rate,
            )

            while True:
                pcm = self.input_q.get()

                if pcm is None:
                    break

                if isinstance(pcm, str) and pcm == "progress":
                    self.out_q.put(
                        AudioProgress(
                            self.first_sample,
                            generation=self.generation,
                        )
                    )
                    continue

                enhanced = processor.process_chunk(pcm)

                if len(enhanced) == 0:
                    continue

                filtered, filter_state = lfilter(
                    FILTER_TAPS,
                    [1.0],
                    enhanced,
                    zi=filter_state,
                )

                first_index = (-decimation_phase) % DECIMATION
                audio_16k = filtered[first_index::DECIMATION]

                decimation_phase = (
                    decimation_phase + len(enhanced)
                ) % DECIMATION

                if len(audio_16k):
                    buffer = np.concatenate((buffer, audio_16k))

                while len(buffer) >= OUTPUT_CHUNK:
                    self._emit(buffer[:OUTPUT_CHUNK])
                    buffer = buffer[OUTPUT_CHUNK:]

            final = processor.finalize()
            processor = None

            if len(final):
                filtered, filter_state = lfilter(
                    FILTER_TAPS,
                    [1.0],
                    final,
                    zi=filter_state,
                )

                first_index = (-decimation_phase) % DECIMATION
                audio_16k = filtered[first_index::DECIMATION]

                if len(audio_16k):
                    buffer = np.concatenate((buffer, audio_16k))

            if len(buffer):
                self._emit(buffer)

        except Exception as exc:
            self.error = exc
            log.exception("DeepFilterNet processing failed")

        finally:
            if processor is not None:
                try:
                    processor.finalize()
                except Exception:
                    pass