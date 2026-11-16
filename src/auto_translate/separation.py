
"""Buffered GPU vocal separation using audio-separator."""

import logging
import queue
import tempfile
import threading
import time
from concurrent.futures import CancelledError
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from .types import SAMPLE_RATE, AudioChunk
from .windows_subprocess import stopping

log = logging.getLogger(__name__)

INPUT_RATE = 44100
WINDOW_SECONDS = 4
HOP_SECONDS = 3
CONTEXT_SECONDS = 1
OUTPUT_CHUNK_SAMPLES = 4096


class VocalSeparator:
    def __init__(
        self,
        model_path: str = (
            "pretrained_models/separation/UVR-MDX-NET-Voc_FT.onnx"
        ),
    ):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable for vocal separation"
            )

        # audio-separator selects CUDA for PyTorch and ONNX models only when
        # both CUDA runtimes are present. Its default otherwise is CPU.
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in providers:
            raise RuntimeError(
                "Vocal separation requires ONNX Runtime CUDA, but "
                f"CUDAExecutionProvider is unavailable (providers: {providers})."
            )

        from audio_separator.separator import Separator

        model = Path(model_path).resolve()

        if not model.is_file():
            raise FileNotFoundError(
                f"Separation model not found: {model}"
            )

        self.temp = tempfile.TemporaryDirectory(
            prefix="vocal-separation-"
        )
        self.output_dir = Path(self.temp.name)

        try:
            self.separator = Separator(
                log_level=30,
                model_file_dir=str(model.parent),
                output_dir=str(self.output_dir),
                output_format="WAV",
                output_single_stem="Vocals",
                sample_rate=INPUT_RATE,
            )

            self.separator.load_model(
                model_filename=model.name
            )

            log.info(
                "Vocal separation model loaded on CUDA: %s",
                model.name,
            )

        except Exception:
            self.temp.cleanup()
            raise

    def separate(
        self,
        audio: np.ndarray,
        index: int,
    ) -> np.ndarray:
        import soundfile as sf

        source = self.output_dir / f"window_{index:06d}.wav"

        sf.write(
            source,
            audio,
            INPUT_RATE,
            subtype="PCM_16",
        )

        outputs = self.separator.separate(str(source))

        if not outputs:
            raise RuntimeError(
                "audio-separator returned no vocal output"
            )

        vocal_path = Path(outputs[0])

        if not vocal_path.is_absolute():
            vocal_path = self.output_dir / vocal_path

        if not vocal_path.is_file():
            raise RuntimeError(
                f"Vocal output file not found: {vocal_path}"
            )

        vocals, rate = sf.read(
            vocal_path,
            dtype="float32",
            always_2d=True,
        )

        if rate != INPUT_RATE:
            raise RuntimeError(
                f"Unexpected separation sample rate: {rate}"
            )

        if len(vocals) == 0:
            raise RuntimeError(
                "Vocal separation returned empty audio"
            )

        mono = vocals.mean(axis=1)
        return np.clip(mono, -1.0, 1.0).astype(np.float32)

    def close(self) -> None:
        self.temp.cleanup()


class SeparationWorker:
    """Separates buffered stereo audio without blocking FFmpeg."""

    def __init__(
        self,
        out_q: queue.Queue,
        first_sample: int,
        model_path: str,
        generation: int,
        gate=None,
        current=None,
    ):
        self.out_q = out_q
        self.first_sample = first_sample
        self.model_path = model_path
        self.generation = generation
        self.gate = gate
        self.current = current
        self._cancel = threading.Event()
        self._close_lock = threading.Lock()

        self.input_q: queue.Queue = queue.Queue(maxsize=4 if gate is not None and gate.enabled else 0)
        self.error: Exception | None = None
        self._closed = False

        self.worker = threading.Thread(
            target=self._run,
            name="vocal-separation",
            daemon=True,
        )
        self.worker.start()

    def put(self, pcm: np.ndarray) -> None:
        if self.error is not None:
            raise RuntimeError(
                f"Vocal separation worker failed: {self.error}"
            ) from self.error

        if self._closed:
            raise RuntimeError(
                "Vocal separation worker is already closed"
            )

        if pcm.ndim != 2 or pcm.shape[1] != 2:
            raise ValueError(
                "Vocal separation expects stereo audio"
            )

        while True:
            self._check_cancelled()
            if self.error is not None or not self.worker.is_alive():
                raise RuntimeError("Vocal separation worker stopped") from self.error
            if self._closed:
                raise CancelledError("Vocal separation is closing")
            try:
                self.input_q.put(pcm, timeout=0.1)
                return
            except queue.Full:
                continue

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            # Never block putting a sentinel into a full queue after a failed
            # worker, and keep seek/shutdown cancellation observable.
            while self.worker.is_alive():
                if self._is_cancelled():
                    self._cancel.set()
                    break
                try:
                    self.input_q.put(None, timeout=0.1)
                    break
                except queue.Full:
                    continue
            cancellation_deadline = None
            while self.worker.is_alive():
                if self._is_cancelled():
                    self._cancel.set()
                    if cancellation_deadline is None:
                        cancellation_deadline = time.monotonic() + 2
                    if time.monotonic() >= cancellation_deadline:
                        log.warning("Vocal separation still executing after cancellation; worker is daemonized")
                        break
                self.worker.join(timeout=0.1)
            if self.error is not None:
                raise RuntimeError(
                    f"Vocal separation failed: {self.error}"
                ) from self.error

    def _is_cancelled(self):
        return (self._cancel.is_set() or stopping.is_set()
                or (self.gate is not None and self.gate.stop.is_set())
                or (self.current is not None and self.current() != self.generation))

    def _check_cancelled(self):
        if self._is_cancelled():
            raise CancelledError("Vocal separation generation/session cancelled")

    def _operation(self, name):
        self._check_cancelled()
        return (self.gate.operation(name, cancel_event=self._cancel,
                                   current=self.current, generation=self.generation)
                if self.gate is not None else nullcontext())

    def _emit(
        self,
        audio: np.ndarray,
        start_frame: int,
    ) -> None:
        if len(audio) == 0:
            return
        from scipy.signal import resample_poly

        audio_16k = resample_poly(
            audio,
            160,
            441,
        ).astype(np.float32)

        audio_16k = np.clip(
            audio_16k,
            -1.0,
            1.0,
        )

        start_sample = (
            self.first_sample
            + round(start_frame * SAMPLE_RATE / INPUT_RATE)
        )

        for offset in range(
            0,
            len(audio_16k),
            OUTPUT_CHUNK_SAMPLES,
        ):
            chunk = audio_16k[
                offset:offset + OUTPUT_CHUNK_SAMPLES
            ]

            item = AudioChunk(
                    start_sample + offset,
                    chunk,
                    time.monotonic(),
                    generation=self.generation,
                )
            while True:
                self._check_cancelled()
                try:
                    self.out_q.put(item, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def _run(self) -> None:
        separator = None

        try:
            with self._operation("separation_load"):
                separator = VocalSeparator(self.model_path)
            self._process(separator)

        except CancelledError:
            log.debug("Discarded vocal separation generation=%d", self.generation)
        except Exception as exc:
            self.error = exc
            log.exception(
                "Vocal separation worker failed"
            )

        finally:
            if separator is not None:
                separator.close()

    def _process(
        self,
        separator: VocalSeparator,
    ) -> None:
        window_frames = WINDOW_SECONDS * INPUT_RATE
        hop_frames = HOP_SECONDS * INPUT_RATE
        context_frames = CONTEXT_SECONDS * INPUT_RATE

        buffer = np.empty(
            (0, 2),
            dtype=np.float32,
        )

        buffer_start = 0
        emitted_until = 0
        index = 0

        while True:
            self._check_cancelled()
            try:
                item = self.input_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is not None:
                buffer = np.concatenate(
                    (buffer, item),
                    axis=0,
                )

            finished = item is None

            while len(buffer) >= window_frames or (
                finished and len(buffer) > 0
            ):
                length = min(
                    len(buffer),
                    window_frames,
                )

                window = buffer[:length]

                index += 1
                with self._operation("separation"):
                    vocals = separator.separate(
                        window,
                        index,
                    )
                self._check_cancelled()

                # Keep the output length aligned with the input window.
                expected_length = length

                if len(vocals) < expected_length:
                    vocals = np.pad(
                        vocals,
                        (0, expected_length - len(vocals)),
                    )
                elif len(vocals) > expected_length:
                    vocals = vocals[:expected_length]

                window_end = buffer_start + length

                if finished:
                    emit_end = window_end
                else:
                    emit_end = max(
                        emitted_until,
                        window_end - context_frames,
                    )

                emit_start = max(
                    emitted_until,
                    buffer_start,
                )

                local_start = emit_start - buffer_start
                local_end = emit_end - buffer_start

                if local_end > local_start:
                    self._emit(
                        vocals[local_start:local_end],
                        emit_start,
                    )
                    emitted_until = emit_end

                if finished:
                    buffer = np.empty(
                        (0, 2),
                        dtype=np.float32,
                    )
                    break

                buffer = buffer[hop_frames:]
                buffer_start += hop_frames

            if finished:
                break
