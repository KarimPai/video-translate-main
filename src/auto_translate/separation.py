
"""Buffered GPU vocal separation using audio-separator."""

import logging
import queue
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .types import SAMPLE_RATE, AudioChunk

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
    ):
        self.out_q = out_q
        self.first_sample = first_sample
        self.model_path = model_path
        self.generation = generation

        self.input_q: queue.Queue = queue.Queue()
        self.error: Exception | None = None
        self._closed = False
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

        self.input_q.put(pcm)

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        self.input_q.put(None)
        self.worker.join()

        if self.error is not None:
            raise RuntimeError(
                f"Vocal separation failed: {self.error}"
            ) from self.error

    def _emit(
        self,
        audio: np.ndarray,
        start_frame: int,
    ) -> None:
        if len(audio) == 0:
            return

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

            self.out_q.put(
                AudioChunk(
                    start_sample + offset,
                    chunk,
                    time.monotonic(),
                    generation=self.generation,
                )
            )

    def _run(self) -> None:
        separator = None

        try:
            separator = VocalSeparator(self.model_path)
            self._process(separator)

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
            item = self.input_q.get()

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
                vocals = separator.separate(
                    window,
                    index,
                )

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
