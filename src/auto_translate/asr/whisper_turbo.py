"""Optional Japanese Whisper Large-v3 Turbo ASR in a killable CTranslate2 process.

Keep CUDA allocations isolated from Haas Vulkan and llama.cpp. The parent can
enforce a deadline and release VRAM without leaving a timed-out decode running.
"""
import logging
from concurrent.futures import CancelledError
import multiprocessing as mp
import threading
import time
from pathlib import Path

import numpy as np

from ..text_filter import trim_repetition_tail

log = logging.getLogger(__name__)
MODEL = "h2oai/faster-whisper-large-v3-turbo"


def _audio16(audio):
    """Validate the existing 16 kHz mono float samples, without resampling."""
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Whisper Turbo expects finite 16 kHz mono audio")
    return np.ascontiguousarray(np.clip(samples, -1, 1))


def _decode(model, samples, beam_size, initial_prompt=None, *, temperature=0.0):
    """Never let Whisper auto-detect English or choose the translation task."""
    segments, _ = model.transcribe(
        samples,
        task="transcribe",
        language="ja",
        beam_size=beam_size,
        temperature=temperature,
        condition_on_previous_text=False,
        initial_prompt=initial_prompt,
        vad_filter=False,  # FireRedVAD has already segmented original audio.
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
        without_timestamps=True,
        word_timestamps=False,
    )
    phrases = []
    for segment in segments:  # faster-whisper decodes lazily.
        if (getattr(segment, "no_speech_prob", 0.0) > 0.75
                and getattr(segment, "avg_logprob", 0.0) < -0.75):
            continue
        value = segment.text.strip()
        if value:
            phrases.append(trim_repetition_tail(value))
    return " ".join(phrases).strip()


def _worker_main(connection, model_name, device, compute_type, beam_size):
    try:
        Path("logs").mkdir(exist_ok=True)
        logging.basicConfig(filename="logs/whisper-turbo.log", encoding="utf-8",
                            level=logging.INFO, force=True)
        from auto_translate.cuda_dlls import setup
        setup()
        from faster_whisper import WhisperModel
        model = WhisperModel(model_name, device=device,
                             compute_type=compute_type, cpu_threads=4,
                             num_workers=1)
        connection.send({"type": "ready"})
        while True:
            command = connection.recv()
            if command is None:
                break
            try:
                samples = _audio16(command["audio"])
                prompt = command.get("initial_prompt")
                transcript = _decode(model, samples, beam_size, initial_prompt=prompt)
                connection.send({"type": "transcript", "text": transcript, "id": command["id"],
                                 "stop": "complete" if transcript else "unknown"})
            except Exception as exc:
                logging.exception("Whisper Turbo decode failed")
                connection.send({"type": "error", "message": str(exc)})
    except EOFError:
        pass
    except BaseException as exc:
        logging.exception("Whisper Turbo startup or worker failure")
        try:
            connection.send({"type": "error", "message": str(exc)})
        except (OSError, EOFError, BrokenPipeError):
            pass
    finally:
        connection.close()


class WhisperTurboASR:
    """Persistent, single-flight secondary Japanese recognition worker."""

    def __init__(self, *, model=MODEL, device="cuda", compute_type="int8_float16",
                 beam_size=6, startup_timeout=600):
        self.last_stop = "unknown"
        self._lock = threading.Lock()
        self._request_id = 0
        self.process = None
        self.connection = None
        context = mp.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        self.connection = parent
        self.process = context.Process(
            target=_worker_main,
            args=(child, model, device, compute_type, beam_size),
            name="whisper-turbo-asr", daemon=True,
        )
        try:
            self.process.start()
            child.close()
            deadline = time.monotonic() + startup_timeout
            while time.monotonic() < deadline:
                if self.connection.poll(0.2):
                    response = self.connection.recv()
                    if response.get("type") == "ready":
                        log.info("Whisper Turbo ready: model=%s device=%s compute=%s beam=%d",
                                 model, device, compute_type, beam_size)
                        return
                    raise RuntimeError("Whisper Turbo startup: " + response.get("message", "unknown error"))
                if not self.process.is_alive():
                    raise RuntimeError("Whisper Turbo worker exited during model startup")
            raise TimeoutError("Whisper Turbo startup timed out; inspect logs/whisper-turbo.log")
        except BaseException:
            child.close()
            self.close()
            raise

    def transcribe(self, audio, timeout=None, initial_prompt=None, cancel_event=None):
        """Drain one tagged reply even when its caller has moved on."""
        samples = _audio16(audio)
        if timeout is None:
            timeout = 25.0
        if timeout <= 0:
            raise TimeoutError("Whisper Turbo deadline exceeded")
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError("Whisper Turbo request already cancelled")
        connection, process = self.connection, self.process
        if process is None or not process.is_alive() or connection is None:
            raise RuntimeError("Whisper Turbo worker unavailable")
        self._request_id += 1
        request_id = self._request_id
        deadline = time.monotonic() + timeout
        # The optional caller can leave at its soft deadline. A separate hard
        # limit also covers a blocked pipe send, before polling can begin.
        watchdog = threading.Timer(timeout + 30.0, self.close)
        watchdog.daemon = True
        watchdog.start()
        try:
            connection.send({"id": request_id, "audio": samples, "initial_prompt": initial_prompt})
            while not connection.poll(0.1):
                if not process.is_alive():
                    raise RuntimeError("Whisper Turbo worker exited during decode")
            response = connection.recv()
        except (EOFError, BrokenPipeError, OSError) as exc:
            raise RuntimeError("Whisper Turbo worker communication failed") from exc
        finally:
            watchdog.cancel()
            watchdog.join()
        if response.get("type") != "transcript" or response.get("id") != request_id:
            raise RuntimeError("Whisper Turbo worker error: " + response.get("message", "unexpected reply"))
        if time.monotonic() >= deadline or (cancel_event is not None and cancel_event.is_set()):
            log.info("Discarded obsolete Whisper reply request=%d", request_id)
            raise CancelledError("Whisper Turbo reply no longer needed")
        self.last_stop = response.get("stop", "unknown")
        return response["text"]

    def cancel(self):
        """Force-stop a blocked IPC send/decode from a separate watchdog thread."""
        self.close()

    def close(self):
        with self._lock:
            process, connection = self.process, self.connection
            self.process = self.connection = None
        if connection is not None:
            connection.close()
        if process is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
