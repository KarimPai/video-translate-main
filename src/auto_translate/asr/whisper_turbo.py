"""Optional Japanese Whisper Large-v3 Turbo ASR in a killable CTranslate2 process.

Keep CUDA allocations isolated from Haas Vulkan and llama.cpp. The parent can
enforce a deadline and release VRAM without leaving a timed-out decode running.
"""
import logging
import os
from ..windows_subprocess import check_running
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
                connection.send({"type": "error", "id": command["id"], "message": str(exc)})
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
                 beam_size=6, startup_timeout=600, cancel_event=None):
        self.last_stop = "unknown"
        self._lock = threading.Lock()
        self._request_id = 0
        self._decode_lock = threading.Lock()
        self._inflight = None
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
            check_running()
            self.process.start()
            log.info("session=%s name=whisper-turbo pid=%s parent=%s owned=true",
                     os.getenv("VIDEO_TRANSLATE_SESSION", "cli"), self.process.pid, os.getpid())
            child.close()
            deadline = time.monotonic() + startup_timeout
            while time.monotonic() < deadline:
                check_running()
                if cancel_event is not None and cancel_event.is_set():
                    raise CancelledError("Whisper Turbo startup cancelled")
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
        """Keep a single reader and writer on the worker pipe across requests."""
        if not self._decode_lock.acquire(blocking=False):
            raise CancelledError("Whisper Turbo request already in progress")
        try:
            return self._transcribe(audio, timeout, initial_prompt, cancel_event)
        finally:
            self._decode_lock.release()

    def wait_until_idle(self, timeout, cancel_event=None):
        """Bounded final-cue wait for a canceled preview's child reply.

        Keep final scheduling outside the pipe reader and never submit a second
        job before the previous caller has released its single-flight lease.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            check_running()
            if cancel_event is not None and cancel_event.is_set():
                return False
            if not self._decode_lock.acquire(timeout=min(.05, max(0., deadline - time.monotonic()))):
                continue
            try:
                with self._lock:
                    inflight = getattr(self, "_inflight", None)
                    connection, process = self.connection, self.process
                if process is None or connection is None or not process.is_alive():
                    raise RuntimeError("Whisper Turbo worker unavailable")
                if inflight is None or connection.poll(0):
                    return True  # _transcribe drains the old response before send.
            finally:
                self._decode_lock.release()
            if cancel_event is not None:
                cancel_event.wait(.02)
            else:
                time.sleep(.02)
        return False

    def _transcribe(self, audio, timeout, initial_prompt, cancel_event):
        samples = _audio16(audio)
        if timeout is None:
            timeout = 25.0
        if timeout <= 0:
            raise TimeoutError("Whisper Turbo deadline exceeded")
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError("Whisper Turbo request already cancelled")
        with self._lock:
            connection, process = self.connection, self.process
        if process is None or not process.is_alive() or connection is None:
            raise RuntimeError("Whisper Turbo worker unavailable")
        # A canceled caller does not own the child job's lifetime. Keep its
        # watchdog and drain its reply before submitting any new audio. This
        # prevents seeks/previews from building a queue of obsolete decodes.
        previous = getattr(self, "_inflight", None)
        if previous is not None:
            try:
                if not connection.poll(0):
                    raise CancelledError("Whisper Turbo is finishing an obsolete request")
                stale = connection.recv()
            except (EOFError, BrokenPipeError, OSError) as exc:
                raise RuntimeError("Whisper Turbo worker communication failed") from exc
            if stale.get("id") != previous["id"]:
                raise RuntimeError("Unexpected Whisper Turbo reply while draining")
            self._finish_request(previous["id"])
            log.info("Drained obsolete Whisper reply request=%d", previous["id"])
        with self._lock:
            self._request_id += 1
            request_id = self._request_id
        deadline = time.monotonic() + timeout
        # The optional caller can leave at its soft deadline. A separate hard
        # limit also covers a blocked pipe send, before polling can begin.
        watchdog = threading.Timer(timeout + 30.0, self._abort_request, args=(request_id,))
        watchdog.daemon = True
        with self._lock:
            if self.connection is not connection or self.process is not process:
                raise CancelledError("Whisper Turbo worker closed before submission")
            self._inflight = {"id": request_id, "watchdog": watchdog}
        watchdog.start()
        try:
            connection.send({"id": request_id, "audio": samples, "initial_prompt": initial_prompt})
            while True:
                while not connection.poll(0.05):
                    check_running()
                    if not process.is_alive():
                        raise RuntimeError("Whisper Turbo worker exited during decode")
                    if cancel_event is not None and cancel_event.is_set():
                        raise CancelledError("Whisper Turbo request cancelled")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Whisper Turbo transcription timed out")
                response = connection.recv()
                if response.get("id") == request_id:
                    break
                if response.get("id") is not None and response.get("id") < request_id:
                    log.info("Discarded stale Whisper reply request=%d (waiting for %d)",
                             response.get("id"), request_id)
                    continue
                break
        except (EOFError, BrokenPipeError, OSError) as exc:
            raise RuntimeError("Whisper Turbo worker communication failed") from exc
        # Only receiving the job's reply clears its watchdog. Cancellation
        # leaves the deadline alive even when no further utterance arrives.
        self._finish_request(request_id)
        if response.get("type") != "transcript" or response.get("id") != request_id:
            raise RuntimeError("Whisper Turbo worker error: " + response.get("message", "unexpected reply"))
        if time.monotonic() >= deadline or (cancel_event is not None and cancel_event.is_set()):
            log.info("Discarded obsolete Whisper reply request=%d", request_id)
            raise CancelledError("Whisper Turbo reply no longer needed")
        self.last_stop = response.get("stop", "unknown")
        return response["text"]

    def _finish_request(self, request_id):
        with self._lock:
            inflight = getattr(self, "_inflight", None)
            if inflight is None or inflight["id"] != request_id:
                return
            self._inflight = None
        inflight["watchdog"].cancel()

    def _abort_request(self, request_id):
        with self._lock:
            inflight = getattr(self, "_inflight", None)
            if inflight is None or inflight["id"] != request_id:
                return
            process, connection = self.process, self.connection
            self.process = self.connection = None
            self._inflight = None
        log.warning("Whisper Turbo hard deadline expired for request=%d", request_id)
        self._close_process(process, connection)

    def close(self):
        with self._lock:
            process, connection = self.process, self.connection
            self.process = self.connection = None
            inflight = getattr(self, "_inflight", None)
            self._inflight = None
        if inflight is not None:
            inflight["watchdog"].cancel()
        self._close_process(process, connection)

    @staticmethod
    def _close_process(process, connection):
        if connection is not None:
            connection.close()
        if process is not None and process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
