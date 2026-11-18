"""Optional Japanese Whisper Large-v3 Turbo ASR in a killable CTranslate2 process.

Keep CUDA allocations isolated from Haas Vulkan and llama.cpp. The parent can
enforce a deadline and release VRAM without leaving a timed-out decode running.
"""
import logging
import inspect
import math
import os
from ..windows_subprocess import check_running
from concurrent.futures import CancelledError
import multiprocessing as mp
import threading
import time
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version

import numpy as np

from ..text_filter import trim_repetition_tail
from ..runtime_diagnostics import rotate_log

log = logging.getLogger(__name__)
MODEL = "h2oai/faster-whisper-large-v3-turbo"

_DECODE_DEFAULTS = {
    "patience": 1.0, "length_penalty": 1.0, "temperature": 0.0, "best_of": 5,
    "repetition_penalty": 1.0, "no_repeat_ngram_size": 0, "word_timestamps": False,
    "log_prob_threshold": -1.0, "no_speech_threshold": 0.6,
    "compression_ratio_threshold": 2.4, "suppress_tokens": [-1], "max_new_tokens": None,
}


def whisper_decode_options(cfg):
    """Forward only reviewed decoding controls from the typed dual-ASR config."""
    return {name: getattr(cfg, "secondary_" + name, default)
            for name, default in _DECODE_DEFAULTS.items()}


def _supported_decode_options(model, options):
    """Inspect the actual installed implementation, including older runtimes."""
    try:
        parameters = inspect.signature(model.transcribe).parameters
    except (TypeError, ValueError):
        # Keep the already required runtime contract. New optional options and
        # hints are omitted if a wrapper hides its signature.
        return {}, sorted(options)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return options, []
    return ({key: value for key, value in options.items() if key in parameters},
            sorted(key for key in options if key not in parameters))


def _audio16(audio):
    """Validate the existing 16 kHz mono float samples, without resampling."""
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Whisper Turbo expects finite 16 kHz mono audio")
    return np.ascontiguousarray(np.clip(samples, -1, 1))


def _decode(model, samples, beam_size, initial_prompt=None, *, temperature=None,
            hotwords=None, decode_options=None, with_metadata=False):
    """Never let Whisper auto-detect English or choose the translation task."""
    if initial_prompt and hotwords:
        raise ValueError("Choose hotwords or initial_prompt, never both")
    for hint in (initial_prompt, hotwords):
        if hint is not None and (not isinstance(hint, str) or len(hint) > 256):
            raise ValueError("Whisper utterance hints must be bounded text")
    options = dict(_DECODE_DEFAULTS)
    if decode_options:
        unknown = decode_options.keys() - _DECODE_DEFAULTS.keys()
        if unknown:
            raise ValueError("Unsupported decoding controls: " + ", ".join(sorted(unknown)))
        options.update(decode_options)
    if temperature is not None:
        options["temperature"] = temperature
    # Hints must come from the caller's acoustic-grounded utterance evidence.
    # No video title, channel title or static member list is added here.
    options["initial_prompt"] = initial_prompt
    if hotwords:
        options["hotwords"] = hotwords
    supported, unsupported = _supported_decode_options(model, options)
    if unsupported:
        log.info("Installed Whisper transcribe omits unsupported options: %s", unsupported)
    arguments = dict(task="transcribe", language="ja", beam_size=beam_size,
                     condition_on_previous_text=False, vad_filter=False,
                     without_timestamps=False)
    arguments.update(supported)
    segments, _ = model.transcribe(samples, **arguments)
    phrases = []
    confidence_segments = []
    for segment in segments:  # faster-whisper decodes lazily.
        if (getattr(segment, "no_speech_prob", 0.0) > 0.75
                and getattr(segment, "avg_logprob", 0.0) < -0.75):
            continue
        value = segment.text.strip()
        if value:
            phrases.append(trim_repetition_tail(value))
            evidence = {"text": value[:240]}
            for name in ("avg_logprob", "no_speech_prob", "compression_ratio", "start", "end"):
                number = getattr(segment, name, None)
                if type(number) in (int, float) and math.isfinite(number):
                    evidence[name] = float(number)
            confidence_segments.append(evidence)
    transcript = " ".join(phrases).strip()
    if not with_metadata:
        return transcript
    # These are Whisper's own uncalibrated signals, not probabilities of the
    # transcript being correct and never a comparable score for Haas.
    mode = ("initial_prompt" if supported.get("initial_prompt") else
            "hotwords" if supported.get("hotwords") else "none")
    metadata = {"segments": confidence_segments[:24],
                "decoding": {"prompt_mode": mode, "beam_size": beam_size,
                             "temperature": supported.get("temperature", "runtime_default"),
                             "unsupported_options": unsupported}}
    for name in ("avg_logprob", "no_speech_prob", "compression_ratio"):
        values = [part[name] for part in confidence_segments if name in part]
        if values:
            metadata[name] = sum(values) / len(values)
    return transcript, metadata


def _worker_main(connection, model_name, device, compute_type, beam_size, decode_options=None):
    try:
        Path("logs").mkdir(exist_ok=True)
        rotate_log(Path("logs/whisper-turbo.log"))
        logging.basicConfig(filename="logs/whisper-turbo.log", encoding="utf-8",
                            level=logging.INFO, force=True)
        from auto_translate.cuda_dlls import setup
        setup()
        from faster_whisper import WhisperModel
        model = WhisperModel(model_name, device=device,
                             compute_type=compute_type, cpu_threads=4,
                             num_workers=1, local_files_only=True)
        try:
            runtime_version = version("faster-whisper")
        except PackageNotFoundError:
            runtime_version = "unknown"
        connection.send({"type": "ready", "faster_whisper_version": runtime_version})
        while True:
            command = connection.recv()
            if command is None:
                break
            try:
                samples = _audio16(command["audio"])
                prompt = command.get("initial_prompt")
                transcript, confidence = _decode(model, samples, beam_size, initial_prompt=prompt,
                                                 hotwords=command.get("hotwords"),
                                                 decode_options=decode_options, with_metadata=True)
                confidence["decoding"]["runtime_version"] = runtime_version
                connection.send({"type": "transcript", "text": transcript, "id": command["id"],
                                 "stop": "complete" if transcript else "unknown",
                                 "confidence": confidence})
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
                 beam_size=6, startup_timeout=600, decode_options=None):
        self.last_stop = "unknown"
        self.last_confidence = {}
        self.last_prompt_mode = "none"
        self._lock = threading.Lock()
        self._request_id = 0
        self._decode_lock = threading.Lock()
        self.process = None
        self.connection = None
        context = mp.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        self.connection = parent
        self.process = context.Process(
            target=_worker_main,
            args=(child, model, device, compute_type, beam_size, decode_options),
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
                if self.connection.poll(0.2):
                    response = self.connection.recv()
                    if response.get("type") == "ready":
                        self.runtime_version = response.get("faster_whisper_version", "unknown")
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

    def transcribe(self, audio, timeout=None, initial_prompt=None, cancel_event=None, *, hotwords=None):
        """Keep a single reader and writer on the worker pipe across requests."""
        if not self._decode_lock.acquire(blocking=False):
            raise RuntimeError("Whisper Turbo request already in progress")
        try:
            return self._transcribe(audio, timeout, initial_prompt, cancel_event, hotwords=hotwords)
        finally:
            self._decode_lock.release()

    def _transcribe(self, audio, timeout, initial_prompt, cancel_event, *, hotwords=None):
        samples = _audio16(audio)
        if timeout is None:
            timeout = 25.0
        if timeout <= 0:
            raise TimeoutError("Whisper Turbo deadline exceeded")
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError("Whisper Turbo request already cancelled")
        with self._lock:
            connection, process = self.connection, self.process
            self._request_id += 1
            request_id = self._request_id
        if process is None or not process.is_alive() or connection is None:
            raise RuntimeError("Whisper Turbo worker unavailable")
        deadline = time.monotonic() + timeout
        # The optional caller can leave at its soft deadline. A separate hard
        # limit also covers a blocked pipe send, before polling can begin.
        watchdog = threading.Timer(timeout + 30.0, self.close)
        watchdog.daemon = True
        watchdog.start()
        try:
            connection.send({"id": request_id, "audio": samples, "initial_prompt": initial_prompt,
                             "hotwords": hotwords})
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
        finally:
            watchdog.cancel()
            watchdog.join()
        if response.get("type") != "transcript" or response.get("id") != request_id:
            raise RuntimeError("Whisper Turbo worker error: " + response.get("message", "unexpected reply"))
        if time.monotonic() >= deadline or (cancel_event is not None and cancel_event.is_set()):
            log.info("Discarded obsolete Whisper reply request=%d", request_id)
            raise CancelledError("Whisper Turbo reply no longer needed")
        self.last_stop = response.get("stop", "unknown")
        confidence = response.get("confidence")
        self.last_confidence = confidence if isinstance(confidence, dict) else {}
        decoding = self.last_confidence.get("decoding", {})
        self.last_prompt_mode = decoding.get("prompt_mode", "none") if isinstance(decoding, dict) else "none"
        return response["text"]

    def close(self):
        with self._lock:
            process, connection = self.process, self.connection
            self.process = self.connection = None
        if connection is not None:
            connection.close()
        if process is not None and process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
