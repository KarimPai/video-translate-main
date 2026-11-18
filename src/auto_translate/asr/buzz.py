"""Japanese BuzzASR adapter for 16 kHz VAD utterances."""
import base64
import json
import logging
import queue
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from ..config import AsrConfig
from ..runtime_diagnostics import update_metrics, record_event
from ..text_artifacts import clean_asr_text
from ..text_filter import trim_repetition_tail
from ..windows_subprocess import hidden_console_flags, owned_popen
from .quality_guard import transcript_problem

log = logging.getLogger(__name__)


class BuzzWorkerError(RuntimeError):
    """Fatal error reported by the isolated decoder."""


class BuzzASR:
    def __init__(self, cfg: AsrConfig):
        self.cfg = cfg
        self.last_raw_transcript = ""
        self.last_cleaned_transcript = ""
        self.last_rejection = None
        self._generation = 0
        self._closed = False
        self._state = threading.Lock()
        self._decode = threading.Lock()
        self._lifecycle = threading.RLock()
        self._restarts = 0
        self._responses = queue.Queue()
        if not Path(cfg.model_path, "model.safetensors").is_file():
            raise FileNotFoundError(f"BuzzASR checkpoint missing from {cfg.model_path}. Run Setup.bat.")
        self._start_worker()

    def _start_worker(self):
        cfg = self.cfg
        self._responses = queue.Queue()
        self._proc = owned_popen(
            [cfg.python_path, "-u", str(Path(__file__).with_name("buzz_worker.py")),
             cfg.model_path, f"cuda:{cfg.cuda_device}"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, creationflags=hidden_console_flags(),
        )
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        try:
            if self._receive(cfg.startup_timeout_s).get("ready") is not True:
                raise RuntimeError("Unexpected BuzzASR startup response")
        except BaseException:
            self.close()
            raise
        update_metrics(workers={"asr": "ready (BuzzASR Japanese)"})

    def _read(self):
        try:
            for line in self._proc.stdout:
                self._responses.put(line)
        finally:
            self._responses.put(None)

    def _receive(self, timeout):
        try:
            line = self._responses.get(timeout=timeout)
        except queue.Empty as exc:
            self.close()
            raise TimeoutError("BuzzASR worker timed out") from exc
        if line is None:
            raise RuntimeError("BuzzASR worker exited. Check its startup or inference error in the log.")
        response = json.loads(line)
        if "error" in response:
            raise BuzzWorkerError(response["error"])
        return response

    def _restart_worker(self, error):
        with self._lifecycle:
            if self._closed:
                return False
            # A CUDA fault invalidates the entire worker context. Retry once in
            # a fresh process, never reuse the poisoned model or loop forever.
            if self._restarts >= 1 or "CUDA" not in str(error):
                raise error
            self._restarts += 1
            log.warning("Restarting BuzzASR worker after CUDA failure: %s", error)
            self._stop_worker()
            self._start_worker()
            return True

    def cancel_pending(self):
        with self._state:
            self._generation += 1
            self.last_raw_transcript = ""
            self.last_cleaned_transcript = ""
            self.last_rejection = None

    def transcribe(self, audio, probe=False):
        audio = np.asarray(audio, dtype="<f4")
        if audio.ndim != 1 or audio.size > 60 * 16000 or not np.isfinite(audio).all():
            raise ValueError("BuzzASR requires finite mono audio of at most 60 seconds at 16 kHz")
        with self._state:
            generation = self._generation
            self.last_raw_transcript = ""
            self.last_cleaned_transcript = ""
            self.last_rejection = None
            if self._closed or not audio.size or not np.any(audio):
                self.last_rejection = "no_speech"
                return ""
            if probe:
                # The monolingual decoder cannot confirm speech on a VAD-negative probe.
                self.last_rejection = "probe_unconfirmed"
                return ""
        started = time.monotonic()
        with self._decode:
            if self._closed or generation != self._generation:
                return ""
            try:
                payload = base64.b64encode(audio.tobytes()) + b"\n"
                for attempt in range(2):
                    try:
                        self._proc.stdin.write(payload)
                        self._proc.stdin.flush()
                        raw = self._receive(self.cfg.response_timeout_s)["text"]
                        break
                    except BuzzWorkerError as exc:
                        if attempt:
                            raise
                        if not self._restart_worker(exc) or generation != self._generation:
                            return ""
            except TimeoutError:
                raise
            except (OSError, RuntimeError, ValueError):
                if self._closed and generation != self._generation:
                    return ""
                # A final ASR failure leaves the current worker poisoned. Stop it
                # here instead of waiting for pipeline shutdown, so a short-lived
                # CUDA/transport fault cannot leave an orphaned GPU process behind.
                self.close()
                raise
            with self._state:
                if self._closed or generation != self._generation:
                    return ""
                self.last_raw_transcript = raw
                cleaned = clean_asr_text(raw)
                text = trim_repetition_tail(cleaned.text)
                self.last_cleaned_transcript = text
                if cleaned.removed:
                    record_event("asr_artifacts_removed", raw=raw[:512], cleaned=text[:512],
                                 removed=cleaned.removed[:16])
                self.last_rejection = transcript_problem(text, audio.size / 16000)
                if any(word in text for word in self.cfg.blocklist):
                    self.last_rejection = "blocklist"
                update_metrics(stages={"asr_s": round(time.monotonic() - started, 3)})
                return "" if self.last_rejection else text

    def close(self):
        with self._state:
            if self._closed:
                return
            self._closed = True
            self._generation += 1
        with self._lifecycle:
            self._stop_worker()

    def _stop_worker(self):
        if self._proc.poll() is None:
            self._proc.kill()
        self._proc.wait()
        self._reader.join()
        self._proc.stdin.close()
        self._proc.stdout.close()
