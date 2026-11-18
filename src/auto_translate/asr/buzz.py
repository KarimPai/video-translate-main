"""Japanese BuzzASR adapter for 16 kHz VAD utterances."""
import base64
import json
import queue
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from ..config import AsrConfig
from ..runtime_diagnostics import update_metrics
from ..text_filter import trim_repetition_tail
from ..windows_subprocess import hidden_console_flags, owned_popen
from .quality_guard import transcript_problem


class BuzzASR:
    def __init__(self, cfg: AsrConfig):
        self.cfg = cfg
        self.last_raw_transcript = ""
        self.last_rejection = None
        self._generation = 0
        self._closed = False
        self._state = threading.Lock()
        self._decode = threading.Lock()
        self._responses = queue.Queue()
        if not Path(cfg.model_path, "model.safetensors").is_file():
            raise FileNotFoundError(f"BuzzASR checkpoint missing from {cfg.model_path}. Run Setup.bat.")
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
        return json.loads(line)

    def cancel_pending(self):
        with self._state:
            self._generation += 1
            self.last_raw_transcript = ""
            self.last_rejection = None

    def transcribe(self, audio, probe=False):
        audio = np.asarray(audio, dtype="<f4")
        if audio.ndim != 1 or audio.size > 60 * 16000 or not np.isfinite(audio).all():
            raise ValueError("BuzzASR requires finite mono audio of at most 60 seconds at 16 kHz")
        with self._state:
            generation = self._generation
            self.last_raw_transcript = ""
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
                self._proc.stdin.write(base64.b64encode(audio.tobytes()) + b"\n")
                self._proc.stdin.flush()
                raw = self._receive(self.cfg.response_timeout_s)["text"]
            except TimeoutError:
                raise
            except (OSError, RuntimeError, ValueError):
                if self._closed and generation != self._generation:
                    return ""
                raise
            with self._state:
                if self._closed or generation != self._generation:
                    return ""
                self.last_raw_transcript = raw
                text = trim_repetition_tail(raw.strip())
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
        if self._proc.poll() is None:
            self._proc.kill()
        self._proc.wait()
        self._reader.join()
        self._proc.stdin.close()
        self._proc.stdout.close()
