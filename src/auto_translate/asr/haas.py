import base64
import json
import logging
import queue
import threading
import shutil
import subprocess
import time

import numpy as np

from ..config import AsrConfig
from ..text_filter import rejection_reason

log = logging.getLogger(__name__)


class HaasASR:
    """Haas-ja-v1.0 through speech.cpp's persistent Vulkan worker."""

    def __init__(self, cfg: AsrConfig, *, language: str | None = None) -> None:
        self.cfg = cfg
        self.request_counter = 0
        self.last_stop = None

        speech = shutil.which("speech")
        if speech is None:
            raise RuntimeError(
                "Haas ASR requires speech.cpp. Install the Windows x64 Vulkan "
                "release and make sure speech.exe is on PATH."
            )

        command = [
            speech, "worker", cfg.haas_model,
            "--device", cfg.haas_device,
        ]
        if language:
            command.extend(["--language", language])
        else:
            command.append(f"--decode-on-cpu={str(cfg.haas_decode_on_cpu).lower()}")


        try:
            self.process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=None, text=True, encoding="utf-8", bufsize=1,
            )
        except OSError as exc:
            raise RuntimeError(f"Could not start speech.cpp worker: {exc}") from exc

        self._responses: queue.Queue[str | None] = queue.Queue()
        # Avoid an unbounded Windows PIPE.readline() during a hung inference.
        self._reader = threading.Thread(
            target=self._pump_output, args=(self.process.stdout,),
            name=f"{cfg.haas_model}-stdout", daemon=True,
        )
        self._reader.start()
        try:
            response = self._read_response()
            status = json.loads(response)
        except Exception as exc:
            self.close()
            raise RuntimeError(
                f"speech.cpp returned invalid startup response: {exc}"
            ) from exc

        if status.get("type") != "ready":
            self.close()
            raise RuntimeError(f"Haas worker failed to start: {status}")

        log.info(
            "speech.cpp worker ready (%s, %s; speech.cpp %s)",
            cfg.haas_model, cfg.haas_device, status.get("version", "unknown version"),
        )

    def _pump_output(self, stdout) -> None:
        try:
            for line in stdout:
                self._responses.put(line)
        finally:
            self._responses.put(None)

    def _read_response(self, timeout=None) -> str:
        try:
            response = self._responses.get(timeout=self.cfg.response_timeout_s if timeout is None else timeout)
        except queue.Empty as exc:
            raise TimeoutError(
                f"Haas speech.cpp worker unresponsive for {self.cfg.response_timeout_s:.1f}s"
            ) from exc
        if response is None:
            code = self.process.poll() if self.process is not None else None
            raise RuntimeError(f"Haas speech.cpp worker closed stdout (exit code {code})")
        return response

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process is None:
            return
        self.process = None
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3.0)
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                stream.close()

    def transcribe(self, audio: np.ndarray, probe: bool = False, *, timeout=None) -> str:
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Haas ASR worker is no longer running.")

        deadline = time.monotonic() + (timeout if timeout is not None else self.cfg.response_timeout_s)
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        pcm_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
        self.request_counter += 1
        request_id = str(self.request_counter)
        pcm = base64.b64encode(pcm_int16.tobytes()).decode("ascii")

        self.process.stdin.write(json.dumps({
            "type": "chunk", "id": request_id, "seq": 0, "pcm": pcm,
        }) + "\n")
        self.process.stdin.write(json.dumps({
            "type": "transcribe", "id": request_id, "sample_rate": 16000,
        }) + "\n")
        self.process.stdin.flush()

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("speech.cpp recognition deadline expired")
            response = self._read_response(remaining)
            result = json.loads(response)
            if result.get("id") != request_id:
                continue
            if result.get("type") == "error":
                raise RuntimeError(f"Haas transcription failed: {result}")
            if result.get("type") == "end":
                self.last_stop = result.get("stop")
                text = result.get("text", "").strip()
                break

        if not text:
            return ""
        if any(phrase in text for phrase in self.cfg.blocklist):
            log.debug("Discarding blocked Haas text: %s", text)
            return ""

        reason = rejection_reason(text)
        if reason is not None:
            log.debug("Discarding Haas noise (%s): %s", reason, text)
            return ""

        return text
