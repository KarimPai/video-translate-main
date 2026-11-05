import base64
import json
import logging
import shutil
import subprocess

import numpy as np

from ..config import AsrConfig

log = logging.getLogger(__name__)


class HaasASR:
    """Haas-ja-v1.0 through speech.cpp's persistent Vulkan worker."""

    def __init__(self, cfg: AsrConfig) -> None:
        self.cfg = cfg
        self.request_counter = 0

        speech = shutil.which("speech")
        if speech is None:
            raise RuntimeError(
                "Haas ASR requires speech.cpp. Install the Windows x64 Vulkan "
                "release and make sure speech.exe is on PATH."
            )

        command = [speech, "worker", cfg.haas_model, "--device", cfg.haas_device]
        if not cfg.haas_decode_on_cpu:
            command.append("--decode-on-cpu=false")

        try:
            self.process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=None, text=True, encoding="utf-8", bufsize=1,
            )
        except OSError as exc:
            raise RuntimeError(f"Could not start speech.cpp worker: {exc}") from exc

        response = self.process.stdout.readline()
        if not response:
            code = self.process.poll()
            raise RuntimeError(
                f"speech.cpp worker exited during startup (exit code {code}). "
                "Run speech info haas-ja-v1.0 and speech devices in CMD "
                "to check the model and Vulkan device."
            )

        try:
            status = json.loads(response)
        except json.JSONDecodeError as exc:
            self.process.terminate()
            raise RuntimeError(
                f"speech.cpp returned invalid startup data: {response.strip()!r}"
            ) from exc

        if status.get("type") != "ready":
            self.process.terminate()
            raise RuntimeError(f"Haas worker failed to start: {status}")

        log.info(
            "Haas ASR ready (%s, %s; speech.cpp %s)",
            cfg.haas_model, cfg.haas_device, status.get("version", "unknown version"),
        )

    def transcribe(self, audio: np.ndarray, probe: bool = False) -> str:
        if self.process.poll() is not None:
            raise RuntimeError("Haas ASR worker is no longer running.")

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
            response = self.process.stdout.readline()
            if not response:
                raise RuntimeError("Haas ASR worker stopped responding.")
            result = json.loads(response)
            if result.get("id") != request_id:
                continue
            if result.get("type") == "error":
                raise RuntimeError(f"Haas transcription failed: {result}")
            if result.get("type") == "end":
                text = result.get("text", "").strip()
                break

        if not text:
            return ""
        if any(phrase in text for phrase in self.cfg.blocklist):
            log.debug("Discarding blocked Haas text: %s", text)
            return ""
        return text
