import base64
import json
import logging
import subprocess

import numpy as np

from ..config import AsrConfig

log = logging.getLogger(__name__)


class HaasASR:
    def __init__(self, cfg: AsrConfig) -> None:
        self.cfg = cfg
        self.request_counter = 0

        self.process = subprocess.Popen(
            [
                "speech",
                "worker",
                cfg.haas_model,
                "--device",
                "Vulkan0",
                "--decode-on-cpu=false",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

        response = self.process.stdout.readline()

        if not response:
            raise RuntimeError("Haas worker exited during startup.")

        status = json.loads(response)

        if status.get("type") != "ready":
            raise RuntimeError(f"Haas worker failed to start: {status}")

        log.info("Haas ASR worker ready: %s", cfg.haas_model)

    def transcribe(self, audio: np.ndarray, probe: bool = False) -> str:
        if self.process.poll() is not None:
            raise RuntimeError("Haas worker is no longer running.")

        audio = np.ascontiguousarray(audio, dtype=np.float32)
        pcm_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")

        self.request_counter += 1
        request_id = str(self.request_counter)
        pcm = base64.b64encode(pcm_int16.tobytes()).decode("ascii")

        self.process.stdin.write(json.dumps({
            "type": "chunk",
            "id": request_id,
            "seq": 0,
            "pcm": pcm,
        }) + "\n")

        self.process.stdin.write(json.dumps({
            "type": "transcribe",
            "id": request_id,
            "sample_rate": 16000,
        }) + "\n")

        self.process.stdin.flush()

        while True:
            response = self.process.stdout.readline()

            if not response:
                raise RuntimeError("Haas worker stopped responding.")

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