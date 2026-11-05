import base64
import json
import logging
import subprocess
from pathlib import Path

import numpy as np

from ..config import AsrConfig

log = logging.getLogger(__name__)


class Qwen3ASR:
    def __init__(self, cfg: AsrConfig) -> None:
        self.cfg = cfg
        worker = Path(__file__).with_name("qwen_worker.py")

        self.process = subprocess.Popen(
            [cfg.qwen_python, str(worker), cfg.qwen_model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

        response = self.process.stdout.readline()

        if not response:
            raise RuntimeError("Qwen worker exited during startup.")

        status = json.loads(response)

        if not status.get("ready"):
            raise RuntimeError(
                f"Qwen worker failed to load: {status.get('error', 'Unknown error')}"
            )

        log.info("Qwen3-ASR worker ready: %s", cfg.qwen_model)

    def transcribe(self, audio: np.ndarray, probe: bool = False) -> str:
        if self.process.poll() is not None:
            raise RuntimeError("Qwen worker is no longer running.")

        audio = np.ascontiguousarray(audio, dtype=np.float32)
        encoded = base64.b64encode(audio.tobytes()).decode("ascii")

        request = json.dumps({"audio": encoded})

        self.process.stdin.write(request + "\n")
        self.process.stdin.flush()

        response = self.process.stdout.readline()

        if not response:
            raise RuntimeError("Qwen worker stopped responding.")

        result = json.loads(response)

        if "error" in result:
            raise RuntimeError(f"Qwen transcription failed: {result['error']}")

        text = result.get("text", "").strip()

        if any(phrase in text for phrase in self.cfg.blocklist):
            return ""

        return text