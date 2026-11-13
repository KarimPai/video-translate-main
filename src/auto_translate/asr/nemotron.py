"""Japanese Parakeet TDT-CTC 0.6B served by CrispASR Windows CUDA."""
import io
import json
import logging
import os
import shutil
import socket
import subprocess
import time
import urllib.request
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
MODEL = "parakeet-tdt-0.6b-ja-q8_0.gguf"


def _executable():
    override = os.environ.get("CRISPASR_EXE")
    if override:
        return override
    found = shutil.which("crispasr")
    if found:
        return found
    local = Path("tools/crispasr/crispasr.exe")
    if local.is_file():
        return str(local.resolve())
    raise FileNotFoundError("Install CrispASR Windows CUDA and set CRISPASR_EXE")


def _model():
    path = Path(os.environ.get("PARAKEET_JA_MODEL", str(Path("models") / MODEL)))
    if not path.is_file():
        raise FileNotFoundError(f"Parakeet JA GGUF missing: {path}; set PARAKEET_JA_MODEL")
    return str(path.resolve())


def _wav16(audio):
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("Parakeet requires finite 16kHz mono PCM")
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm)
    return buffer.getvalue()


@dataclass(frozen=True)
class LanguageTranscript:
    text: str
    language: str | None = None


def _multipart(audio_wav, *, language="ja", response_format="json"):
    boundary = "----jp-asr-" + uuid.uuid4().hex
    fields = [("language", language), ("response_format", response_format)]
    data = bytearray()
    for key, value in fields:
        data.extend((f"--{boundary}\r\nContent-Disposition: form-data; "
                     f"name=\"{key}\"\r\n\r\n{value}\r\n").encode("ascii"))
    data.extend((f"--{boundary}\r\nContent-Disposition: form-data; "
                 'name="file"; filename="utterance.wav"\r\n'
                 "Content-Type: audio/wav\r\n\r\n").encode("ascii"))
    data.extend(audio_wav)
    data.extend(f"\r\n--{boundary}--\r\n".encode("ascii"))
    return bytes(data), boundary


class ParakeetASR:
    def __init__(self, device="cuda:0", startup_timeout=300):
        self.process = None
        self.last_stop = "unknown"  # CrispASR API does not report model stop causes.
        self.device = device
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        command = [_executable(), "--server", "--backend", "parakeet", "-m", _model(),
                   "--host", "127.0.0.1", "--port", str(port),
                   "-l", "ja"]
        Path("logs").mkdir(exist_ok=True)
        with Path("logs/parakeet.log").open("a", encoding="utf-8") as log_file:
            self.process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL,
                stdout=log_file, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + startup_timeout
            while self.process.poll() is None and time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(self.url + "/health", timeout=1) as response:
                        if response.status == 200:
                            log.info("Parakeet JA ASR server ready on %s (%s)", self.url, device)
                            return
                except OSError:
                    pass
                time.sleep(0.3)
            raise RuntimeError("Parakeet ASR startup failed; see logs/parakeet.log")
        except BaseException:
            self.close()
            raise

    def _request_transcription(self, audio, timeout, *, language, response_format):
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Parakeet ASR server exited")
        if timeout is None:
            timeout = 25.0
        if timeout <= 0:
            raise TimeoutError("Parakeet deadline exceeded")
        payload, boundary = _multipart(
            _wav16(audio), language=language, response_format=response_format
        )
        request = urllib.request.Request(
            self.url + "/v1/audio/transcriptions",
            data=payload,
            headers={"Content-Type": "multipart/form-data; boundary=" + boundary},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        if not isinstance(result, dict) or not isinstance(result.get("text"), str):
            raise ValueError("Parakeet returned no transcription text")
        self.last_stop = "unknown"
        return result

    def transcribe(self, audio, timeout=None):
        """Previous forced-Japanese route; keep compatibility and accuracy fallback."""
        result = self._request_transcription(
            audio, timeout, language="ja", response_format="json"
        )
        return result["text"].strip()

    def transcribe_with_language(self, audio, timeout=None):
        raise NotImplementedError("Parakeet JA cannot identify spoken language")

    def close(self):
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
