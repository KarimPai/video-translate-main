import base64
import json
import logging
import shutil
import subprocess
import tomllib
from pathlib import Path

import numpy as np

from ..config import AsrConfig
from ..text_filter import is_whole_utterance_blocked, rejection_reason

log = logging.getLogger(__name__)


def load_haas_keywords(filename: str) -> str:
    """Load a small, editable, newline-separated Haas keyword list from TOML.

    speech.cpp takes a *string*, not a JSON array, in worker requests.
    Each line is preferred written form, optionally =kana readings.
    """
    path = Path(filename)
    if not path.is_absolute():
        # Works from the launcher directory and from an installed package.
        from_cwd = Path.cwd() / path
        path = from_cwd if from_cwd.is_file() else Path(__file__).resolve().parents[3] / path
    try:
        with path.open("rb") as source:
            data = tomllib.load(source)
    except FileNotFoundError:
        log.warning("Haas vocabulary file not found: %s; continuing without keyword boosting", path)
        return ""

    entries = data.get("keywords")
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected a top-level keywords = [...] array")
    seen: set[str] = set()
    selected: list[str] = []
    for index, item in enumerate(entries, start=1):
        if not isinstance(item, str) or not item.strip() or any(c in item for c in "\r\n"):
            raise ValueError(f"{path}: invalid keyword at index {index}")
        keyword = item.strip()
        written, separator, readings = keyword.partition("=")
        if not written.strip() or (separator and not readings.strip()):
            raise ValueError(f"{path}: invalid keyword format at index {index}")
        if written in seen:
            log.warning("Ignoring duplicate Haas vocabulary entry: %s", written)
            continue
        seen.add(written)
        selected.append(keyword)
    return "\n".join(selected)


class HaasASR:
    """Haas-ja-v1.0 through speech.cpp's persistent Vulkan worker."""

    def __init__(self, cfg: AsrConfig) -> None:
        self.cfg = cfg
        self.request_counter = 0
        self.keywords = ""
        if cfg.haas_keywords_enabled:
            self.keywords = load_haas_keywords(cfg.haas_keywords_file)
            if self.keywords:
                log.info(
                    "Haas keyword recognition enabled (%d terms, alpha=%.2f, guard=%s)",
                    len(self.keywords.splitlines()),
                    cfg.haas_keywords_alpha,
                    cfg.haas_keywords_guard,
                )

        speech = shutil.which("speech")
        if speech is None:
            raise RuntimeError(
                "Haas ASR requires speech.cpp. Install the Windows x64 Vulkan "
                "release and make sure speech.exe is on PATH."
            )

        command = [
            speech, "worker", cfg.haas_model,
            "--device", cfg.haas_device,
            f"--decode-on-cpu={str(cfg.haas_decode_on_cpu).lower()}",
        ]

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

    def _transcribe_pcm(self, pcm: str, *, with_keywords: bool) -> str:
        self.request_counter += 1
        request_id = str(self.request_counter)
        message = {"type": "transcribe", "id": request_id, "sample_rate": 16000}
        if with_keywords:
            message.update({
                "keywords": self.keywords,
                "keywords_alpha": self.cfg.haas_keywords_alpha,
                "keywords_guard": self.cfg.haas_keywords_guard,
            })

        self.process.stdin.write(json.dumps({
            "type": "chunk", "id": request_id, "seq": 0, "pcm": pcm,
        }) + "\n")
        self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

        while True:
            response = self.process.stdout.readline()
            if not response:
                raise RuntimeError("Haas ASR worker stopped responding.")
            result = json.loads(response)
            if result.get("id") != request_id:
                continue
            if result.get("type") == "error":
                error = result.get("error", {})
                option = error.get("option", "") if isinstance(error, dict) else ""
                detail = error.get("message", "") if isinstance(error, dict) else str(error)
                if with_keywords and (
                    option in {"keywords", "keywords_alpha", "keywords_guard"}
                    or (
                        isinstance(error, dict)
                        and "keyword" in detail.lower()
                        and error.get("code") in {"unsupported", "invalid_argument"}
                    )
                ):
                    log.warning(
                        "speech.cpp rejected Haas keyword recognition (%s); "
                        "falling back to unboosted ASR for this session", detail
                    )
                    self.keywords = ""
                    return self._transcribe_pcm(pcm, with_keywords=False)
                raise RuntimeError(f"Haas transcription failed: {result}")
            if result.get("type") == "end":
                return result.get("text", "").strip()

    def transcribe(self, audio: np.ndarray, probe: bool = False) -> str:
        if self.process.poll() is not None:
            raise RuntimeError("Haas ASR worker is no longer running.")

        audio = np.ascontiguousarray(audio, dtype=np.float32)
        pcm_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
        pcm = base64.b64encode(pcm_int16.tobytes()).decode("ascii")

        # Fallback probes check short/uncertain audio; avoid doubling their decode cost.
        text = self._transcribe_pcm(pcm, with_keywords=bool(self.keywords and not probe))
        if not text:
            return ""
        if is_whole_utterance_blocked(text, self.cfg.blocklist):
            log.debug("Discarding blocked Haas text: %s", text)
            return ""
        reason = rejection_reason(text)
        if reason is not None:
            log.debug("Discarding Haas noise (%s): %s", reason, text)
            return ""
        return text
