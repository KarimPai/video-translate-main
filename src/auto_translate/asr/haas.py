import base64
import json
import logging
import queue
import threading
import shutil
import subprocess
from ..windows_subprocess import check_running, owned_popen

from ..windows_subprocess import hidden_console_flags
import time

import numpy as np

from ..config import AsrConfig
from ..text_filter import rejection_reason, trim_repetition_tail

log = logging.getLogger(__name__)


class _KeywordOptionsRejected(RuntimeError):
    """A terminal worker error permits one unboosted retry of the same audio."""


def _advertised_options(status):
    """Read actual model capabilities; old Haas GGUFs have no vocabulary."""
    model = status.get("model")
    options = model.get("options", ()) if isinstance(model, dict) else ()
    if not isinstance(options, (list, tuple)):
        return set()
    return {option["name"] for option in options
            if isinstance(option, dict) and isinstance(option.get("name"), str)}


def _keyword_text(keywords):
    """Bound an explicitly reviewed utterance vocabulary, preserving its forms.

    speech.cpp rewrites readings to the written form. Callers must never map
    a short spoken nickname to a full stage name. This is not a title prompt.
    """
    if keywords is None:
        return ""
    entries = keywords.splitlines() if isinstance(keywords, str) else list(keywords)
    if len(entries) > 8 or any(not isinstance(item, str) for item in entries):
        raise ValueError("Haas vocabulary requires at most eight reviewed entries")
    entries = list(dict.fromkeys(item.strip() for item in entries if item.strip()))
    if any(len(item) > 64 or any(ord(char) < 32 for char in item) for item in entries):
        raise ValueError("Invalid Haas vocabulary entry")
    return "\n".join(entries)


class HaasASR:
    """Haas-ja-v1.0 through speech.cpp's persistent Vulkan worker."""

    def __init__(self, cfg: AsrConfig, *, language: str | None = None) -> None:
        self.cfg = cfg
        self._close_lock = threading.Lock()
        self.request_counter = 0
        self.last_stop = None
        self.supported_options = set()
        self.last_keyword_diagnostics = {"active": False, "supported": False, "fallback": None}

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
            self.process = owned_popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=None, text=True, encoding="utf-8", bufsize=1,
                creationflags=hidden_console_flags(),
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
            status = self._read_message(time.monotonic() + self.cfg.startup_timeout_s)
        except Exception as exc:
            self.close()
            raise RuntimeError(
                f"speech.cpp returned invalid startup response: {exc}"
            ) from exc

        if status.get("type") != "ready":
            self.close()
            raise RuntimeError(f"Haas worker failed to start: {status}")

        self.supported_options = _advertised_options(status)
        self.worker_version = status.get("version", "unknown")
        model = status.get("model")
        self.model_capabilities = {key: model[key] for key in ("architecture", "layout") if key in model} \
            if isinstance(model, dict) else {}
        self.keyword_support = {"keywords", "keywords_alpha", "keywords_guard"} <= self.supported_options
        if cfg.haas_keywords_enabled and not self.keyword_support:
            log.warning("Haas model/worker has no guarded keyword vocabulary; decoding without hints. "
                        "Older Haas GGUFs may require speech pull after upgrading speech.cpp.")

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

    def _read_message(self, deadline):
        while (remaining := deadline - time.monotonic()) > 0:
            check_running()
            try:
                response = self._read_response(min(remaining, 0.25))
            except TimeoutError:
                continue
            try:
                message = json.loads(response)
            except json.JSONDecodeError:
                log.debug("Ignoring speech.cpp output: %r", response[:200])
                continue
            if isinstance(message, dict):
                return message
        raise TimeoutError("speech.cpp response deadline expired")

    def close(self) -> None:
        with self._close_lock:
            process = self.process
            self.process = None
        if process is None:
            return
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
        self._reader.join(timeout=0.5)

    def transcribe(self, audio: np.ndarray, probe: bool = False, *, timeout=None,
                   keywords=None, keywords_alpha=None) -> str:
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Haas ASR worker is no longer running.")

        deadline = time.monotonic() + (timeout if timeout is not None else self.cfg.response_timeout_s)
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        pcm_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
        pcm = base64.b64encode(pcm_int16.tobytes()).decode("ascii")
        vocabulary = _keyword_text(keywords)
        supported = getattr(self, "keyword_support", False)
        active = bool(vocabulary and self.cfg.haas_keywords_enabled and supported and not probe)
        self.last_keyword_diagnostics = dict(active=active, attempted=active, supported=supported, fallback=None,
                                            guard=True, alpha=None, entries=len(vocabulary.splitlines()),
                                            worker_version=getattr(self, "worker_version", "unknown"),
                                            model=getattr(self, "model_capabilities", {}))
        options = {}
        if active:
            alpha = self.cfg.haas_keywords_alpha if keywords_alpha is None else keywords_alpha
            if type(alpha) not in (int, float) or not np.isfinite(alpha) or not 0 < alpha <= 2:
                raise ValueError("Haas keyword alpha must be finite in (0,2]")
            options = dict(keywords=vocabulary, keywords_alpha=alpha, keywords_guard=True)
            self.last_keyword_diagnostics["alpha"] = alpha
        elif vocabulary:
            self.last_keyword_diagnostics["fallback"] = "unsupported" if not supported else "disabled"
        try:
            text = self._transcribe_request(pcm, deadline, options)
        except _KeywordOptionsRejected as exc:
            # The rejection consumed this request. Re-submit original PCM under
            # a fresh ID, preserving the single overall submission/decode deadline.
            self.keyword_support = False
            self.last_keyword_diagnostics.update(active=False, fallback="unsupported", error=str(exc)[:160])
            log.warning("Haas rejected guarded vocabulary; retrying unboosted within original deadline")
            text = self._transcribe_request(pcm, deadline, {})

        if not text:
            return ""
        if any(phrase in text for phrase in self.cfg.blocklist):
            log.debug("Discarding blocked Haas text: %s", text)
            return ""

        text = trim_repetition_tail(text)
        reason = rejection_reason(text)
        if reason is not None:
            log.debug("Discarding Haas noise (%s): %s", reason, text)
            return ""

        return text

    def _transcribe_request(self, pcm, deadline, options):
        if deadline <= time.monotonic():
            raise TimeoutError("speech.cpp recognition deadline expired")
        self.request_counter += 1
        request_id = str(self.request_counter)

        # A hung speech.cpp worker can block stdin.write as well as
        # stdout.readline. Enforce one deadline across submission and decode.
        # PrimaryASR retires the child after any timeout.
        pipe = self.process.stdin
        submitted = threading.Event()
        write_errors = []
        chunk = json.dumps({"type": "chunk", "id": request_id, "seq": 0, "pcm": pcm}) + "\n"
        decode = json.dumps({"type": "transcribe", "id": request_id, "sample_rate": 16000,
                             **options}) + "\n"

        def submit():
            try:
                pipe.write(chunk)
                pipe.write(decode)
                pipe.flush()
            except (OSError, ValueError) as exc:
                write_errors.append(exc)
            finally:
                submitted.set()

        threading.Thread(target=submit, name="haas-stdin", daemon=True).start()
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not submitted.wait(remaining):
            raise TimeoutError("speech.cpp stdin submission deadline expired")
        if write_errors:
            raise RuntimeError(f"speech.cpp stdin failed: {write_errors[0]}") from write_errors[0]

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("speech.cpp recognition deadline expired")
            result = self._read_message(deadline)
            if result.get("id") != request_id:
                continue
            if result.get("type") == "error":
                error = result.get("error", {})
                if (options and isinstance(error, dict)
                        and error.get("code") in ("unsupported", "invalid_argument", "out_of_range")
                        and error.get("option") in options):
                    raise _KeywordOptionsRejected(str(error))
                raise RuntimeError(f"Haas transcription failed: {result}")
            if result.get("type") == "end":
                self.last_stop = result.get("stop")
                return result.get("text", "").strip()
