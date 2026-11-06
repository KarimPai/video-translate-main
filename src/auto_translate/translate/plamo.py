"""PLaMo 3 Translate with an automatically managed local llama.cpp server."""

import atexit
import json
import logging
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from ..config import TranslateConfig
from ..glossary import Glossary

log = logging.getLogger(__name__)


class PlamoTranslator:
    """Japanese-to-English PLaMo translator; inference stays on the local machine."""

    def __init__(self, cfg: TranslateConfig, glossary: Glossary) -> None:
        self.cfg = cfg
        self.glossary = glossary
        self.url = cfg.plamo_base_url.rstrip("/") + "/chat/completions"
        self.base_url = cfg.plamo_base_url.rstrip("/")
        self.process: subprocess.Popen | None = None
        self._server_log = None
        self._start_server()
        try:
            # Warm up before ingest starts so first subtitles do not pay load cost.
            self._chat("こんにちは", [], timeout=cfg.load_timeout_s)
        except Exception:
            self.close()
            raise

    def _start_server(self) -> None:
        """Reuse a responding local server; otherwise launch and own one."""
        if _server_responds(self.base_url):
            log.info("Using existing llama.cpp server at %s", self.base_url)
            return

        executable = shutil.which("llama-server")
        if not executable:
            raise RuntimeError(
                "PLaMo requires llama-server, but it was not found in PATH. "
                "Install a CUDA/Vulkan-enabled llama.cpp build and reopen CMD."
            )

        parsed = urllib.parse.urlparse(self.base_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        command = [
            executable,
            "-hf", self.cfg.plamo_hf_model,
            "-c", str(self.cfg.plamo_context),
            "--host", host,
            "--port", str(port),
            "--alias", self.cfg.plamo_model,
            "--jinja",
            "--no-prefill-assistant",
            "-ngl", str(self.cfg.plamo_gpu_layers),
        ]

        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        self._server_log = (log_dir / "plamo-server.log").open(
            "a", encoding="utf-8"
        )
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=self._server_log,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
        except Exception:
            self._server_log.close()
            self._server_log = None
            raise

        atexit.register(self.close)
        log.info(
            "Started local PLaMo server (PID %s); loading model...",
            self.process.pid,
        )

        deadline = time.monotonic() + self.cfg.load_timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.close()
                raise RuntimeError(
                    "llama-server exited during startup. See logs\\plamo-server.log."
                )
            if _server_responds(self.base_url):
                log.info("PLaMo server is ready at %s", self.base_url)
                return
            time.sleep(0.5)

        self.close()
        raise TimeoutError(
            f"llama-server did not become ready within "
            f"{self.cfg.load_timeout_s:.0f}s. See logs\\plamo-server.log."
        )

    def close(self) -> None:
        """Stop only the llama-server process started by this translator."""
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            log.info("Stopping PLaMo server (PID %s)...", process.pid)
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if self._server_log is not None:
            self._server_log.close()
            self._server_log = None
        atexit.unregister(self.close)

    def translate(self, ja: str, history: list[tuple[str, str]]) -> str:
        return _clean(self._chat(ja, history, timeout=self.cfg.timeout_s))

    def _chat(
        self,
        ja: str,
        history: list[tuple[str, str]],
        timeout: float,
    ) -> str:
        messages: list[dict[str, str]] = []
        for source, translation in history[-self.cfg.context_lines:]:
            messages.append({"role": "user", "content": source})
            messages.append({"role": "assistant", "content": translation})
        messages.append({"role": "user", "content": ja})

        options: dict = {
            "source_lang": "ja",
            "target_lang": "en",
        }
        if hits := self.glossary.find(ja):
            options["glossary"] = [
                {"source": source, "target": target}
                for source, target in hits
            ]

        body = {
            "model": self.cfg.plamo_model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.cfg.max_tokens,
            "stop": ["<|plamo:tag|>", "<|plamo:bos|>"],
            "chat_template_kwargs": options,
            "stream": False,
        }
        req = urllib.request.Request(
            self.url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
        return data["choices"][0]["message"]["content"]


def _server_responds(base_url: str) -> bool:
    """Check the OpenAI-compatible models endpoint without sending inference."""
    try:
        with urllib.request.urlopen(base_url + "/models", timeout=1.0):
            return True
    except urllib.error.HTTPError:
        return True
    except (OSError, urllib.error.URLError):
        return False


def _clean(text: str) -> str:
    """Remove empty lines, surrounding quotes, and accidental role prefixes."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    line = lines[0]
    if line.upper().startswith(("EN:", "ENGLISH:")):
        line = line.split(":", 1)[1].strip()
    return line.strip('"“”「」')
