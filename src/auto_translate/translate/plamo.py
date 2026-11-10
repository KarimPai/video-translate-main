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
from ..context_memory import build_recent_context, reuse_recent_short_term
from ..translation_integrity import (
    enforce_known_name_spellings,
    enforce_numeric_fidelity,
    japanese_number_facts,
    contextual_translation_hints,
)
from ..glossary import Glossary
from ..subject_guard import neutralize_unsupported_subjects
from ..watame import correct_translation

log = logging.getLogger(__name__)


# PLaMo 3 GGUF ignores system messages; reference guidance belongs in
# chat_template_kwargs["context"], alongside the video title.
_NEUTRAL_SUBJECT_CONTEXT = (
    "Informal Japanese livestream and gaming conversation. "
    "Japanese often omits the subject or possessor. "
    "If the current Japanese does not explicitly identify who is acting "
    "or who owns something, prefer natural, subject-neutral English "
    "instead of guessing I, you, my, your, we, our, he, she, his, or her. "
    "For example, 運転下手だね means 'The driving is bad', not "
    "'My driving is bad' or 'Your driving is bad'. "
    "Preserve people and possessives explicitly stated in the Japanese. "
    "A video title or previous English translation is not proof of "
    "an omitted subject's identity. "
    "If a driver is not explicitly named, say The driving was bad, "
    "not My driving was bad or Your driving was bad. "
    "If the source only describes a crash, say There was a collision, "
    "not I crashed or You crashed. "
    "Translate ONLY the current Japanese line. Previous subtitles are context, "
    "not additional words to include. They may be spoken by different people. "
    "Keep physical movement, directions and place references literal when supported "
    "by the Japanese. Do not invent emotions, names, destinations or actions. "
    "Use exact glossary spellings for names and preserve every Japanese number."
)


def _translation_context(vod_title: str) -> str:
    if vod_title:
        return f"Video title: {vod_title}\n{_NEUTRAL_SUBJECT_CONTEXT}"
    return _NEUTRAL_SUBJECT_CONTEXT


class PlamoTranslator:
    """Japanese-to-English PLaMo translator; inference stays on the local machine."""

    def __init__(self, cfg: TranslateConfig, glossary: Glossary, *, vod_title: str = "") -> None:
        self.cfg = cfg
        self.memory_lines = max(0, getattr(cfg, "memory_lines", 24))
        self.glossary = glossary
        self.vod_title = " ".join(vod_title.split())[:200]
        self.url = cfg.plamo_base_url.rstrip("/") + "/chat/completions"
        self.base_url = cfg.plamo_base_url.rstrip("/")
        self.process: subprocess.Popen | None = None
        self._server_log = None
        self._start_server()
        try:
            # Warm up before ingest starts so first subtitles do not pay load cost.
            self._warmup()
        except Exception:
            self.close()
            raise

    def _start_server(self) -> None:
        """Reuse a responding local server; otherwise launch and own one."""
        if _server_responds(self.base_url):
            log.warning(
                "Reusing existing llama.cpp server at %s. "
                "Requested -ngl %s / -c %s cannot be applied to an already "
                "running server; close the old server if it uses more VRAM.",
                self.base_url, self.cfg.plamo_gpu_layers,
                self.cfg.plamo_context,
            )
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

    def _warmup(self) -> None:
        """Retry transient 503 responses while llama.cpp finishes model setup."""
        deadline = time.monotonic() + self.cfg.load_timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"PLaMo did not accept a warm-up request within "
                    f"{self.cfg.load_timeout_s:.0f}s. See logs\\plamo-server.log."
                )
            try:
                self._chat("こんにちは", [], timeout=min(10.0, remaining))
                return
            except urllib.error.HTTPError as exc:
                if exc.code != 503:
                    raise
                log.info("PLaMo model is still loading; retrying warm-up...")
            time.sleep(min(1.0, max(0.0, remaining)))

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

    def translate(
        self, ja: str, history: list[tuple[str, str]],
        *, fragment_context: tuple[str, ...] = (),
    ) -> str:
        if getattr(self, "memory_lines", 0):
            reused = reuse_recent_short_term(ja, history)
            if reused is not None:
                log.debug("Reusing recent short term: JP=%r EN=%r", ja, reused)
                return reused
        english = _clean(self._chat(
            ja, history, timeout=self.cfg.timeout_s,
            fragment_context=fragment_context,
        ))
        if getattr(self.cfg, "protect_numbers", True):
            english = enforce_numeric_fidelity(ja, english)
        if getattr(self.cfg, "protect_names", True):
            english = enforce_known_name_spellings(ja, english, self.glossary)
        english = correct_translation(ja, english)
        corrected = neutralize_unsupported_subjects(ja, english)
        if corrected != english:
            log.debug("Neutralized unsupported subject: JP=%r EN=%r -> %r", ja, english, corrected)
        return corrected

    def _chat(
        self,
        ja: str,
        history: list[tuple[str, str]],
        timeout: float,
        fragment_context: tuple[str, ...] = (),
    ) -> str:
        messages: list[dict[str, str]] = []
        dialogue_history = history[-self.cfg.context_lines:] if self.cfg.context_lines > 0 else []
        for source, translation in dialogue_history:
            messages.append({"role": "user", "content": source})
            # Prior inferred pronouns must not contaminate new translations.
            messages.append({
                "role": "assistant",
                "content": neutralize_unsupported_subjects(source, translation),
            })
        messages.append({"role": "user", "content": ja})

        memory = build_recent_context(
            history, self.glossary,
            context_lines=self.cfg.context_lines,
            memory_lines=getattr(self.cfg, "memory_lines", 24),
        )
        options: dict = {
            "source_lang": "ja",
            "target_lang": "en",
            "context": _translation_context(self.vod_title) + (
                "\n" + memory if memory else ""
            ),
        }
        if fragment_context:
            options["context"] += (
                "\nImmediately preceding Japanese speech fragments "
                "(context only, possibly different speakers; do NOT translate "
                "these fragments again): " + " / ".join(fragment_context)
            )
        # Possible ASR homophones are translation hints only; never
        # mutate the source JP or blindly normalize taboo/ambiguous words.
        if hints := contextual_translation_hints(ja, recent_fragments=fragment_context):
            options["context"] += (
                "\nPossible ASR homophones in CURRENT Japanese. Use only if "
                "supported by adjacent conversation; never invent unheard "
                "words or rewrite the source: " + " ".join(hints)
            )
        if number_facts := japanese_number_facts(ja):
            pairs = "; ".join(
                f"{original} = {number}"
                for original, number in number_facts[:4]
            )
            options["context"] += (
                "\nExact numbers in the CURRENT Japanese line: " + pairs
            )
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
        # llama.cpp can return 503 while the model is still loading.
        return False
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
