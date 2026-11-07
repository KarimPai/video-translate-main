"""PLaMo 3 Translate with an automatically managed local llama.cpp server."""

import atexit
import json
import logging
import shutil
import subprocess
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from ..config import TranslateConfig
from ..context_memory import build_recent_context, reuse_recent_short_term
from ..glossary import Glossary
from ..subject_guard import neutralize_unsupported_subjects
from ..watame import correct_translation
from .quality import (
    SKIP_TOKEN, is_obvious_nonverbal_japanese, is_skip_token,
    translation_review_reason,
)

log = logging.getLogger(__name__)


# PLaMo 3 GGUF ignores system messages; reference guidance belongs in
# chat_template_kwargs["context"], alongside the video title.
_NEUTRAL_SUBJECT_CONTEXT = (
    "Informal Japanese livestream and gaming conversation. "
    "Japanese often omits the subject or possessor. "
    "If the current Japanese does not explicitly identify who is acting "
    "or who owns something, prefer natural, subject-neutral English "
    "instead of guessing I, you, my, your, we, our, he, she, his, or her. "
    "Preserve people and possessives explicitly stated in Japanese, plus negation and intensity. "
    "Use the current Japanese to decide whether the speaker expresses praise, "
    "criticism, or surprise. Prior dialogue provides context. "
    "In longer phrases, すごく or すごい can intensify the following description. "
    "Translate meaningful Japanese to natural English, not phonetic romanization. "
    "Omit meaningless rhythmic syllables or nonverbal sounds mixed into speech. "
    "If and only if the ENTIRE utterance has no meaningful speech, output exactly [SKIP]. "
    "Keep names, game titles and glossary terminology even if uncommon. "
    "Never guess content that the Japanese does not convey."
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

    def translate(self, ja: str, history: list[tuple[str, str]]) -> str:
        # An isolated exclamation can express awe or shock in either scene.
        reaction = unicodedata.normalize("NFKC", ja).strip().rstrip("!.。…").rstrip()
        if reaction in {"すご", "すごっ", "すごい", "スゴ", "スゴッ", "スゴイ", "凄", "凄っ", "凄い"}:
            return "Wow!"
        if getattr(self, "memory_lines", 0):
            reused = reuse_recent_short_term(ja, history)
            if reused is not None:
                log.debug("Reusing recent short term: JP=%r EN=%r", ja, reused)
                return reused
        # A clear rhythmic-syllable loop can be skipped without paying for
        # translation or a review. A glossary hit always protects names/terms.
        glossary = getattr(self, "glossary", None)
        hits = glossary.find(ja) if glossary is not None else []
        if not hits and is_obvious_nonverbal_japanese(ja):
            log.debug("Suppressing nonverbal syllable loop: %r", ja)
            return SKIP_TOKEN

        english = _clean(self._chat(ja, history, timeout=self.cfg.timeout_s))
        reason = translation_review_reason(ja, english)

        # Only questionable output gets a second request. It goes through
        # the same local PLaMo, with identical video context and glossary.
        if reason and getattr(self.cfg, "quality_recheck_enabled", True):
            now = time.monotonic()
            gap = max(0.0, getattr(self.cfg, "quality_recheck_min_interval_s", 10.0))
            if now - getattr(self, "_last_recheck_at", float("-inf")) >= gap:
                self._last_recheck_at = now
                timeout = min(
                    self.cfg.timeout_s,
                    max(0.5, getattr(self.cfg, "quality_recheck_timeout_s", 3.0)),
                )
                try:
                    revised = _clean(self._chat(
                        ja, history, timeout=timeout, review_hint=reason,
                    ))
                except Exception as exc:
                    log.debug("PLaMo selective recheck failed (%s): %s", reason, exc)
                else:
                    if revised and not is_skip_token(revised):
                        # Do not trade a dubious output for another dubious
                        # output or lose a proper name the first pass retained.
                        preserved_names = all(
                            target.casefold() in revised.casefold()
                            for _, target in hits
                            if target.casefold() in english.casefold()
                        )
                        if preserved_names and translation_review_reason(ja, revised) is None:
                            log.debug("PLaMo selective recheck accepted (%s)", reason)
                            english = revised
                        else:
                            log.debug("PLaMo selective recheck rejected (%s)", reason)

        # Treat model-produced [SKIP] as a request for review, NEVER as
        # permission to silently discard ordinary Japanese speech.
        if is_skip_token(english):
            log.warning("PLaMo returned [SKIP] for meaningful Japanese: %r", ja)
            return "[translation failed]"

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
        review_hint: str = "",
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
        if review_hint:
            options["context"] += (
                "\nCarefully retry this Japanese-to-English translation. "
                f"Possible first-pass issue: {review_hint}. "
                "Follow the Japanese, video context and glossary; preserve negation, "
                "names and meaning. Return only the final English translation "
                "or [SKIP] if there are no meaningful spoken words."
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
