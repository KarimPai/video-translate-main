"""CAT-Translate-3.3B Q4_K_M: the sole offline Japanese-to-English translator.

Uses the model author's one-user-message Japanese->English prompt with the GGUF
Sarashina instruct chat template applied by llama.cpp. No Index-specific
constraints, context injection, quality-retry engine or external translation API.
"""
from __future__ import annotations

import atexit
import json
import logging
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from ..config import TranslateConfig
from ..context_memory import reuse_recent_short_term
from ..glossary import Glossary
from ..process_cleanup import _close_job, _create_kill_job
from ..stream_context import (
    repair_ark_tier_translation,
    repair_barotrauma_translation,
    repair_valorant_translation,
)
from ..subject_guard import neutralize_unsupported_subjects
from ..watame import correct_translation
from .quality import (
    SKIP_TOKEN,
    is_obvious_nonverbal_japanese,
    is_skip_token,
    repair_high_confidence_meaning,
)

log = logging.getLogger(__name__)

# https://huggingface.co/cyberagent/CAT-Translate-3.3b
# Apply via llama.cpp --jinja and our pinned official Sarashina chat template.
# Community GGUF metadata is not assumed to contain the correct template.
# The Japanese source is the ONLY dialogue sent to CAT; the model never sees
# earlier Japanese/English lines, preventing context-echo regressions.
_PROMPT = "Translate the following Japanese text into English.\n\n"


def build_cat_prompt(japanese: str) -> str:
    """Official CAT-Translate plain Japanese→English prompt."""
    return _PROMPT + japanese.strip()


def _clean(text: str) -> str:
    """Clean simple response wrappers without changing Japanese meaning."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    # The GGUF/Sarashina tokenizer sometimes returns its literal EOS
    # marker as visible text (e.g. "Well!</s>"). An EOS token ends the
    # response; never display it or text beyond it in an English subtitle.
    text = text.split("</s>", 1)[0].strip()
    text = re.sub(r"^(?:Translation|English|EN)\s*:\s*", "", text, flags=re.I)
    text = text.strip(' \t\r\n"“”')
    return " ".join(text.splitlines()).strip()


# These are translation-assistant *meta responses*, not a translation of
# short conversational speech. Keep patterns narrow to avoid dropping a
# genuine stream discussion about translation or help.
_META_REPLY = re.compile(
    r"(?:\bi(?:['’]m| am) (?:ready|here|happy) to (?:help )?translate\b|"
    r"\bi (?:do not|don['’]t) (?:see|have) (?:the |any )?(?:[\w-]+ )?japanese (?:text|phrase)\b|"
    r"\b(?:please|could you) (?:provide|paste|send) (?:me )?(?:the |some )?(?:japanese )?text\b|"
    r"\b(?:the |a )?\d+-character japanese (?:text|phrase)\b)",
    flags=re.IGNORECASE,
)
_PUNCTUATION_SPIRAL = re.compile(r"(?:[.!?…]\s*){12,}")


def cat_output_problem(japanese: str, english: str) -> str | None:
    """Detect only clearly invalid completions, not subjective translations."""
    if not english or is_skip_token(english):
        return "empty"
    if _META_REPLY.search(english):
        return "assistant_meta_reply"
    if _PUNCTUATION_SPIRAL.search(english):
        return "punctuation_loop"
    if len(english) > max(300, len(japanese) * 8):
        return "runaway_length"
    return None


# Very short, unambiguous conversational interjections. Exact full-line
# matches only; never remove context or combine distinct speakers.
_COMMON_JP = {
    "なるほど": "I see.",
    "あかんか": "So that won't work?",
}


def _model_ids(base_url: str) -> list[str] | None:
    """Return loaded IDs or None while the configured endpoint is not ready."""
    try:
        with urllib.request.urlopen(base_url + "/models", timeout=1.0) as response:
            data = json.load(response)
        return [str(item.get("id", "")) for item in data.get("data", [])]
    except urllib.error.HTTPError as exc:
        if exc.code == 503:
            return None
        raise
    except (OSError, urllib.error.URLError):
        return None


class CatTranslator:
    """One locally owned CAT-Translate-3.3B llama.cpp inference server."""

    def __init__(self, cfg: TranslateConfig, glossary: Glossary, *, vod_title: str = "") -> None:
        self.cfg = cfg
        self.glossary = glossary
        # Shared, text-only short-term repeat consistency; NOT LLM prompt history.
        self.memory_lines = max(0, cfg.memory_lines)
        self.vod_title = " ".join(vod_title.split())[:200]
        self.base_url = cfg.cat_base_url.rstrip("/")
        self.url = self.base_url + "/chat/completions"
        self.process: subprocess.Popen | None = None
        self._job = None
        self._server_log = None
        self._registered_exit = False
        self._start_server()
        try:
            self._warmup()
        except Exception:
            self.close()
            raise

    def _start_server(self) -> None:
        existing = _model_ids(self.base_url)
        if existing is not None:
            raise RuntimeError(
                "Another llama.cpp server already occupies " + self.base_url
                + " (" + repr(existing) + "). Close it or change cat_base_url. "
                "CAT never reuses or terminates a server it did not start."
            )
        binary = shutil.which("llama-server")
        if not binary:
            raise RuntimeError(
                "CAT-Translate requires a recent CUDA-capable llama-server in PATH "
                "(including --hf-repo, --hf-file, and --jinja support)."
            )
        parsed = urllib.parse.urlparse(self.base_url)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost"):
            raise ValueError("cat_base_url must be a local http://127.0.0.1 URL")
        host, port = parsed.hostname, parsed.port or 8003
        command = [
            binary,
            "-hf", self.cfg.cat_hf_repo,
            "-hff", self.cfg.cat_hf_file,
            "-c", str(self.cfg.cat_context),
            "--host", host, "--port", str(port),
            "--alias", self.cfg.cat_model,
            "--jinja",
            "--chat-template-file", str(Path(__file__).with_name("cat_sarashina.jinja")),
            "--no-mmproj",
            "-ngl", str(self.cfg.cat_gpu_layers),
            "-np", "1",
        ]
        logs = Path("logs")
        logs.mkdir(parents=True, exist_ok=True)
        self._server_log = (logs / "cat-server.log").open("a", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=self._server_log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self._job = _create_kill_job(self.process)
            atexit.register(self.close)
            self._registered_exit = True
        except Exception:
            self.close()
            raise

        log.info(
            "Starting CAT-Translate-3.3B Q4_K_M (%s), server PID %s",
            self.cfg.cat_hf_file, self.process.pid,
        )
        deadline = time.monotonic() + self.cfg.load_timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.close()
                raise RuntimeError(
                    "CAT llama-server exited during startup. Check logs/cat-server.log "
                    "for GGUF download, unsupported chat template, or CUDA errors."
                )
            loaded = _model_ids(self.base_url)
            if loaded is not None:
                if self.cfg.cat_model not in loaded:
                    self.close()
                    raise RuntimeError("Unexpected model on CAT server: " + repr(loaded))
                log.info("CAT-Translate-3.3B ready at %s", self.base_url)
                return
            time.sleep(0.5)
        self.close()
        raise TimeoutError(
            f"CAT model load exceeded {self.cfg.load_timeout_s:.0f}s; "
            "see logs/cat-server.log"
        )

    def _warmup(self) -> None:
        deadline = time.monotonic() + self.cfg.load_timeout_s
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError("CAT llama-server exited during warm-up.")
            try:
                translation = self._chat("こんにちは", timeout=min(
                    self.cfg.timeout_s, max(1.0, deadline - time.monotonic())
                ))
                if _clean(translation):
                    return
            except urllib.error.HTTPError as exc:
                if exc.code != 503:
                    raise
            except (TimeoutError, urllib.error.URLError):
                if self.process is not None and self.process.poll() is not None:
                    raise RuntimeError("CAT server stopped during warm-up.")
            time.sleep(0.5)
        raise TimeoutError("CAT translator warm-up timed out; see logs/cat-server.log")

    def _chat(self, japanese: str, *, timeout: float, retry: bool = False) -> str:
        # CAT's model card specifies exactly one user role and the short prompt.
        # No Index instruction/metadata stop strings, thinking flags, or
        # model-specific quality-retry chat parameters are sent here. The
        # lone stop string is the Sarashina EOS seen in real CAT outputs.
        payload = {
            "model": self.cfg.cat_model,
            "messages": [{"role": "user", "content": build_cat_prompt(japanese)}],
            "temperature": 0.35 if retry else 0,
            "repeat_penalty": 1.08 if retry else 1.05,
            "max_tokens": min(self.cfg.max_tokens, 96) if retry else self.cfg.max_tokens,
            # Sarashina's model EOS is </s>. Explicitly stop here in case
            # this community GGUF exposes EOS as an ordinary text token.
            "stop": ["</s>"],
            "stream": False,
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
        return str(data["choices"][0]["message"].get("content") or "")

    def translate(self, ja: str, history: list[tuple[str, str]]) -> str:
        if not ja.strip():
            return SKIP_TOKEN
        if is_obvious_nonverbal_japanese(ja) and not self.glossary.find(ja):
            return SKIP_TOKEN
        if self.memory_lines:
            reused = reuse_recent_short_term(ja, history)
            if reused is not None:
                return reused
        bare = ja.strip().strip(" \t\r\n。、!?！？")
        if bare in _COMMON_JP:
            return _COMMON_JP[bare]

        english = _clean(self._chat(ja, timeout=self.cfg.timeout_s))
        problem = cat_output_problem(ja, english)
        if problem:
            log.warning("CAT output rejected (%s), source=%r", problem, ja[:100])
            # One bounded re-attempt using the same model and source-only
            # prompt. Slight sampling/rep penalty can escape an EOS loop;
            # never show assistant instructions or repetitive punctuation.
            try:
                alternative = _clean(self._chat(
                    ja, timeout=min(4.0, self.cfg.timeout_s), retry=True
                ))
                if cat_output_problem(ja, alternative) is None:
                    english = alternative
                    problem = None
            except (OSError, ValueError, KeyError, IndexError) as exc:
                log.warning("CAT invalid-output retry failed: %s", exc)
        if problem:
            log.warning("CAT skipped invalid subtitle (%s)", problem)
            return SKIP_TOKEN

        # Shared conservative postprocessing only. These do not ask any
        # additional language model to finish interrupted speech.
        english = repair_high_confidence_meaning(ja, english)
        recent_ja = [j for j, _ in history[-8:]]
        english = repair_valorant_translation(ja, english, recent_ja, False)
        english = repair_barotrauma_translation(ja, english)
        english = repair_ark_tier_translation(ja, english, self.vod_title, recent_ja)
        english = correct_translation(ja, english)
        english = neutralize_unsupported_subjects(ja, english)
        return english or "[translation failed]"

    def close(self) -> None:
        """Terminate only this session's model server, then close job/log."""
        process, self.process = self.process, None
        job, self._job = self._job, None
        try:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        finally:
            if job is not None:
                _close_job(job)
            if self._server_log is not None:
                self._server_log.close()
                self._server_log = None
            if self._registered_exit:
                atexit.unregister(self.close)
                self._registered_exit = False
