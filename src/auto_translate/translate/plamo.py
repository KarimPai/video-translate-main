"""PLaMo 3 Translate 8B Q3_K_M: sole offline Japanese-to-English model.

Uses the official PFML3 completion format instead of an instruction-chat
template. The llama.cpp server is started/stopped only by this process.
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

# Official examples:
# https://huggingface.co/pfnet/plamo-3-translate-8b-GGUF
# PFML3 includes native context/history and uses /v1/completions (no chat wrapper).
_PFML3_START = "<|plamo:tag|>dataset<|plamo:msg|>translation"
_PFML3_INPUT = "<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>"
_PFML3_OUTPUT_EN = "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>"
_STOP = ["<|plamo:tag|>", "<|plamo:bos|>"]
_BAD_META = re.compile(
    r"\bi(?:['’]m| am) (?:ready|happy) to (?:help )?translate\b|"
    r"\bi (?:don't|do not) see (?:the |any )?japanese (?:text|phrase)\b",
    re.I,
)
_PUNCT_LOOP = re.compile(r"(?:[.!?…]\s*){12,}")


_PFML3_CONTEXT = "<|plamo:tag|>context<|plamo:msg|>"


def _safe_pfml3_value(value: str) -> str:
    """Prevent untrusted caption/title text from introducing structural tokens."""
    return " ".join(str(value).replace("<|plamo:", "＜|plamo:").split())


def build_plamo_prompt(
    japanese: str,
    matched_terms: list[tuple[str, str]] | None = None,
    *,
    history: list[tuple[str, str]] | None = None,
    context: str = "",
) -> str:
    """Official PFML3 order: context, past input/output pairs, glossary, input.

    Only the last output tag is unfinished, so PLaMo translates exactly the
    current caption instead of regenerating any historical subtitles.
    https://huggingface.co/pfnet/plamo-3-translate-8b-GGUF
    """
    prompt = _PFML3_START
    safe_context = _safe_pfml3_value(context)[:180]
    if safe_context:
        prompt += _PFML3_CONTEXT + safe_context

    for previous_ja, previous_en in history or ():
        src = _safe_pfml3_value(previous_ja)
        dst = _safe_pfml3_value(previous_en)
        if (
            not src or not dst or len(src) > 240 or len(dst) > 320
            or is_skip_token(dst) or dst == "[translation failed]"
        ):
            continue
        prompt += _PFML3_INPUT + src + _PFML3_OUTPUT_EN + dst

    glossary_started = False
    for ja_term, english_term in (matched_terms or [])[:8]:
        if not ja_term or not english_term:
            continue
        if any(marker in ja_term or marker in english_term
               for marker in ("<|plamo:", "\n", "\r")):
            continue
        if not glossary_started:
            prompt += "<|plamo:tag|>glossary"
            glossary_started = True
        prompt += "<|plamo:index|>" + ja_term + "<|plamo:item|>" + english_term
    return prompt + _PFML3_INPUT + _safe_pfml3_value(japanese) + _PFML3_OUTPUT_EN


def _estimate_prompt_tokens(prompt: str) -> int:
    """Conservative no-extra-request estimate for 2048-token llama.cpp slot.

    Estimates are intentionally overbudgeted for Japanese and non-ASCII
    characters. The tokenizer itself is authoritative; this avoids an extra
    HTTP tokenization round-trip before every subtitle.
    """
    text_only = re.sub(r"<\\|plamo:[^|>]+\\|>", "", prompt)
    return int(
        sum(2 if ord(ch) > 127 else 0.5 for ch in text_only)
        + 4 * prompt.count("<|plamo:") + 24
    )


def _clean(text: str) -> str:
    """Strip completion wrappers while keeping the translated sentence intact."""
    text = re.split(r"<\|plamo:(?:tag|bos)\|>", text, maxsplit=1)[0]
    text = re.sub(r"^(?:English|Translation|EN)\s*:\s*", "", text.strip(), flags=re.I)
    return " ".join(text.strip(' \t\r\n"“”').splitlines()).strip()


def _invalid_output(japanese: str, english: str) -> bool:
    """Suppress obvious model failures, not ambiguous conversational meaning."""
    return (
        not english
        or is_skip_token(english)
        or bool(_BAD_META.search(english))
        or bool(_PUNCT_LOOP.search(english))
        or len(english) > max(300, len(japanese) * 8)
    )


def _model_ids(base_url: str) -> list[str] | None:
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


class PlamoTranslator:
    """One owned local PLaMo 3 Translate 8B Q3_K_M llama.cpp process."""

    def __init__(self, cfg: TranslateConfig, glossary: Glossary, *, vod_title: str = "") -> None:
        self.cfg = cfg
        self.glossary = glossary
        self.memory_lines = max(0, cfg.memory_lines)
        self._last_recheck_at = float("-inf")
        # Temperature is zero but previous dialogue changes the correct output.
        # Cache by the full effective source/history/title combination.
        self._raw_cache: dict[tuple, str] = {}
        self._raw_cache_limit = 512
        self.vod_title = " ".join(vod_title.split())[:200]
        self.base_url = cfg.plamo_base_url.rstrip("/")
        self.url = self.base_url + "/completions"
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
                "Another server occupies " + self.base_url + " (" + repr(existing)
                + "). Close it or change plamo_base_url. The translator "
                "never reuses or terminates a server it did not start."
            )
        binary = shutil.which("llama-server")
        if not binary:
            raise RuntimeError(
                "PLaMo 8B requires a recent CUDA-enabled llama-server in PATH "
                "with -hf, -hff, -fa and quantized KV-cache flags."
            )
        parsed = urllib.parse.urlparse(self.base_url)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost"):
            raise ValueError("plamo_base_url must be a local http://127.0.0.1 URL")
        host, port = parsed.hostname, parsed.port or 8005
        command = [
            binary,
            "-hf", self.cfg.plamo_hf_repo,
            "-hff", self.cfg.plamo_hf_file,
            "-c", str(self.cfg.plamo_context),
            "--host", host, "--port", str(port),
            "--alias", self.cfg.plamo_model,
            "-ngl", str(self.cfg.plamo_gpu_layers),
            "-np", "1",
            "-fa", "on",
            "-ctk", "q8_0",
            "-ctv", "q8_0",
            "--jinja",
            "--no-prefill-assistant",
        ]
        logs = Path("logs")
        logs.mkdir(parents=True, exist_ok=True)
        self._server_log = (logs / "plamo8b-server.log").open("a", encoding="utf-8")
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
            "Starting PLaMo 3 Translate 8B Q3_K_M (%s), PID %s",
            self.cfg.plamo_hf_file, self.process.pid,
        )
        deadline = time.monotonic() + self.cfg.load_timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.close()
                raise RuntimeError(
                    "PLaMo llama-server exited during startup. Check logs/plamo8b-server.log "
                    "for model downloads, unsupported CUDA, Flash Attention or KV cache."
                )
            loaded = _model_ids(self.base_url)
            if loaded is not None:
                if self.cfg.plamo_model not in loaded:
                    self.close()
                    raise RuntimeError("Unexpected model at PLaMo server: " + repr(loaded))
                log.info("PLaMo 3 Translate 8B ready at %s", self.base_url)
                return
            time.sleep(0.5)
        self.close()
        raise TimeoutError(
            "PLaMo 8B model load timed out; see logs/plamo8b-server.log"
        )

    def _warmup(self) -> None:
        deadline = time.monotonic() + self.cfg.load_timeout_s
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError("PLaMo server exited during warm-up.")
            try:
                result = self._complete("こんにちは", timeout=min(
                    self.cfg.timeout_s, max(1.0, deadline - time.monotonic())
                ))
                if _clean(result):
                    return
            except urllib.error.HTTPError as exc:
                if exc.code != 503:
                    raise
            except (TimeoutError, urllib.error.URLError):
                if self.process is not None and self.process.poll() is not None:
                    raise RuntimeError("PLaMo server stopped during warm-up.")
            time.sleep(0.5)
        raise TimeoutError("PLaMo translator warm-up timed out; see logs/plamo8b-server.log")

    def _complete(
        self,
        japanese: str,
        *,
        timeout: float,
        history: list[tuple[str, str]] | None = None,
    ) -> str:
        # Build only official PFML3 elements. Drop the oldest history pairs
        # when the approximate input budget would exceed the configured slot.
        history_pairs = list(history or ())[-max(0, min(int(self.cfg.context_lines), 8)):]
        if not self.cfg.context_lines:
            history_pairs = []
        title_context = self.vod_title if self.cfg.context_title_enabled else ""
        glossary_hits = self.glossary.find(japanese)
        prompt_budget = max(256, self.cfg.plamo_context - self.cfg.max_tokens - 96)
        prompt = ""
        for keep in range(len(history_pairs), -1, -1):
            chosen = history_pairs[-keep:] if keep else []
            candidate = build_plamo_prompt(
                japanese, glossary_hits,
                history=chosen, context=title_context,
            )
            if _estimate_prompt_tokens(candidate) <= prompt_budget:
                prompt = candidate
                if keep != len(history_pairs):
                    log.debug(
                        "PLaMo prompt budget: retained %d/%d recent pairs",
                        keep, len(history_pairs),
                    )
                break
        if not prompt:
            # Never truncate the current Japanese caption. If even the
            # source-only prompt cannot fit, the server reports an error.
            prompt = build_plamo_prompt(japanese, glossary_hits)
        payload = {
            "model": self.cfg.plamo_model,
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": self.cfg.max_tokens,
            "stop": _STOP,
            "stream": False,
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
        return str(data["choices"][0].get("text") or "")

    def translate(self, ja: str, history: list[tuple[str, str]]) -> str:
        if not ja.strip():
            return SKIP_TOKEN
        if is_obvious_nonverbal_japanese(ja) and not self.glossary.find(ja):
            return SKIP_TOKEN
        if self.memory_lines:
            reused = reuse_recent_short_term(ja, history)
            if reused is not None:
                return reused
        previous = (
            list(history[-max(0, min(int(self.cfg.context_lines), 8)):])
            if self.cfg.context_lines else []
        )
        # A raw completion for an identical caption can differ between
        # conversations. The original JA-only cache was context-unsafe.
        cache_key = (
            ja, self.vod_title if self.cfg.context_title_enabled else "",
            tuple(previous),
        )
        english = self._raw_cache.get(cache_key)
        if english is None:
            english = _clean(self._complete(
                ja, timeout=self.cfg.timeout_s, history=previous,
            ))
        # PFML3 does not support a free-form system instruction. Keep its
        # official completion format and retry only an invalid completion.
        if (_invalid_output(ja, english)
                and self.cfg.quality_recheck_enabled
                and time.monotonic() - self._last_recheck_at >= self.cfg.quality_recheck_min_interval_s):
            self._last_recheck_at = time.monotonic()
            try:
                retry = _clean(self._complete(
                    ja,
                    timeout=min(self.cfg.timeout_s, self.cfg.quality_recheck_timeout_s),
                    history=previous,
                ))
                if not _invalid_output(ja, retry):
                    english = retry
            except Exception as exc:
                log.debug("PLaMo 8B invalid-completion retry failed: %s", exc)
        if _invalid_output(ja, english):
            log.warning("PLaMo invalid completion skipped for source=%r", ja[:100])
            return SKIP_TOKEN
        if cache_key not in self._raw_cache:
            if len(self._raw_cache) >= self._raw_cache_limit:
                self._raw_cache.pop(next(iter(self._raw_cache)))
            self._raw_cache[cache_key] = english
        english = repair_high_confidence_meaning(ja, english)
        recent_ja = [j for j, _ in history[-8:]]
        english = repair_valorant_translation(ja, english, recent_ja, False)
        english = repair_barotrauma_translation(ja, english)
        english = repair_ark_tier_translation(ja, english, self.vod_title, recent_ja)
        english = correct_translation(ja, english)
        english = neutralize_unsupported_subjects(ja, english)
        return english or SKIP_TOKEN

    def close(self) -> None:
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
