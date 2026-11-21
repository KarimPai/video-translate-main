"""Tencent Hy-MT2-7B served by an owned llama.cpp server.

Prompts follow the official Hy-MT2 templates: glossary hits use the
terminology form, recent dialogue uses the background-information form, an
optional style uses the style form, and they are combined when present.
Hy-MT2 is used without a system prompt.

Context comes first and grows between cues so llama.cpp can reuse the cached
prompt prefix; per-cue terms and the source line come last.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from difflib import SequenceMatcher

import httpx

from ..config import TranslateConfig
from ..glossary import Glossary
from ..llama_server import (
    LlamaServer,
    LlamaServerError,
    detect_device,
    gguf_block_count,
    require_gguf,
    resolve_executable,
)
from ..runtime_diagnostics import increment_counter, record_event, update_metrics
from ..stream_context import StreamContext
from ..subtitle_clean import english_output_issue, unmatched_double_quote_cleanup
from ..windows_subprocess import check_running
from .base import TranslatorUnavailable

log = logging.getLogger(__name__)

MAX_TERMS = 16
RELIEF_LAYERS = 4
_PLAIN = "将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
_TERMS = "参考下面的翻译：\n{terms}\n"
_BACKGROUND = "【背景信息】\n{context}\n\n"
_CONTEXT = "请结合背景信息将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释。\n\n【待翻译文本】\n{text}"
_STYLE_PLAIN = "请将以下文本翻译为 英语。\n注意翻译的风格要严格符合【{style}】\n\n{text}"
_STYLE_CONTEXT = "请结合背景信息将以下文本翻译为 英语。\n注意翻译的风格要严格符合【{style}】\n\n【待翻译文本】\n{text}"
_DIALOGUE = "直前の会話："
_ROLE = re.compile(r"^(?:EN|ENGLISH|TRANSLATION)\s*[:：]\s*", re.IGNORECASE)
_CAPITALIZED = re.compile(r"\b[A-Z][A-Za-z]*(?:[-‐‑][A-Za-z]+)*")
_HONORIFICS = {"chan", "san", "sama", "kun", "senpai", "sensei", "tan"}


def build_prompt(text: str, terms: list[tuple[str, str]], context: str, style: str = "") -> str:
    block = _TERMS.format(terms="\n".join(f"{ja} 翻译成 {en}" for ja, en in terms)) if terms else ""
    if context:
        tail = _STYLE_CONTEXT.format(style=style, text=text) if style else _CONTEXT.format(text=text)
        return _BACKGROUND.format(context=context) + block + tail
    return block + (_STYLE_PLAIN.format(style=style, text=text) if style else _PLAIN.format(text=text))


# Alias for the __init__.py export name
build_hymt_prompt = build_prompt


def useful_context(text: str) -> bool:
    compact = re.sub(r"[\s、。！？!?…ー〜～]+", "", text)
    return bool(compact) and compact not in {
        "はい", "うん", "ううん", "え", "ええ", "あ", "ああ", "お", "おお", "そう", "そうそう",
    }


def build_context(header: str, history: list[tuple[str, str]], lines: int, max_chars: int,
                  *, english: bool = False) -> str:
    """Budgeted dialogue; English is opt-in because it can recycle translation errors."""
    header = header[:max_chars]
    sources: list[str] = []
    used = len(header) + (len(_DIALOGUE) + 1 if header else 0)
    recent = [(ja, en) for ja, en in history if useful_context(ja)]
    for ja, en in reversed(recent[-lines:] if lines else []):
        line = f"{ja}（英訳：{en}）" if english and en else ja
        if used + len(line) + bool(used) > max_chars:
            continue
        sources.insert(0, line)
        used += len(line) + bool(used)
    if header and sources:
        return "\n".join((header, _DIALOGUE, *sources))
    return "\n".join(part for part in (header, *sources) if part)


def missing_names(english: str, names: list[tuple[str, str]]) -> list[str]:
    missing = []
    for _, name in names:
        tokens = re.split(r"[- ]+", name)
        pattern = r"(?<!\w)" + r"[\s\-‐‑–—]+".join(re.escape(token) for token in tokens) + r"(?!\w)"
        if not re.search(pattern, english, re.IGNORECASE):
            missing.append(name)
    return missing


def garbled_name(english: str, missing: list[str], present: list[str] = ()) -> bool:
    """A capitalized near-miss (e.g. "Suixian" for "Sui-chan") is a wrong name, not an omission."""
    known = {token.casefold() for name in present for token in re.split(r"[-‐‑ ]+", name)}
    targets = set()
    for name in missing:
        parts = [part.casefold() for part in re.split(r"[-‐‑ ]+", name)]
        targets.add("".join(parts))
        targets.update(part for part in parts if len(part) >= 3 and part not in _HONORIFICS)
    for word in _CAPITALIZED.findall(english):
        parts = [part.casefold() for part in re.split(r"[-‐‑]", word)]
        honorific = len(parts) > 1 and parts[-1] in _HONORIFICS
        candidates = ({"".join(p for p in parts if p not in _HONORIFICS)} if honorific
                      else {"".join(parts), *parts})
        for candidate in candidates - known:
            for target in targets:
                threshold = .6 if honorific or len(target) > 3 else .8
                if SequenceMatcher(None, candidate, target).ratio() >= threshold:
                    return True
    return False


def clean_output(text: str) -> str:
    """Join lines and remove role prefixes, wrapping quotes and orphan quotes."""
    line = " ".join(part.strip() for part in text.strip().splitlines() if part.strip())
    line = _ROLE.sub("", line).strip()
    return unmatched_double_quote_cleanup(line.strip("「」『』"))


class HymtTranslator:
    """Tencent Hy-MT2-7B translator for Japanese-to-English subtitles."""

    def __init__(self, cfg: TranslateConfig, glossary: Glossary, *, stream: StreamContext | None = None,
                 gpu_layers: int | None = None) -> None:
        self.cfg = cfg
        self.glossary = glossary
        self.stream = stream or StreamContext()
        self._position: float | None = None
        self._anchor: tuple[str, str] | None = None
        self._newest: tuple[str, str] | None = None
        self.model = require_gguf(cfg.model_path, "Hy-MT2 model")
        self.layer_count = gguf_block_count(self.model) or 32
        requested = cfg.gpu_layers if gpu_layers is None else gpu_layers
        self.gpu_layers = self.layer_count if requested < 0 else min(requested, self.layer_count)
        self._relief = threading.Event()
        self._lock = threading.Lock()
        self._reload_lock = threading.Lock()  # Separate lock for reload to prevent deadlock
        self._closed = False
        self.executable = resolve_executable(cfg.server_path)
        self.server = LlamaServer("mt", self.executable, self._args(),
                                  startup_timeout_s=cfg.load_timeout_s)
        try:
            self._start()
        except BaseException:
            self.close()
            raise

    def _args(self) -> list[str]:
        """Build llama-server arguments for Hy-MT2 translation."""
        # Offload all layers +1 so the output layer also stays on the GPU.
        if self.gpu_layers >= self.layer_count:
            layers = self.gpu_layers + 1
        else:
            layers = self.gpu_layers
        device = detect_device(self.executable)
        return [
            "-m", str(self.model),
            "-c", str(self.cfg.context_size),
            "-np", str(self.cfg.parallel_slots),
            "--jinja",
            "-fa", "on",
            "-ctk", "q8_0",
            "-ctv", "q8_0",
            "-fit", "off",
            "-ngl", str(layers),
            "-t", str(self.cfg.threads),
            "--device", device,
        ]

    def _start(self) -> None:
        check_running()
        if self._closed:
            raise LlamaServerError("Hy-MT2 translator is closed")
        self.server.args = self._args()
        try:
            self.server.start()
            # Reserve dialogue-sized prefill workspace before measuring GPU headroom.
            chars = min(self.cfg.max_context_chars, self.cfg.context_size // 2)
            context = ("これは以前の会話です。" * (chars // 11 + 1))[:chars]
            prompt = build_prompt("こんにちは。今日はみんなでゲームをします。よろしくお願いします。",
                                  self.glossary.terms[:MAX_TERMS], context, self.cfg.style)
            self._chat(prompt, timeout=self.cfg.load_timeout_s)
            check_running()
            # Short batches use a different CUDA workspace than dialogue prefill.
            self._chat(build_prompt("みんな、スパチャありがとう！今日はマインクラフトで遊びます。", [], "",
                                    self.cfg.style),
                       timeout=self.cfg.load_timeout_s)
            check_running()
        except (httpx.HTTPError, ValueError) as exc:
            check_running()
            raise LlamaServerError(f"Hy-MT2 startup request failed: {exc}. {self.server.log_tail()}") from exc
        self._relief.clear()
        update_metrics(workers={"mt": f"ready (Hy-MT2-7B, {self.gpu_layers}/{self.layer_count} GPU layers)"})
        log.info("Hy-MT2 ready with %d/%d GPU layers (slots=%d)", self.gpu_layers, self.layer_count, self.cfg.parallel_slots)

    def reload(self, gpu_layers: int, reason: str) -> None:
        """Reload with different GPU layers. Uses separate lock to prevent deadlock."""
        with self._reload_lock:
            self.gpu_layers = max(0, min(gpu_layers, self.layer_count))
            log.warning("Reloading Hy-MT2 at %d GPU layers (%s)", self.gpu_layers, reason)
            record_event("mt_reload", gpu_layers=self.gpu_layers, reason=reason)
            update_metrics(workers={"mt": "reloading"})
            self._start()

    def request_gpu_relief(self) -> bool:
        """Ask for fewer GPU layers; applied between cues by the translation thread."""
        if self._closed or self.gpu_layers == 0:
            return False
        self._relief.set()
        return True

    def _chat(self, prompt: str, *, timeout: float, temperature: float | None = None,
              max_tokens: int | None = None) -> tuple[str, str]:
        """Send a chat request and return (content, finish_reason)."""
        actual_max = max_tokens if max_tokens is not None else self.cfg.max_tokens
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "top_k": self.cfg.top_k,
            "top_p": self.cfg.top_p,
            "repeat_penalty": self.cfg.repeat_penalty,
            "max_tokens": actual_max,
            "stream": False,
            "cache_prompt": True,
        }
        check_running()
        if self._closed:
            return "", ""
        started = time.monotonic()
        payload = self.server.chat(body, timeout=timeout)
        check_running()
        request_time = time.monotonic() - started
        choice = payload["choices"][0]
        content = choice["message"].get("content") or ""
        finish_reason = choice.get("finish_reason", "")
        if finish_reason == "length":
            log.debug("Truncated response (max_tokens=%d)", actual_max)
            increment_counter("mt_truncated")
        elif finish_reason == "stop":
            increment_counter("mt_finish_stop")
        elif finish_reason:
            log.debug("Unexpected finish_reason: %s", finish_reason)
            increment_counter("mt_finish_other")
        else:
            increment_counter("mt_finish_none")
        record_event(
            "mt_request",
            finish_reason=finish_reason,
            request_time_s=round(request_time, 3),
            max_tokens=actual_max,
            has_content=bool(content.strip()),
        )
        return content, finish_reason

    def set_position(self, start_s: float | None) -> None:
        self._position = start_s

    def _window(self, history: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Grow from a fixed first line, then restart at half size, so the prefix stays cached."""
        lines = self.cfg.context_lines
        usable = [pair for pair in history if useful_context(pair[0])][-lines:] if lines else []
        start = 0
        if self._anchor is not None:
            index = next((i for i, pair in enumerate(usable) if pair is self._anchor), None)
            if index is not None:
                start = index
            elif any(pair is self._newest for pair in usable):
                start = max(0, len(usable) - max(1, lines // 2))
        window = usable[start:]
        self._anchor = window[0] if window else None
        self._newest = usable[-1] if usable else None
        return window

    def _context(self, history: list[tuple[str, str]]) -> str:
        header = "\n".join(self.stream.lines(self._position))
        window = self._window(history)
        return build_context(header, window, len(window), self.cfg.max_context_chars,
                             english=self.cfg.context_english)

    def _terms_for(self, ja: str, context: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        source_context = f"{context}\n{ja}"
        names = self.glossary.name_terms(ja, context=source_context)[:MAX_TERMS]
        found = self.glossary.find(ja, context=source_context)
        extra = [term for term in found if term not in names]
        exact = [t for t in extra if t[0] == ja or f" {t[0]} " in f" {ja} "]
        rest = [t for t in extra if t not in exact]
        terms = names + exact + rest
        return terms[:MAX_TERMS], names

    def prompt_for(self, ja: str, history: list[tuple[str, str]]) -> str:
        context = self._context(history)
        terms, _ = self._terms_for(ja, context)
        return build_prompt(ja, terms, context, self.cfg.style)

    _MAX_RETRIES = 3
    _MAX_TRUNCATION_RETRIES = 1

    def translate(self, ja: str, history: list[tuple[str, str]]) -> str:
        # BUG FIX: Check relief and reload OUTSIDE the inference lock to prevent deadlock
        if self._relief.is_set():
            self._relief.clear()
            self.reload(self.gpu_layers - RELIEF_LAYERS, "GPU headroom low")

        with self._lock:
            if self._closed:
                return ""
            check_running()
            try:
                if not self.server.alive():
                    self._recover("server exited")
                context = self._context(history)
                terms, names = self._terms_for(ja, context)
                prompt = build_prompt(ja, terms, context, self.cfg.style)

                started = time.monotonic()
                latency_budget = self.cfg.timeout_s * 0.7

                retries_left = self._MAX_RETRIES
                truncation_retries = 0

                try:
                    english, finish_reason = self._chat(prompt, timeout=self.cfg.timeout_s)

                    if finish_reason == "length" and retries_left > 0 and truncation_retries < self._MAX_TRUNCATION_RETRIES:
                        retries_left -= 1
                        truncation_retries += 1
                        retry_tokens = max(self.cfg.max_tokens * 2, 512)
                        record_event("mt_truncation_retry", max_tokens=retry_tokens)
                        increment_counter("mt_retries")
                        english, finish_reason = self._chat(
                            prompt, timeout=self.cfg.timeout_s, max_tokens=retry_tokens
                        )
                        if finish_reason == "length":
                            record_event("mt_truncation_failed", max_tokens=retry_tokens)
                            increment_counter("mt_rejected")
                            return ""

                    english = clean_output(english)
                except (httpx.HTTPError, LlamaServerError, ValueError) as exc:
                    if self._closed:
                        return ""
                    check_running()
                    if self.server.alive():
                        raise
                    self._recover(str(exc))
                    # BUG FIX: Rebuild prompt after recovery — server state may have changed
                    prompt = build_prompt(ja, terms, context, self.cfg.style)
                    if retries_left > 0:
                        retries_left -= 1
                        increment_counter("mt_retries")
                    english, _ = self._chat(prompt, timeout=self.cfg.timeout_s)
                    english = clean_output(english)

                elapsed = time.monotonic() - started
                if elapsed > latency_budget:
                    log.warning("Translation latency %.2fs exceeded budget %.2fs", elapsed, latency_budget)
                    increment_counter("mt_latency_exceeded")

                # Retry 1: empty output
                if not english and retries_left > 0:
                    issue = english_output_issue(english)
                    if issue:
                        record_event("mt_rejected", reason=issue, retry=True)
                    minimal = build_prompt(ja, [], "")
                    retries_left -= 1
                    increment_counter("mt_retries")
                    english, _ = self._chat(minimal, timeout=self.cfg.timeout_s)
                    english = clean_output(english)

                # Retry 2: untranslated Japanese
                if retries_left > 0 and english:
                    output_issue = english_output_issue(english)
                    if output_issue == "untranslated_japanese":
                        record_event("mt_rejected", reason=output_issue, retry=True)
                        increment_counter("mt_rejected")
                        minimal = build_prompt(ja, [], "")
                        retries_left -= 1
                        increment_counter("mt_retries")
                        candidate, _ = self._chat(minimal, timeout=self.cfg.timeout_s, temperature=0.2)
                        cleaned = clean_output(candidate)
                        if cleaned and not english_output_issue(cleaned):
                            english = cleaned

                # Retry 3: missing or garbled names
                output_issue = english_output_issue(english)
                missing = missing_names(english, names)
                if not self._closed and retries_left > 0 and (output_issue or missing):
                    if output_issue:
                        record_event("mt_rejected", reason=output_issue, retry=True)
                        increment_counter("mt_rejected")
                    fallback = build_prompt(ja, terms, "")
                    candidates = [english]
                    if fallback != prompt or self.cfg.temperature != 0:
                        retries_left -= 1
                        increment_counter("mt_retries")
                        if missing:
                            record_event("mt_name_retry", names=missing)
                        candidate, _ = self._chat(fallback, timeout=self.cfg.timeout_s, temperature=0.0)
                        candidates.append(clean_output(candidate))
                    english = self._best(candidates, names)

                retry_count = self._MAX_RETRIES - retries_left
                record_event(
                    "mt_result",
                    has_output=bool(english),
                    retry_count=retry_count,
                    elapsed_s=round(time.monotonic() - started, 3),
                )

                return "" if self._closed else english
            except (httpx.HTTPError, LlamaServerError, ValueError):
                if self._closed:
                    return ""
                raise

    @staticmethod
    def _best(candidates: list[str], names: list[tuple[str, str]]) -> str:
        """Prefer valid output, then correct names; an omitted name beats a wrong one."""
        scored = []
        for index, text in enumerate(candidates):
            missing = missing_names(text, names)
            present = [en for _, en in names if en not in missing]
            scored.append((english_output_issue(text) is not None, garbled_name(text, missing, present),
                           len(missing), index, text, missing))
        issue, garbled, _, _, text, missing = min(scored)
        if issue or garbled:
            increment_counter("mt_rejected")
            if missing:
                record_event("mt_name_rejected", names=missing)
            return ""
        if missing:
            record_event("mt_name_omitted", names=missing)
        return text

    def _recover(self, reason: str) -> None:
        try:
            self.reload(self.gpu_layers - RELIEF_LAYERS * 2, reason)
        except LlamaServerError as exc:
            raise TranslatorUnavailable(f"Hy-MT2 server could not restart: {exc}") from exc

    def interrupt(self) -> None:
        self._closed = True
        server = getattr(self, "server", None)
        if server is not None:
            server.close()

    def close(self) -> None:
        self.interrupt()
