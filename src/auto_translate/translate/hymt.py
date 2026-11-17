"""Tencent HY-MT1.5-7B served by an owned llama.cpp server.

Prompts follow the official HY-MT templates: glossary hits use the
terminology-intervention form, recent dialogue uses the contextual form, and
both are combined when present. HY-MT is used without a system prompt.
"""
from __future__ import annotations

import logging
import re
import threading

import httpx

from ..config import TranslateConfig
from ..glossary import Glossary
from ..llama_server import LlamaServer, LlamaServerError, gguf_block_count, require_gguf, resolve_executable
from ..runtime_diagnostics import increment_counter, record_event, update_metrics
from ..subtitle_clean import english_output_issue, unmatched_double_quote_cleanup
from ..windows_subprocess import check_running
from .base import TranslatorUnavailable

log = logging.getLogger(__name__)

MAX_TERMS = 12
RELIEF_LAYERS = 4
_PLAIN = "Translate the following segment into English, without additional explanation.\n\n{text}"
_TERMS = "参考下面的翻译：\n{terms}\n\n"
_TERMS_ONLY = "将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：\n{text}"
_CONTEXT = "{context}\n参考上面的信息，把下面的文本翻译成英语，注意不需要翻译上文，也不要额外解释：\n{text}"
_ROLE = re.compile(r"^(?:EN|ENGLISH|TRANSLATION)\s*[:：]\s*", re.I)


def build_prompt(text: str, terms: list[tuple[str, str]], context: str) -> str:
    prefix = _TERMS.format(terms="\n".join(f"{ja} 翻译成 {en}" for ja, en in terms)) if terms else ""
    if context:
        return prefix + _CONTEXT.format(context=context, text=text)
    if prefix:
        return prefix + _TERMS_ONLY.format(text=text)
    return _PLAIN.format(text=text)


def useful_context(text: str) -> bool:
    compact = re.sub(r"[\s、。！？!?…ー〜～]+", "", text)
    return bool(compact) and compact not in {
        "はい", "うん", "ううん", "え", "ええ", "あ", "ああ", "お", "おお", "そう", "そうそう",
    }


def build_context(title: str, history: list[tuple[str, str]], lines: int, max_chars: int) -> str:
    """Budgeted Japanese-only dialogue avoids recycling translation errors."""
    title = title[:max_chars]
    sources: list[str] = []
    used = len(title)
    recent = [ja for ja, _ in history if useful_context(ja)]
    for ja in reversed(recent[-lines:] if lines else []):
        if used + len(ja) + bool(used) > max_chars:
            continue
        sources.insert(0, ja)
        used += len(ja) + bool(used)
    return "\n".join(part for part in (title, *sources) if part)


def missing_names(english: str, names: list[tuple[str, str]]) -> list[str]:
    missing = []
    for _, name in names:
        tokens = re.split(r"[- ]+", name)
        pattern = r"(?<!\w)" + r"[\s\-‐‑–—]+".join(re.escape(token) for token in tokens) + r"(?!\w)"
        if not re.search(pattern, english, re.I):
            missing.append(name)
    return missing


def clean_output(text: str) -> str:
    """Join lines and remove role prefixes, wrapping quotes and orphan quotes."""
    line = " ".join(part.strip() for part in text.strip().splitlines() if part.strip())
    line = _ROLE.sub("", line).strip()
    return unmatched_double_quote_cleanup(line.strip("「」『』"))


class HymtTranslator:
    def __init__(self, cfg: TranslateConfig, glossary: Glossary, *, vod_title: str = "",
                 gpu_layers: int | None = None) -> None:
        self.cfg = cfg
        self.glossary = glossary
        self.title = vod_title.strip()[:200]
        self.model = require_gguf(cfg.model_path, "HY-MT model")
        self.layer_count = gguf_block_count(self.model) or 32
        requested = cfg.gpu_layers if gpu_layers is None else gpu_layers
        self.gpu_layers = self.layer_count if requested < 0 else min(requested, self.layer_count)
        self._relief = threading.Event()
        self._lock = threading.Lock()
        self._closed = False
        self.server = LlamaServer("mt", resolve_executable(cfg.server_path), self._args(),
                                  startup_timeout_s=cfg.load_timeout_s)
        try:
            self._start()
        except BaseException:
            self.close()
            raise

    def _args(self) -> list[str]:
        # Offload all layers +1 so the output layer also stays on the GPU.
        layers = self.gpu_layers + 1 if self.gpu_layers >= self.layer_count else self.gpu_layers
        return ["-m", str(self.model), "-c", str(self.cfg.context_size), "-np", "1", "--jinja",
                "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-fit", "off", "-ngl", str(layers),
                "-t", str(self.cfg.threads), "--device", f"CUDA{self.cfg.cuda_device}"]

    def _start(self) -> None:
        check_running()
        if self._closed:
            raise LlamaServerError("HY-MT translator is closed")
        self.server.args = self._args()
        try:
            self.server.start()
            # Reserve dialogue-sized prefill workspace before measuring GPU headroom.
            chars = min(self.cfg.max_context_chars, self.cfg.context_size // 2)
            context = ("これは以前の会話です。" * (chars // 11 + 1))[:chars]
            prompt = build_prompt("こんにちは。今日はみんなでゲームをします。よろしくお願いします。",
                                  self.glossary.terms[:MAX_TERMS], context)
            self._chat(prompt, timeout=self.cfg.load_timeout_s)
            check_running()
            # Short batches use a different CUDA workspace than dialogue prefill.
            self._chat(build_prompt("みんな、スパチャありがとう！今日はマインクラフトで遊びます。", [], ""),
                       timeout=self.cfg.load_timeout_s)
            check_running()
        except (httpx.HTTPError, ValueError) as exc:
            check_running()
            raise LlamaServerError(f"HY-MT startup request failed: {exc}. {self.server.log_tail()}") from exc
        self._relief.clear()
        update_metrics(workers={"mt": f"ready (HY-MT1.5-7B, {self.gpu_layers}/{self.layer_count} GPU layers)"})
        log.info("HY-MT ready with %d/%d GPU layers", self.gpu_layers, self.layer_count)

    def reload(self, gpu_layers: int, reason: str) -> None:
        self.gpu_layers = max(0, min(gpu_layers, self.layer_count))
        log.warning("Reloading HY-MT at %d GPU layers (%s)", self.gpu_layers, reason)
        record_event("mt_reload", gpu_layers=self.gpu_layers, reason=reason)
        update_metrics(workers={"mt": "reloading"})
        self._start()

    def request_gpu_relief(self) -> bool:
        """Ask for fewer GPU layers; applied between cues by the translation thread."""
        if self._closed or self.gpu_layers == 0:
            return False
        self._relief.set()
        return True

    def _chat(self, prompt: str, *, timeout: float, temperature: float | None = None) -> str:
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "top_k": self.cfg.top_k, "top_p": self.cfg.top_p,
            "repeat_penalty": self.cfg.repeat_penalty,
            "max_tokens": self.cfg.max_tokens, "stream": False,
        }
        check_running()
        if self._closed:
            return ""
        payload = self.server.chat(body, timeout=timeout)
        check_running()
        choice = payload["choices"][0]
        return "" if choice.get("finish_reason") == "length" else choice["message"]["content"]

    def _terms_for(self, ja: str, context: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        source_context = "\n".join((context, ja))
        names = self.glossary.name_terms(ja, context=source_context)[:MAX_TERMS]
        terms = names + [term for term in self.glossary.find(ja, context=source_context) if term not in names]
        return terms[:MAX_TERMS], names

    def prompt_for(self, ja: str, history: list[tuple[str, str]]) -> str:
        context = build_context(self.title, history, self.cfg.context_lines, self.cfg.max_context_chars)
        terms, _ = self._terms_for(ja, context)
        return build_prompt(ja, terms, context)

    def translate(self, ja: str, history: list[tuple[str, str]]) -> str:
        with self._lock:
            if self._closed:
                return ""
            check_running()
            try:
                if self._relief.is_set():
                    self._relief.clear()
                    self.reload(self.gpu_layers - RELIEF_LAYERS, "GPU headroom low")
                if not self.server.alive():
                    self._recover("server exited")
                context = build_context(self.title, history, self.cfg.context_lines, self.cfg.max_context_chars)
                terms, names = self._terms_for(ja, context)
                prompt = build_prompt(ja, terms, context)
                try:
                    english = clean_output(self._chat(prompt, timeout=self.cfg.timeout_s))
                except (httpx.HTTPError, LlamaServerError, ValueError) as exc:
                    if self._closed:
                        return ""
                    check_running()
                    if self.server.alive():
                        raise
                    self._recover(str(exc))
                    english = clean_output(self._chat(prompt, timeout=self.cfg.timeout_s))
                missing = missing_names(english, names)
                if not self._closed and (english_output_issue(english) or missing):
                    increment_counter("mt_retries")
                    if missing:
                        record_event("mt_name_retry", names=missing)
                    retry = clean_output(self._chat(build_prompt(ja, terms, ""), timeout=self.cfg.timeout_s,
                                                    temperature=0.0))
                    rejected_names = missing_names(retry, names)
                    if english_output_issue(retry) or rejected_names:
                        increment_counter("mt_rejected")
                        if rejected_names:
                            record_event("mt_name_rejected", names=rejected_names)
                        english = ""
                    else:
                        english = retry
                return "" if self._closed else english
            except (httpx.HTTPError, LlamaServerError, ValueError):
                if self._closed:
                    return ""
                raise

    def _recover(self, reason: str) -> None:
        # A dead server on a shared GPU is almost always an allocation failure.
        try:
            self.reload(self.gpu_layers - RELIEF_LAYERS * 2, reason)
        except LlamaServerError as exc:
            raise TranslatorUnavailable(f"HY-MT server could not restart: {exc}") from exc

    def interrupt(self) -> None:
        self._closed = True
        server = getattr(self, "server", None)
        if server is not None:
            server.close()

    def close(self) -> None:
        self.interrupt()
