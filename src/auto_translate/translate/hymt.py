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


def missing_names(english: str, names: list[str]) -> list[str]:
    """Find names that are missing in the translation.

    Args:
        english: the translated English text.
        names: English name strings to check for (e.g. ["Miko", "Subaru"]).
    """
    missing = []
    for name in names:
        tokens = re.split(r"[- ]+", name)
        pattern = r"(?<!\w)" + r"[\s\-‐‑–—]+".join(re.escape(token) for token in tokens) + r"(?!\w)"
        if not re.search(pattern, english, re.IGNORECASE):
            missing.append(name)
    return missing


def garbled_name(english: str, missing: list[str], present: list[str] = (),
                 *, utterance_names: list[str] | None = None) -> bool:
    """A capitalized near-miss (e.g. "Suixian" for "Sui-chan") is a wrong name, not an omission.

    Only treats a word as a garbled name when it is a clear fuzzy match for a
    name that should appear — not for common English words that happen to
    resemble a glossary entry (e.g. "Sure" ≠ "Subaru").

    Args:
        utterance_names: names explicitly mentioned in the current Japanese
            utterance. When provided, only these names trigger garbled
            rejection. Context-only names are treated as hints, not mandatory.
    """

    _COMMON = frozenset({
        "sure", "so", "some", "same", "see", "say", "said", "set", "sent",
        "sentiment", "sense", "since", "site", "side", "sight", "sign",
        "simple", "size", "skill", "skin", "skip", "small",
        "smart", "smile", "soft", "someone", "something",
        "son", "song", "soon", "sort", "sound", "source", "space", "span",
        "speak", "special", "speed", "spell", "spend", "spirit", "split",
        "sport", "spot", "spread", "spring", "square", "staff", "stage",
        "stand", "start", "state", "stay", "step", "still", "story",
        "straight", "strange", "street", "strong", "structure", "study",
        "stuff", "style", "subject", "such", "sudden", "suffer", "suggest",
        "suit", "summer", "support", "suppose", "surface", "system",
        "table", "take", "talk", "task", "tax", "teach", "team", "tell",
        "ten", "tend", "term", "test", "text", "than", "thank", "that",
        "the", "their", "them", "then", "there", "these", "they", "thing",
        "think", "this", "those", "though", "thought", "thread", "three",
        "through", "throw", "thus", "time", "to", "today", "together",
        "tonight", "too", "took", "top", "total", "touch", "toward",
        "town", "track", "trade", "train", "transfer", "travel", "treat",
        "tree", "trip", "trouble", "true", "turn", "two", "type",
        "under", "understand", "unit", "until", "up", "upon", "us", "use",
        "usually", "value", "various", "very", "view", "visit", "voice",
        "wait", "walk", "wall", "want", "war", "watch", "water", "way",
        "we", "wear", "week", "well", "west", "what", "when", "where",
        "which", "while", "white", "who", "whole", "why", "wide", "will",
        "win", "wind", "window", "wish", "with", "within", "without",
        "woman", "wonder", "word", "work", "world", "worry", "would",
        "write", "year", "yes", "yet", "you", "young", "your",
        "i", "me", "my", "our", "he", "him",
        "his", "she", "her", "it", "about", "above", "after", "again", "against", "all", "also",
        "always", "among", "an", "and", "another", "any", "appear",
        "around", "as", "ask", "at", "back", "be", "because", "been",
        "before", "begin", "behind", "being", "below", "best", "better",
        "between", "big", "both", "but", "by", "call", "can", "come",
        "could", "day", "did", "different", "do", "does", "done", "during",
        "each", "early", "end", "even", "ever", "every", "example",
        "eye", "face", "fact", "far", "feel", "few", "find", "first",
        "follow", "for", "from", "full", "further", "get", "give", "go",
        "good", "great", "had", "has", "have", "here", "how", "however", "if", "important", "in", "into", "is",
        "its", "just", "keep", "kind", "know", "last", "late", "least",
        "less", "let", "like", "long", "look", "make", "many", "may",
        "might", "most", "must", "need", "never", "new", "next",
        "no", "not", "now", "number", "of", "off", "often", "oh", "ok",
        "old", "on", "one", "only", "or", "other", "out", "over",
        "own", "part", "people", "place", "point", "possible", "problem",
        "put", "quite", "rather", "really", "right", "saw", "seem", "shall", "should", "show", "was", "were", "surely", "sugar", "surprise", "similar", "sincere",
    })

    if utterance_names is not None:
        utterance_set = {n.casefold() for n in utterance_names}
        missing = [n for n in missing if n.casefold() in utterance_set]

    if not missing:
        return False

    known = {token.casefold() for name in present for token in re.split(r"[-‐‑ ]+", name)}
    targets = set()
    for name in missing:
        parts = [part.casefold() for part in re.split(r"[-‐‑ ]+", name)]
        targets.add("".join(parts))
        targets.update(part for part in parts if len(part) >= 4 and part not in _HONORIFICS)

    for word in _CAPITALIZED.findall(english):
        parts = [part.casefold() for part in re.split(r"[-‐‑]", word)]
        honorific = len(parts) > 1 and parts[-1] in _HONORIFICS
        candidates = ({"".join(p for p in parts if p not in _HONORIFICS)} if honorific
                      else {"".join(parts), *parts})
        for candidate in candidates - known:
            if candidate in _COMMON:
                continue
            for target in targets:
                if len(target) <= 3:
                    threshold = 0.85
                elif honorific:
                    threshold = 0.7
                else:
                    threshold = 0.75
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

    def _terms_for(self, ja: str, context: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
        """Return (terms, all_names, utterance_names).

        `terms` — all glossary terms for the prompt.
        `all_names` — names found in the utterance OR prior context (used for missing_names check).
        `utterance_names` — names actually present in the current Japanese text (used for garbled_name check).

        Key distinction: all_names includes names from prior dialogue context so that
        context-only names do not trigger garbled rejection. utterance_names contains
        only names actually spoken in the current utterance, using source context for
        disambiguation so that context-dependent nicknames (e.g. Miko) are recognized.
        """
        source_context = f"{context}\n{ja}"
        # Names in the current utterance with source context for disambiguation.
        # This lets context-dependent nicknames like Miko be recognized.
        utterance_names = self.glossary.name_terms(ja, context=source_context)
        # Also gather names mentioned in prior context — these are hints, not mandatory.
        context_names = []
        if context:
            context_names = self.glossary.name_terms(context, context="")
        # all_names = utterance names + context-only names (for missing_names check)
        all_names = list(utterance_names)
        seen = {en for _, en in utterance_names}
        for term, en in context_names:
            if en not in seen:
                all_names.append((term, en))
                seen.add(en)
        all_names = all_names[:MAX_TERMS]
        # Current speech takes precedence over optional context hints at the cap.
        context_only_names = [name for name in all_names if name not in utterance_names]
        found = self.glossary.find(ja, context=source_context)
        extra = [term for term in found if term not in utterance_names]
        exact = [t for t in extra if t[0] == ja or f" {t[0]} " in f" {ja} "]
        rest = [t for t in extra if t not in exact]
        terms = utterance_names + exact + rest + context_only_names
        return terms[:MAX_TERMS], all_names, utterance_names

    def prompt_for(self, ja: str, history: list[tuple[str, str]]) -> str:
        context = self._context(history)
        terms, _, _ = self._terms_for(ja, context)
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
                terms, _all_names, utterance_names = self._terms_for(ja, context)
                prompt = build_prompt(ja, terms, context, self.cfg.style)

                started = time.monotonic()
                latency_budget = self.cfg.timeout_s * 0.7

                retries_left = self._MAX_RETRIES
                truncation_retries = 0

                # Estimate input tokens for context-budget checks
                ja_tokens = int(len(ja) * 1.8)
                glossary_tokens = sum(
                    7 + int((len(source) + len(target)) * 1.8)
                    for source, target in terms
                )
                context_tokens = int(len(context) * 1.8) if context else 0
                template_overhead = 50
                input_tokens = ja_tokens + glossary_tokens + context_tokens + template_overhead
                slot_context_size = self.cfg.context_size // self.cfg.parallel_slots

                def _fits_context(itokens: int, otokens: int) -> bool:
                    return itokens + otokens + 256 <= slot_context_size

                simplified_input = ja_tokens + template_overhead
                if not _fits_context(simplified_input, self.cfg.max_tokens):
                    record_event("mt_rejected", reason="context_budget",
                                 estimated_input_tokens=simplified_input)
                    increment_counter("mt_rejected")
                    log.warning("Translation utterance exceeds per-slot context budget")
                    return ""
                skip_full_prompt = not _fits_context(input_tokens, self.cfg.max_tokens)
                if skip_full_prompt:
                    prompt = build_prompt(ja, [], "", self.cfg.style)
                    record_event("mt_request", strategy="simplified_first",
                                 estimated_input_tokens=input_tokens)

                try:
                    english, finish_reason = self._chat(prompt, timeout=self.cfg.timeout_s)

                    if finish_reason == "length" and retries_left > 0 and truncation_retries < self._MAX_TRUNCATION_RETRIES:
                        retries_left -= 1
                        truncation_retries += 1

                        # --- Context-aware truncation retry ---
                        # Independently calculate token budgets for each recovery strategy.
                        # Each budget accounts for: per-slot context, input tokens, output tokens,
                        # and a 256-token safety margin. Never submit max_tokens <= 0.
                        #
                        # Recovery order (least destructive first):
                        #   1. Full prompt with increased output allowance
                        #   2. Glossary-only prompt (keeps names, drops dialogue context)
                        #   3. Simplified prompt (no glossary, no context)

                        simplified_input = ja_tokens + template_overhead
                        glossary_only_input = ja_tokens + glossary_tokens + template_overhead

                        # Output budgets for each strategy
                        full_output_budget = max(0, slot_context_size - input_tokens - 256)
                        glossary_output_budget = max(0, slot_context_size - glossary_only_input - 256)
                        simplified_output_budget = max(0, slot_context_size - simplified_input - 256)

                        # Desired retry output: at least double the original, minimum 512
                        desired_output = max(self.cfg.max_tokens * 2, 512)

                        # Select the least destructive strategy that fits a meaningful increase
                        retry_prompt = None
                        retry_max_tokens = 0
                        retry_strategy = ""

                        # 1. Full prompt: preserves everything (glossary + context)
                        if not skip_full_prompt and full_output_budget > self.cfg.max_tokens:
                            retry_max_tokens = min(desired_output, full_output_budget)
                            retry_prompt = prompt
                            retry_strategy = "full_increased_tokens"
                        # 2. Glossary-only: keeps names, drops dialogue context
                        elif glossary_output_budget > self.cfg.max_tokens:
                            retry_max_tokens = min(desired_output, glossary_output_budget)
                            retry_prompt = build_prompt(ja, terms, "", self.cfg.style)
                            retry_strategy = "glossary_only"
                        # 3. Simplified: no glossary, no context — last resort
                        elif simplified_output_budget > self.cfg.max_tokens:
                            retry_max_tokens = min(desired_output, simplified_output_budget)
                            retry_prompt = build_prompt(ja, [], "", self.cfg.style)
                            retry_strategy = "simplified"

                        # Execute retry only if it meaningfully increases output allowance
                        if retry_prompt is not None and retry_max_tokens > self.cfg.max_tokens:
                            record_event("mt_truncation_retry", strategy=retry_strategy,
                                         max_tokens=retry_max_tokens)
                            increment_counter("mt_retries")
                            english, finish_reason = self._chat(
                                retry_prompt, timeout=self.cfg.timeout_s, max_tokens=retry_max_tokens
                            )
                        else:
                            record_event("mt_truncation_failed", reason="no_viable_retry",
                                         full_budget=full_output_budget,
                                         glossary_budget=glossary_output_budget,
                                         simplified_budget=simplified_output_budget)
                            increment_counter("mt_rejected")
                            return ""

                        if finish_reason == "length":
                            record_event("mt_truncation_failed", max_tokens=retry_max_tokens)
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
                    prompt = (build_prompt(ja, [], "", self.cfg.style) if skip_full_prompt
                              else build_prompt(ja, terms, context, self.cfg.style))
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

                # Retry 2: untranslated Japanese — try deterministically
                if retries_left > 0 and english:
                    output_issue = english_output_issue(english)
                    if output_issue == "untranslated_japanese":
                        record_event("mt_rejected", reason=output_issue, retry=True)
                        increment_counter("mt_rejected")
                        minimal = build_prompt(ja, [], "")
                        retries_left -= 1
                        increment_counter("mt_retries")
                        candidate, _ = self._chat(minimal, timeout=self.cfg.timeout_s, temperature=0.0)
                        cleaned = clean_output(candidate)
                        if cleaned and not english_output_issue(cleaned):
                            english = cleaned

                # Retry 3: missing or garbled names
                # Only check names actually spoken in the current utterance.
                # Context-only names must not trigger name-correction retries.
                output_issue = english_output_issue(english)
                utterance_name_list = [en for _, en in utterance_names]
                missing = missing_names(english, utterance_name_list)
                if not self._closed and retries_left > 0 and (output_issue or missing):
                    if output_issue:
                        record_event("mt_rejected", reason=output_issue, retry=True)
                        increment_counter("mt_rejected")
                    fallback = build_prompt(ja, terms, "", self.cfg.style)
                    fallback_tokens = simplified_input + glossary_tokens
                    fallback_fits = _fits_context(fallback_tokens, self.cfg.max_tokens)
                    candidates = [english]
                    if fallback_fits and (fallback != prompt or self.cfg.temperature != 0):
                        retries_left -= 1
                        increment_counter("mt_retries")
                        if missing:
                            record_event("mt_name_retry", names=missing)
                        candidate, _ = self._chat(fallback, timeout=self.cfg.timeout_s, temperature=0.0)
                        candidates.append(clean_output(candidate))
                    english = self._best(candidates, utterance_name_list)

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
    def _best(candidates: list[str], names: list[str]) -> str:
        """Prefer valid output, then correct names; an omitted name beats a wrong one.

        Args:
            candidates: list of translation candidates.
            names: English name strings actually spoken in the current utterance.
                Context-only names must not be included — they are hints, not mandatory.
        """
        scored = []
        for index, text in enumerate(candidates):
            missing = missing_names(text, names)
            present = [n for n in names if n not in missing]
            garbled = garbled_name(text, missing, present, utterance_names=names)
            scored.append((english_output_issue(text) is not None, garbled,
                           len(missing), index, text, missing))
        issue, garbled, _, _, text, missing = min(scored)
        if issue:
            increment_counter("mt_rejected")
            if missing:
                record_event("mt_name_rejected", names=missing)
            return ""
        if garbled:
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
