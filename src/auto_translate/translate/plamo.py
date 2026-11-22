"""PLaMo 3 Translate adapter for Japanese-to-English subtitles.

PLaMo 3 Translate (pfnet/plamo-3-translate) is a translation-specialized model
that uses a dedicated chat template with target_lang kwargs instead of a system
prompt. It requires llama.cpp 0.7.0+ for the plamo3 tokenizer.

Key differences from Gemma4:
- No system prompt — translation is driven by the chat template
- Uses chat_template_kwargs with target_lang in the API body
- Specific stop tokens: <|plamo:tag|> and <|plamo:bos|>
- Requires --alias flag for the model name in API requests
- Uses --chat-template-file for the official jinja template
"""
from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path

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
from ..subtitle_clean import (
    english_output_issue,
    looks_like_prompt_echo,
    unmatched_double_quote_cleanup,
)
from ..windows_subprocess import check_running
from .base import TranslatorUnavailable

log = logging.getLogger(__name__)

MAX_TERMS = 16
RELIEF_LAYERS = 4

# PLaMo-specific stop tokens
PLAMO_STOP_TOKENS = ["<|plamo:tag|>", "<|plamo:bos|>"]

# PLaMo output cleaning — similar to Gemma4 but adapted for PLaMo's output format
_SYSTEM_ECHO = re.compile(
    r"(?:"
    r"you are a professional"
    r"|japanese-to-english subtitle translator"
    r"|output only the english translation"
    r"|output exactly one english translation"
    r"|never output slash-separated"
    r"|never add explanatory text"
    r"|prioritize meaning and natural flow"
    r"|preserve japanese names"
    r")",
    re.IGNORECASE,
)

_PROMPT_ECHO = re.compile(
    r"(?:"
    r"reference context \(not part of the speech"
    r"|translate this japanese text"
    r"|name/term glossary"
    r"|translation style\s*[:：]"
    r"|直前の会話"
    r")",
    re.IGNORECASE,
)

_ROLE = re.compile(r"^(?:EN|ENGLISH|TRANSLATION|Translate)\s*[:：]\s*", re.IGNORECASE)
_MARKDOWN = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*|(?<!\w)\*(?=\S)(.+?)(?<=\S)\*(?!\w)")
_CAPITALIZED = re.compile(r"\b[A-Z][A-Za-z]*(?:[-‐‑][A-Za-z]+)*")
_HONORIFICS = {"chan", "san", "sama", "kun", "senpai", "sensei", "tan"}

_EXPLANATORY = re.compile(
    r"^(?:"
    r"(?:英訳|翻訳)\s*[:：]"
    r"|(?:Note|Notes|Explanation|Explanations|Alternative|Alternatives|Meaning|Definition)\s*[:：]"
    r"|(?:Here is|Here's)\s+(?:the\s+)?translation"
    r"|The\s+translation(?:\s+(?:is|of))"
    r"|This\s+means\s*[:：]"
    r"|(?:I would translate|I believe)\s"
    r")",
    re.IGNORECASE,
)


def useful_context(text: str) -> bool:
    """Check if text contains meaningful content worth keeping in context."""
    compact = re.sub(r"[\s、。！？!?…ー〜～]+", "", text)
    return bool(compact) and compact not in {
        "はい", "うん", "ううん", "え", "ええ", "あ", "ああ", "お", "おお", "そう", "そうそう",
    }


def _strip_reasoning_tags(text: str) -> str:
    """Strip reasoning/thinking tags and their content from model output."""
    # PLaMo doesn't use reasoning tags, but handle them defensively
    reasoning_block = re.compile(
        r"<think(?:ing)?(?:\s[^>]*)?>.*?</think(?:ing)?\s*>"
        r"|<reasoning(?:\s[^>]*)?>.*?</reasoning\s*>"
        r"|<\|think\|>.*?<\|end_of_think\|>"
        r"|<\|channel>thought\s.*?\s<channel\|>",
        re.DOTALL,
    )
    reasoning_open = re.compile(
        r"<\/?(?:think(?:ing)?|reasoning)(?:\s[^>]*)?>"
        r"|<\|think\|>"
        r"|<\|end_of_think\|>"
        r"|<\|channel>thought",
    )
    result = reasoning_block.sub("", text)
    result = reasoning_open.sub("", result)
    result = " ".join(part.strip() for part in result.strip().splitlines() if part.strip())
    return result


def clean_output(text: str) -> str:
    """Clean PLaMo translation output: strip reasoning tags, remove role prefixes, markdown, orphan quotes."""
    line = " ".join(part.strip() for part in text.strip().splitlines() if part.strip())
    if not line:
        return ""
    line = _strip_reasoning_tags(line)
    if not line:
        return ""
    if looks_like_prompt_echo(line):
        return ""
    if _SYSTEM_ECHO.search(line):
        return ""
    if _PROMPT_ECHO.search(line):
        return ""
    if any(phrase in line.casefold() for phrase in (
        "output only the translation",
        "translate only the current japanese",
        "reference below is not part of the current speech",
        "you are a professional japanese-to-english subtitle translator",
    )):
        return ""
    line = _ROLE.sub("", line).strip()
    if _EXPLANATORY.match(line):
        return ""
    line = _MARKDOWN.sub(lambda match: match[1] or match[2], line)
    return unmatched_double_quote_cleanup(line.strip("「」『』"))


def missing_names(english: str, names: list[tuple[str, str]]) -> list[str]:
    """Find names from the glossary that are missing in the translation."""

    missing = []
    for _, name in names:
        tokens = re.split(r"[- ]+", name)
        pattern = r"(?<!\w)" + r"[\s\-‐‑–—]+".join(re.escape(token) for token in tokens) + r"(?!\w)"
        if not re.search(pattern, english, re.IGNORECASE):
            missing.append(name)
    return missing


def garbled_name(english: str, missing: list[str], present: list[str] = (),
                 *, utterance_names: list[str] | None = None) -> bool:
    """A capitalized near-miss is a wrong name, not an omission.

    Only treats a word as a garbled name when it is a clear fuzzy match for a
    name that should appear — not for common English words that happen to
    resemble a glossary entry (e.g. "Sure" ≠ "Subaru").

    Args:
        utterance_names: names explicitly mentioned in the current Japanese
            utterance. When provided, only these names trigger garbled
            rejection. Context-only names are treated as hints, not mandatory.
    """
    from difflib import SequenceMatcher

    # Common English words that should never be flagged as garbled names,
    # even if they fuzzy-match a glossary entry.
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
        "put", "quite", "rather", "really", "right", "saw", "seem", "shall", "should", "show", "was", "were", # Words that previously caused false positives with VTuber names
        "surely", "sugar", "surprise", "similar", "sincere",
    })

    # Determine which names are "mandatory" — i.e. explicitly in the utterance.
    # When utterance_names is provided, only those names can trigger garbled
    # rejection. Context-only names are hints; missing them is not an error.
    if utterance_names is not None:
        # Filter missing to only utterance names
        utterance_set = {n.casefold() for n in utterance_names}
        missing = [n for n in missing if n.casefold() in utterance_set]

    if not missing:
        return False

    known = {token.casefold() for name in present for token in re.split(r"[-‐‑ ]+", name)}
    targets = set()
    for name in missing:
        parts = [part.casefold() for part in re.split(r"[-‐‑ ]+", name)]
        targets.add("".join(parts))
        # Require at least 4 characters for fuzzy matching to reduce false positives
        targets.update(part for part in parts if len(part) >= 4 and part not in _HONORIFICS)

    for word in _CAPITALIZED.findall(english):
        parts = [part.casefold() for part in re.split(r"[-‐‑]", word)]
        honorific = len(parts) > 1 and parts[-1] in _HONORIFICS
        candidates = ({"".join(p for p in parts if p not in _HONORIFICS)} if honorific
                      else {"".join(parts), *parts})
        for candidate in candidates - known:
            # Skip common English words — they are not garbled names
            if candidate in _COMMON:
                continue
            for target in targets:
                # Higher threshold for short targets to avoid false positives
                # "su" (2 chars) matching "suba" should not trigger at 0.6
                if len(target) <= 3:
                    threshold = 0.85
                elif honorific:
                    threshold = 0.7
                else:
                    threshold = 0.75
                if SequenceMatcher(None, candidate, target).ratio() >= threshold:
                    return True
    return False


def build_context(header: str, history: list[tuple[str, str]], lines: int, max_chars: int,
                  *, english: bool = False) -> str:
    """Budgeted dialogue context for PLaMo."""
    _DIALOGUE = "直前の会話："
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


def build_plamo_prompt(text: str, terms: list[tuple[str, str]], context: str, *, simplified: bool = False) -> dict:
    """Build a chat-completion prompt for PLaMo 3 Translate.

    PLaMo uses the official chat template: Japanese speech is the user message
    and source_lang/target_lang (plus optional context/glossary) are passed via
    chat_template_kwargs, never embedded in the user text. This keeps the source
    separate from instructions (no prompt leakage into subtitles) and reduces
    input tokens for faster prefill.

    When `simplified=True`, omit glossary/context to reduce input tokens
    for truncation recovery retries.
    """
    kwargs: dict[str, object] = {"source_lang": "ja", "target_lang": "en"}
    if simplified:
        user_content = text
    else:
        user_content = text
        if context:
            kwargs["context"] = context
        if terms:
            kwargs["glossary"] = [{"source": ja, "target": en} for ja, en in terms]
    return {
        "messages": [{"role": "user", "content": user_content}],
        "chat_template_kwargs": kwargs,
    }


class PlamoTranslator:
    """PLaMo 3 Translate adapter for Japanese-to-English subtitles.

    Uses the same TranslateConfig as Gemma4Translator but with PLaMo-specific
    prompt format, server arguments, and stop tokens.
    """

    def __init__(self, cfg: TranslateConfig, glossary: Glossary, *, stream: StreamContext | None = None,
                 gpu_layers: int | None = None) -> None:
        self.cfg = cfg
        self.glossary = glossary
        self.stream = stream or StreamContext()
        self._position: float | None = None
        self._anchor: tuple[str, str] | None = None
        self._newest: tuple[str, str] | None = None
        self.model = require_gguf(cfg.model_path, "PLaMo 3 Translate model")
        self.layer_count = gguf_block_count(self.model) or 32
        requested = cfg.gpu_layers if gpu_layers is None else gpu_layers
        self.gpu_layers = self.layer_count if requested < 0 else min(requested, self.layer_count)
        self._relief = threading.Event()
        self._lock = threading.Lock()
        self._reload_lock = threading.Lock()
        self._closed = False
        self._failures = 0
        self._restarts = 0
        self.executable = resolve_executable(cfg.server_path)
        self.server = LlamaServer("mt", self.executable, self._args(),
                                  startup_timeout_s=cfg.load_timeout_s)
        try:
            self._start()
        except BaseException:
            self.close()
            raise

    def _args(self) -> list[str]:
        """Build llama-server arguments for PLaMo 3 Translate."""
        if self.gpu_layers >= self.layer_count:
            layers = 999
        else:
            layers = self.gpu_layers + 1
        device = detect_device(self.executable)
        
        # Find the chat template file next to the model or in the same directory
        template_path = self._find_chat_template()
        
        args = [
            "-m", str(self.model),
            "-c", str(self.cfg.context_size),
            "-np", str(self.cfg.parallel_slots),
            "--jinja",
            "--no-prefill-assistant",
            "-fa", "on",
            "-fit", "off",
            "-ngl", str(layers),
            "-t", str(self.cfg.threads),
            "--device", device,
            "--alias", "plamo-3-translate",
            "--reasoning", "off",
        ]
        
        if template_path:
            args.extend(["--chat-template-file", str(template_path)])
        
        return args

    def _find_chat_template(self) -> Path | None:
        """Find the chat_template.jinja file for PLaMo 3 Translate.

        Looks in the same directory as the model file, then in common model directories.
        """
        model_dir = self.model.parent
        # Check next to the model file
        template = model_dir / "chat_template.jinja"
        if template.is_file():
            return template
        # Check in the models/plamo3 directory
        plamo_dir = model_dir.parent / "plamo3"
        if plamo_dir.is_dir():
            template = plamo_dir / "chat_template.jinja"
            if template.is_file():
                return template
        log.warning(
            "chat_template.jinja not found — PLaMo will use the embedded template. "
            "Download from https://huggingface.co/pfnet/plamo-3-translate-2b-GGUF/raw/main/chat_template.jinja "
            "and place it next to the model file for best results."
        )
        return None

    def _start(self) -> None:
        check_running()
        if self._closed:
            raise LlamaServerError("PLaMo 3 Translate translator is closed")
        self.server.args = self._args()
        try:
            self.server.start()
            # Warmup with a simple translation
            prompt = build_plamo_prompt("こんにちは。今日はみんなでゲームをします。よろしくお願いします。", [], "")
            self._chat(prompt, timeout=self.cfg.load_timeout_s)
            check_running()
            self._chat(build_plamo_prompt("みんな、スパチャありがとう！今日はマインクラフトで遊びます。", [], ""),
                       timeout=self.cfg.load_timeout_s)
            check_running()
        except (httpx.HTTPError, ValueError) as exc:
            check_running()
            raise LlamaServerError(f"PLaMo 3 Translate startup request failed: {exc}. {self.server.log_tail()}") from exc
        self._relief.clear()
        update_metrics(workers={"mt": f"ready (PLaMo 3 Translate, {self.gpu_layers}/{self.layer_count} GPU layers)"})
        log.info("PLaMo 3 Translate ready with %d/%d GPU layers (slots=%d)", self.gpu_layers, self.layer_count, self.cfg.parallel_slots)

    def reload(self, gpu_layers: int, reason: str) -> None:
        """Reload with different GPU layers. Uses separate lock to prevent deadlock."""
        with self._reload_lock:
            self.gpu_layers = max(0, min(gpu_layers, self.layer_count))
            log.warning("Reloading PLaMo 3 Translate at %d GPU layers (%s)", self.gpu_layers, reason)
            record_event("mt_reload", gpu_layers=self.gpu_layers, reason=reason)
            update_metrics(workers={"mt": "reloading"})
            self._start()

    def request_gpu_relief(self) -> bool:
        """Ask for fewer GPU layers; applied between cues by the translation thread."""
        if self._closed or self.gpu_layers == 0:
            return False
        self._relief.set()
        return True

    def _chat(self, prompt: dict, *, timeout: float, temperature: float | None = None,
              max_tokens: int | None = None) -> tuple[str, str]:
        """Send a chat request and return (content, finish_reason)."""
        actual_max = max_tokens if max_tokens is not None else self.cfg.max_tokens
        body = {
            **prompt,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "top_k": self.cfg.top_k,
            "top_p": self.cfg.top_p,
            "repeat_penalty": self.cfg.repeat_penalty,
            "max_tokens": actual_max,
            "stream": False,
            "cache_prompt": True,
            "stop": PLAMO_STOP_TOKENS,
        }
        check_running()
        if self._closed:
            return "", ""

        started = time.monotonic()
        payload = self.server.chat(body, timeout=timeout)
        check_running()
        request_time = time.monotonic() - started

        choice = payload["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""

        usage = payload.get("usage")
        input_tokens = 0
        output_tokens = 0
        prompt_eval_time = 0.0
        generation_time = 0.0
        cache_hits = 0

        if isinstance(usage, dict):
            input_tokens = usage.get("prompt_tokens", 0) or 0
            output_tokens = usage.get("completion_tokens", 0) or 0
            prompt_eval_time = usage.get("prompt_eval_time", 0.0) or 0.0
            generation_time = usage.get("generation_time", 0.0) or 0.0
            cache_hits = usage.get("cache_hits", 0) or 0

        finish_reason = choice.get("finish_reason", "")
        if finish_reason == "length":
            log.warning("Truncated response (max_tokens=%d, output_tokens=%d)", actual_max, output_tokens)
            increment_counter("mt_truncated")
        elif finish_reason == "stop":
            increment_counter("mt_finish_stop")
        elif finish_reason:
            log.info("Unexpected finish_reason: %s (max_tokens=%d, output_tokens=%d)", finish_reason, actual_max, output_tokens)
            increment_counter("mt_finish_other")
        else:
            log.debug("No finish_reason returned (max_tokens=%d, output_tokens=%d)", actual_max, output_tokens)
            increment_counter("mt_finish_none")

        record_event(
            "mt_request",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
            request_time_s=round(request_time, 3),
            prompt_eval_time_s=round(prompt_eval_time, 4) if prompt_eval_time else None,
            generation_time_s=round(generation_time, 4) if generation_time else None,
            cache_hits=cache_hits,
            cache_enabled=True,
            max_tokens=actual_max,
            has_content=bool(content.strip()),
        )

        return content, finish_reason

    def set_position(self, start_s: float | None) -> None:
        self._position = start_s

    def _window(self, history: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Grow from a fixed first line, then restart at half size."""
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
        `all_names` — names found in the utterance OR context (used for missing_names check).
        `utterance_names` — names actually present in the current Japanese text (used for garbled_name check).

        Key: use source context for disambiguation so context-dependent nicknames
        (e.g. Miko) are recognized, but only require names that appear in the current
        utterance. Context-only names are hints, not mandatory.
        """
        source_context = f"{context}\n{ja}"
        # Use source context for disambiguation — this lets context-dependent
        # nicknames like Miko be recognized even when the utterance alone is ambiguous.
        all_names = self.glossary.name_terms(ja, context=source_context)[:MAX_TERMS]
        # Names actually spoken in the current utterance — use source context for
        # disambiguation so that context-dependent nicknames are still recognized.
        # This preserves exactly the nickname spoken without requiring names from
        # previous dialogue to appear in the current subtitle.
        utterance_names = self.glossary.name_terms(ja, context=source_context)
        found = self.glossary.find(ja, context=source_context)
        extra = [term for term in found if term not in all_names]
        exact = [t for t in extra if t[0] == ja or f" {t[0]} " in f" {ja} "]
        rest = [t for t in extra if t not in exact]
        terms = all_names + exact + rest
        return terms[:MAX_TERMS], all_names, utterance_names

    def prompt_for(self, ja: str, history: list[tuple[str, str]]) -> dict:
        context = self._context(history)
        terms, _, _ = self._terms_for(ja, context)
        return build_plamo_prompt(ja, terms, context)

    _MAX_RETRIES = 3
    _MAX_TRUNCATION_RETRIES = 1
    _MAX_ELAPSED_S = 30.0

    @staticmethod
    def _is_invalid(output: str) -> bool:
        """True when the output is empty or has a detectable issue."""
        return not output or english_output_issue(output) is not None

    @staticmethod
    def _normalize(output: str) -> str:
        """Normalize output for deterministic comparison."""
        return " ".join(output.split())

    def translate(self, ja: str, history: list[tuple[str, str]]) -> str:
        # Check relief and reload OUTSIDE the inference lock to prevent deadlock
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
                terms, names, utterance_names = self._terms_for(ja, context)
                prompt = build_plamo_prompt(ja, terms, context)

                started = time.monotonic()
                latency_budget = self.cfg.timeout_s * 0.7

                retries_left = self._MAX_RETRIES
                truncation_retries = 0
                # Track (normalized_output, request_key) pairs so we can distinguish
                # identical requests from genuinely different recovery attempts.
                previous_outputs: list[tuple[str, str]] = []

                def _repeated(output: str, request_key: str) -> bool:
                    """True if this invalid output was already generated by the same request."""
                    norm = self._normalize(output)
                    for prev_norm, prev_key in previous_outputs:
                        if prev_key == request_key and prev_norm == norm:
                            return True
                    return False

                def _record(output: str, request_key: str) -> None:
                    """Record the output AFTER checking for repetition."""
                    previous_outputs.append((self._normalize(output), request_key))

                def _elapsed_ok() -> bool:
                    return time.monotonic() - started < self._MAX_ELAPSED_S

                # Budget against the model's *input + output* context, not
                # max_tokens alone. Estimates are conservative; every retry
                # must independently fit its own requested output budget.
                simplified_prompt = build_plamo_prompt(ja, [], "", simplified=True)
                ja_tokens = int(len(ja) * 1.8)
                # Include each source/target spelling, not a fixed cost per term.
                glossary_tokens = sum(
                    7 + int((len(source) + len(target)) * 1.8)
                    for source, target in terms
                )
                context_tokens = int(len(context) * 1.8) if context else 0
                template_overhead = 50
                simplified_input_tokens = ja_tokens + template_overhead
                full_input_tokens = simplified_input_tokens + glossary_tokens + context_tokens

                # llama-server's -c is the total KV context shared by -np
                # parallel slots. Budget against a single request's slot size.
                slot_context_size = self.cfg.context_size // self.cfg.parallel_slots

                def _fits_context(input_tokens: int, output_tokens: int) -> bool:
                    # Allow 256 tokens for prompt/template estimation error.
                    return input_tokens + output_tokens + 256 <= slot_context_size

                if not _fits_context(simplified_input_tokens, self.cfg.max_tokens):
                    record_event("mt_rejected", reason="context_budget",
                                 estimated_input_tokens=simplified_input_tokens)
                    increment_counter("mt_rejected")
                    log.warning(
                        "PLaMO utterance exceeds context even without glossary "
                        "(estimated input=%d, max_output=%d, slot_context=%d)",
                        simplified_input_tokens, self.cfg.max_tokens, slot_context_size,
                    )
                    return ""

                skip_full_prompt = not _fits_context(full_input_tokens, self.cfg.max_tokens)

                try:
                    first_key = "simplified" if skip_full_prompt else "full"
                    first_prompt = simplified_prompt if skip_full_prompt else prompt
                    english, finish_reason = self._chat(first_prompt, timeout=self.cfg.timeout_s)
                    if skip_full_prompt:
                        record_event("mt_request", strategy="simplified_first",
                                     estimated_input_tokens=full_input_tokens)
                    _record(english, first_key)

                    if (finish_reason == "length" and retries_left > 0
                            and truncation_retries < self._MAX_TRUNCATION_RETRIES
                            and _elapsed_ok()):
                        retry_tokens = max(self.cfg.max_tokens * 2, 512)
                        if skip_full_prompt:
                            # Never retry the full prompt: we already know it
                            # exceeds the context budget. Increase the output
                            # allowance on the simplified prompt only if it fits.
                            if not _fits_context(simplified_input_tokens, retry_tokens):
                                record_event("mt_truncation_failed", reason="context_budget",
                                             max_tokens=retry_tokens)
                                increment_counter("mt_rejected")
                                return ""
                            retries_left -= 1
                            truncation_retries += 1
                            increment_counter("mt_retries")
                            record_event("mt_truncation_retry",
                                         strategy="simplified_increased_tokens",
                                         max_tokens=retry_tokens)
                            english, finish_reason = self._chat(
                                simplified_prompt, timeout=self.cfg.timeout_s, max_tokens=retry_tokens
                            )
                            _record(english, "simplified_high_tokens")
                        else:
                            # First prefer stripping context to free input space
                            # without increasing the output allowance.
                            retries_left -= 1
                            truncation_retries += 1
                            increment_counter("mt_retries")
                            record_event("mt_truncation_retry", strategy="simplified_prompt")
                            english, finish_reason = self._chat(
                                simplified_prompt, timeout=self.cfg.timeout_s
                            )
                            _record(english, "simplified_retry")

                            if finish_reason == "length" and retries_left > 0 and _elapsed_ok():
                                # Prefer the full prompt for names when a larger
                                # generation budget still fits, otherwise simplify.
                                if _fits_context(full_input_tokens, retry_tokens):
                                    retry_prompt = prompt
                                    strategy = "full_increased_tokens"
                                elif _fits_context(simplified_input_tokens, retry_tokens):
                                    retry_prompt = simplified_prompt
                                    strategy = "simplified_increased_tokens"
                                else:
                                    record_event("mt_truncation_failed", reason="context_budget",
                                                 max_tokens=retry_tokens)
                                    increment_counter("mt_rejected")
                                    return ""
                                retries_left -= 1
                                increment_counter("mt_retries")
                                record_event("mt_truncation_retry",
                                             strategy=strategy, max_tokens=retry_tokens)
                                english, finish_reason = self._chat(
                                    retry_prompt, timeout=self.cfg.timeout_s, max_tokens=retry_tokens
                                )
                                _record(english, "high_tokens")

                    if finish_reason == "length":
                        record_event("mt_truncation_failed", max_tokens=self.cfg.max_tokens)
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
                    # Rebuild prompt after recovery — server state may have changed
                    prompt = build_plamo_prompt(ja, terms, context)
                    if retries_left > 0:
                        retries_left -= 1
                        increment_counter("mt_retries")
                    # Recovery must preserve the original context-budget decision.
                    recovery_prompt = simplified_prompt if skip_full_prompt else prompt
                    english, _ = self._chat(recovery_prompt, timeout=self.cfg.timeout_s)
                    _record(english, "recovery")
                    english = clean_output(english)

                elapsed = time.monotonic() - started
                if elapsed > latency_budget:
                    log.warning("Translation latency %.2fs exceeded budget %.2fs", elapsed, latency_budget)
                    increment_counter("mt_latency_exceeded")

                # Retry 1: empty or invalid output — improved retry sequence:
                # 1a. Try with original glossary and lower temperature first (preserves names)
                # 1b. Fall back to simplified prompt only if 1a fails or doesn't fit
                if self._is_invalid(english) and retries_left > 0 and _elapsed_ok():
                    issue = english_output_issue(english)
                    if issue:
                        record_event("mt_rejected", reason=issue, retry=True)

                    # Step 1a: retry with glossary but lower temperature — preserves name accuracy
                    glossary_only_prompt = build_plamo_prompt(ja, terms, "")
                    glossary_only_input_tokens = simplified_input_tokens + glossary_tokens
                    glossary_only_fits = _fits_context(glossary_only_input_tokens, self.cfg.max_tokens)

                    if glossary_only_fits and (glossary_only_prompt != first_prompt or self.cfg.temperature != 0):
                        req_key = "glossary_low_temp"
                        if not _repeated(english, req_key):
                            retries_left -= 1
                            increment_counter("mt_retries")
                            english, _ = self._chat(glossary_only_prompt, timeout=self.cfg.timeout_s, temperature=0.2)
                            _record(english, req_key)
                            english = clean_output(english)

                    # Step 1b: if still invalid, try simplified prompt (no glossary/context)
                    # to give the model more output tokens
                    if self._is_invalid(english) and retries_left > 0 and _elapsed_ok():
                        req_key = "minimal"
                        if not _repeated(english, req_key):
                            retries_left -= 1
                            increment_counter("mt_retries")
                            english, _ = self._chat(simplified_prompt, timeout=self.cfg.timeout_s)
                            _record(english, req_key)
                            english = clean_output(english)

                # Retry 2: untranslated Japanese — try with lower temperature
                if retries_left > 0 and english and _elapsed_ok():
                    output_issue = english_output_issue(english)
                    if output_issue == "untranslated_japanese":
                        req_key = "low_temp"
                        if not _repeated(english, req_key):
                            record_event("mt_rejected", reason=output_issue, retry=True)
                            increment_counter("mt_rejected")
                            minimal = build_plamo_prompt(ja, [], "")
                            retries_left -= 1
                            increment_counter("mt_retries")
                            candidate, _ = self._chat(minimal, timeout=self.cfg.timeout_s, temperature=0.2)
                            _record(candidate, req_key)
                            cleaned = clean_output(candidate)
                            if cleaned and english_output_issue(cleaned) is None:
                                english = cleaned

                # Retry 3: missing or garbled names — try with glossary, zero temperature
                output_issue = english_output_issue(english)
                missing = missing_names(english, names)
                if not self._closed and retries_left > 0 and (output_issue or missing) and _elapsed_ok():
                    req_key = "name_fix"
                    if not _repeated(english, req_key):
                        if output_issue:
                            record_event("mt_rejected", reason=output_issue, retry=True)
                            increment_counter("mt_rejected")
                        fallback = build_plamo_prompt(ja, terms, "")
                        fallback_input_tokens = simplified_input_tokens + glossary_tokens
                        # A glossary-only retry must fit, even when the original
                        # full prompt was too large. Otherwise preserve the best
                        # existing result rather than making a doomed request.
                        fallback_fits = _fits_context(fallback_input_tokens, self.cfg.max_tokens)
                        candidates = [english]
                        if fallback_fits and (fallback != prompt or self.cfg.temperature != 0):
                            retries_left -= 1
                            increment_counter("mt_retries")
                            if missing:
                                record_event("mt_name_retry", names=missing)
                            candidate, _ = self._chat(fallback, timeout=self.cfg.timeout_s, temperature=0.0)
                            _record(candidate, req_key)
                            candidates.append(clean_output(candidate))
                        english = self._best(candidates, names, utterance_names=[en for _, en in utterance_names])

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
    def _best(candidates: list[str], names: list[tuple[str, str]],
              *, utterance_names: list[str] | None = None) -> str:
        """Prefer valid output, then correct names; an omitted name beats a wrong one.

        Only names explicitly mentioned in the current utterance can trigger
        garbled rejection. Context-only names are hints — missing them is fine.
        """
        scored = []
        for index, text in enumerate(candidates):
            missing = missing_names(text, names)
            present = [en for _, en in names if en not in missing]
            garbled = garbled_name(text, missing, present, utterance_names=utterance_names)
            scored.append((english_output_issue(text) is not None, garbled,
                           len(missing), index, text, missing))
        issue, garbled, _, _, text, missing = min(scored)
        # Only reject if there's a real output issue or a garbled utterance name.
        # Context-only name garbling should not discard a valid translation.
        if issue:
            increment_counter("mt_rejected")
            if missing:
                record_event("mt_name_rejected", names=missing)
            return ""
        if garbled and utterance_names is not None:
            # Garbled name in an utterance-mandated name — reject
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
            raise TranslatorUnavailable(f"PLaMo 3 Translate server could not restart: {exc}") from exc

    def interrupt(self) -> None:
        self._closed = True
        server = getattr(self, "server", None)
        if server is not None:
            server.close()

    def close(self) -> None:
        self.interrupt()
