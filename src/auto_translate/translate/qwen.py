"""Qwen3.5-4B Japanese-to-English translation via local llama.cpp.

No PLaMo-specific chat-template fields or prior English completions are sent
to the model. The glossary and Japanese transcript context remain available.
"""
import atexit
import json
import logging
import re
import shutil
import subprocess
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from ..config import TranslateConfig
from ..context_memory import reuse_recent_short_term
from ..glossary import Glossary
from ..process_cleanup import _create_kill_job, _close_job
from ..stream_context import ark_glossary_hits, is_ark_context, is_incomplete_fragment
from ..subject_guard import neutralize_unsupported_subjects
from ..watame import correct_translation
from .quality import (
    SKIP_TOKEN, is_obvious_nonverbal_japanese, is_skip_token,
    repair_high_confidence_meaning, translation_review_reason,
)

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You translate Japanese speech from VTuber and gaming streams into English subtitles. "
    "Return ONLY a clear, concise, natural English translation of the CURRENT Japanese line, "
    "never commentary, explanations, multiple options, or Japanese text. "
    "Preserve tone, humor, strong language, names, honorifics, numbers, negation, "
    "uncertainty, and meaning without adding or censoring information. "
    "Do not reverse praise, rejection, or other sentiment. "
    "Spoken names ending with -chan, -san or -sama retain those suffixes in English. "
    "Do not expand a nickname to a full name or invent a member name based on context. "
    "Japanese may omit the grammatical subject; avoid guessing who acted. "
    "The input is automatic speech recognition and may contain errors; "
    "use the provided terminology only when the current Japanese actually supports it. "
    "Previous lines and video title are background, not words spoken in the current line. "
    "For unfinished clauses translate the fragment without guessing the missing ending. "
    "Skip only if the WHOLE line is clearly nonverbal noise; then reply exactly [SKIP]. "
    "Otherwise always translate the meaningful Japanese, even if unclear."
)
_REVIEW_HINTS = {
    "positive_appraisal_reversed": "Japanese clearly praises something; do not turn it into criticism.",
    "loading_performance_reversed": "軽くなった means lighter or less demanding, NOT slower loading.",
    "rejection_reversed": "Repeated のん or ノン expresses rejection: no, no, no.",
    "possible_missing_negation": "Recheck explicit Japanese negation without inventing one.",
    "affirmative_ability_reversed": "The Japanese says someone CAN do it; retain that ability.",
    "unexpected_skip": "This is meaningful speech: provide its translation, not [SKIP].",
    "untranslated_japanese": "Translate the Japanese; do not repeat Japanese characters.",
    "romanized_syllable_loop": "Translate meaning rather than writing a meaningless sound loop.",
    "phonetic_transliteration": "Translate meaning instead of phonetic gibberish.",
    "empty_english": "Return an English translation rather than an empty response.",
}
# Only protect complete names that are particularly susceptible to unsupported expansion.
_FULL_NAMES = {
    "Ichijou Ririka": ("一条莉々華", "一条りりか", "いちじょうりりか", "莉々華", "りりか", "リリカ"),
    "Usada Pekora": ("兎田ぺこら", "うさだぺこら", "ぺこら", "ペコラ", "ぺこちゃん", "ペコちゃん"),
}
_SUGOI = {"すご", "すごっ", "すごい", "スゴ", "スゴッ", "スゴイ", "凄", "凄っ", "凄い"}


def _clean(text: str) -> str:
    """Discard leaked reasoning and routine response prefixes."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    line = lines[0]
    line = re.sub(r"^(?:EN(?:GLISH)?|TRANSLATION)\s*:\s*", "", line, flags=re.I)
    return line.strip(' "“”「」')


def _model_ids(base_url: str) -> list[str] | None:
    """None means no responding server; a list identifies its loaded models."""
    try:
        with urllib.request.urlopen(base_url + "/models", timeout=1.0) as response:
            document = json.load(response)
        return [str(item.get("id", "")) for item in document.get("data", [])]
    except urllib.error.HTTPError as exc:
        if exc.code == 503:
            return None
        raise
    except (OSError, urllib.error.URLError):
        return None


def _remove_unmentioned_full_names(japanese: str, english: str) -> str:
    for full_name, japanese_forms in _FULL_NAMES.items():
        if any(form in japanese for form in japanese_forms):
            continue
        pattern = re.compile(r"(?<![A-Za-z])" + re.escape(full_name) + r"(?![A-Za-z])", re.I)
        if pattern.search(english):
            log.debug("Discarding unsupported complete member name %s: JP=%r", full_name, japanese)
            english = pattern.sub("", english)
            english = re.sub(r"\s{2,}", " ", english).strip(" ,")
    return english


class QwenTranslator:
    """Only local Qwen3.5-4B; no external API or alternative translator."""

    def __init__(self, cfg: TranslateConfig, glossary: Glossary, *, vod_title: str = "") -> None:
        self.cfg = cfg
        self.glossary = glossary
        self.memory_lines = max(0, cfg.memory_lines)
        self.vod_title = " ".join(vod_title.split())[:200]
        self.base_url = cfg.qwen_base_url.rstrip("/")
        self.url = self.base_url + "/chat/completions"
        self.process: subprocess.Popen | None = None
        self._job = None  # Windows kill-on-close job for a server started by us
        self._server_log = None
        self._start_server()
        try:
            self._warmup()
        except Exception:
            self.close()
            raise

    def _start_server(self) -> None:
        current_models = _model_ids(self.base_url)
        if current_models is not None:
            # Do not adopt an existing process: its PID/ownership is unknown,
            # and silently reusing it would prevent guaranteed auto-cleanup.
            raise RuntimeError(
                f"A llama.cpp server is already listening at {self.base_url}: "
                f"{current_models!r}. Close that server before launching this "
                "session, or set [translate] qwen_base_url to another free port. "
                "This translator only starts and stops servers it owns."
            )

        executable = shutil.which("llama-server")
        if not executable:
            raise RuntimeError(
                "Qwen3.5-4B requires a recent CUDA-enabled llama-server in PATH. "
                "Update llama.cpp to a build supporting Qwen3.5 and restart CMD."
            )
        parsed = urllib.parse.urlparse(self.base_url)
        host, port = parsed.hostname or "127.0.0.1", parsed.port or 8001
        command = [
            executable, "-hf", self.cfg.qwen_hf_model,
            "-c", str(self.cfg.qwen_context),
            "--host", host, "--port", str(port),
            "--alias", self.cfg.qwen_model, "--jinja",
            "--no-mmproj",  # text-only translation; save ~0.7 GB GPU projector
            "--reasoning", "off",
            "-ngl", str(self.cfg.qwen_gpu_layers),
            "-np", "1",
        ]
        logdir = Path("logs")
        logdir.mkdir(exist_ok=True, parents=True)
        self._server_log = (logdir / "qwen-server.log").open("a", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL,
                stdout=self._server_log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            self._server_log.close()
            self._server_log = None
            raise
        try:
            # Job Object also closes llama-server when CMD is closed abruptly.
            # Fail closed if Windows cannot attach it rather than orphan a server.
            self._job = _create_kill_job(self.process)
        except Exception:
            self.close()
            raise
        atexit.register(self.close)
        log.info("Started managed Qwen3.5-4B server (PID %s)", self.process.pid)
        deadline = time.monotonic() + self.cfg.load_timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.close()
                raise RuntimeError(
                    "Qwen llama-server exited while loading. Check logs\\qwen-server.log "
                    "and update llama.cpp if Qwen3.5 is unsupported."
                )
            ids = _model_ids(self.base_url)
            if ids is not None:
                if self.cfg.qwen_model not in ids:
                    self.close()
                    raise RuntimeError(f"Unexpected model on Qwen port: {ids!r}")
                log.info("Qwen3.5-4B server ready at %s", self.base_url)
                return
            time.sleep(0.5)
        self.close()
        raise TimeoutError(
            f"Qwen server startup exceeded {self.cfg.load_timeout_s:.0f}s. "
            "Check logs\\qwen-server.log."
        )

    def _warmup(self) -> None:
        deadline = time.monotonic() + self.cfg.load_timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Qwen warm-up timed out; check logs\\qwen-server.log.")
            try:
                result = self._chat("こんにちは", [], timeout=min(20.0, remaining))
                if result.strip():
                    return
            except urllib.error.HTTPError as exc:
                if exc.code != 503:
                    raise
            except (TimeoutError, urllib.error.URLError):
                if self.process is not None and self.process.poll() is not None:
                    raise RuntimeError("Qwen server stopped during warm-up.")
            time.sleep(min(1.0, remaining))

    def close(self) -> None:
        """Stop our own llama-server and release the crash-safe job handle.

        Idempotent. An existing external server is never adopted or killed.
        """
        process, self.process = self.process, None
        job, self._job = getattr(self, "_job", None), None
        try:
            if process is not None and process.poll() is None:
                log.info("Stopping managed Qwen3.5-4B server (PID %s)", process.pid)
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    log.warning("Qwen server did not exit promptly; forcing shutdown")
                    process.kill()
                    process.wait(timeout=2)
        finally:
            # A close of the kill-on-job-close handle guarantees shutdown even
            # when terminate()/wait() failed. On other OSes this is a no-op.
            _close_job(job)
            if self._server_log is not None:
                self._server_log.close()
                self._server_log = None
            atexit.unregister(self.close)

    def _glossary_hits(self, ja: str, history: list[tuple[str, str]]) -> list[tuple[str, str]]:
        hits = self.glossary.find(ja)
        if is_ark_context(self.vod_title, ja, [jp for jp, _ in history[-8:]]):
            existing = {source for source, _ in hits}
            hits.extend((src, target) for src, target in ark_glossary_hits(ja) if src not in existing)
        return hits[:10]

    def translate(self, ja: str, history: list[tuple[str, str]]) -> str:
        short = unicodedata.normalize("NFKC", ja).strip().rstrip("!.。…").strip()
        if short in _SUGOI:
            return "Wow!"
        if self.memory_lines:
            cached = reuse_recent_short_term(ja, history)
            if cached is not None:
                return cached
        hits = self._glossary_hits(ja, history)
        if not hits and is_obvious_nonverbal_japanese(ja):
            return SKIP_TOKEN

        english = _clean(self._chat(ja, history, timeout=self.cfg.timeout_s))
        reason = translation_review_reason(ja, english)
        if reason and self.cfg.quality_recheck_enabled:
            now = time.monotonic()
            last = getattr(self, "_last_recheck_at", float("-inf"))
            if now - last >= self.cfg.quality_recheck_min_interval_s:
                self._last_recheck_at = now
                try:
                    revised = _clean(self._chat(
                        ja, history,
                        timeout=min(self.cfg.timeout_s, self.cfg.quality_recheck_timeout_s),
                        review_hint=reason,
                    ))
                except Exception as exc:
                    log.debug("Qwen selective recheck failed (%s): %s", reason, exc)
                else:
                    kept_names = all(
                        target.casefold() in revised.casefold()
                        for _, target in hits if target.casefold() in english.casefold()
                    )
                    if (revised and not is_skip_token(revised) and kept_names
                            and translation_review_reason(ja, revised) is None):
                        english = revised

        if not english or is_skip_token(english):
            log.warning("Qwen did not translate meaningful speech: %r", ja)
            return "[translation failed]"

        english = repair_high_confidence_meaning(ja, english)
        english = correct_translation(ja, english)
        english = _remove_unmentioned_full_names(ja, english)
        english = neutralize_unsupported_subjects(ja, english)
        return english if english else "[translation failed]"

    def _chat(self, ja: str, history: list[tuple[str, str]],
              timeout: float, review_hint: str = "") -> str:
        count = max(0, self.cfg.context_lines)
        relevant = history[-count:] if count else []
        recent = [source.strip()[:80] for source, _ in relevant if source.strip()]
        blocks = []
        if self.vod_title:
            blocks.append("Video context (NOT spoken words): " + self.vod_title)
        if recent:
            blocks.append("Recent Japanese (context only): " + " / ".join(recent)[-420:])
        hits = self._glossary_hits(ja, history)
        if hits:
            blocks.append(
                "Term mappings ONLY if present in the current Japanese:\n"
                + "\n".join(src + " => " + target for src, target in hits)
            )
        if is_ark_context(self.vod_title, ja, [jp for jp, _ in history[-8:]]):
            blocks.append("Game context: ARK dinosaurs and survival gameplay. ギガノト is Giganotosaurus.")
        if is_incomplete_fragment(ja):
            blocks.append("Current sentence is unfinished; translate only the heard fragment.")
        elif history and is_incomplete_fragment(history[-1][0]):
            blocks.append("Current line may continue an unfinished prior sentence; do not repeat prior translation.")
        if review_hint:
            blocks.append("RECHECK: " + _REVIEW_HINTS.get(review_hint, "Preserve the Japanese meaning exactly."))
        blocks.append("CURRENT JAPANESE (translate only this line):\n" + ja)
        body = {
            "model": self.cfg.qwen_model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": "\n\n".join(blocks)},
            ],
            "temperature": 0.1,
            "top_p": 0.8,
            "max_tokens": self.cfg.max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
        message = data["choices"][0]["message"]
        return str(message.get("content") or "")
