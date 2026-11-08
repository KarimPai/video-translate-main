"""Index-Translate-2B Q4_K_M: sole offline Japanese-to-English translator.

Uses the Index team's plain one-message translation prompt and a managed llama.cpp
server. No Qwen3.5 chat engine, external provider, or translator fallback.
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
    is_incomplete_fragment,
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
    translation_review_reason,
)

log = logging.getLogger(__name__)

# For short streaming subtitles use Index's official *plain translation*
# format, rather than the multi-page instTrans constraints that the Q4 model
# sometimes copied into the English output. Keep only one brief style hint and
# glossary mappings actually heard in the current Japanese.
_PROMPT_PREFIX = (
    "请将以下日语游戏直播对话翻译成自然、简洁、口语化的英语字幕。"
    "保留原意、否定、称呼与语气，不要添加未说出的内容。"
    "被打断或未说完的句子只翻译已经说出的内容，不要补全。"
    "直接输出译文，不要解释。"
)

# The model occasionally continues into copies/translations of its instructions.
# These delimiters are not legitimate streaming subtitles. Catch them even if
# the server ignores stop sequences, or starts the response with a leaked header.
_LEAK_MARKERS = re.compile(
    r"〖|【(?:源文|原文|约束要求|約束要求|硬性要求|注意|"
    r"Source|Constraints?|Constraint Requirements|Hard Requirement)】"
    r"|(?:^|\n|\s{2,})(?:Constraint Requirements|Hard Requirements?|"
    r"Translation Requirements?)\s*[:：]?",
    flags=re.IGNORECASE,
)


def build_index_prompt(
    japanese: str,
    history: list[tuple[str, str]],
    glossary: Glossary,
    *,
    vod_title: str = "",
    context_lines: int = 2,
    review_reason: str = "",
) -> str:
    """Brief translation prompt with bounded non-authoritative Japanese context.

    Only the final Japanese line is a source to translate. Prior EN output is
    never injected, so previous model mistakes cannot become new evidence.
    """
    hits = glossary.find(japanese)[:6]
    parts = [_PROMPT_PREFIX]
    if hits:
        parts.append(
            "仅在原句出现时使用这些译名：" +
            "；".join(src + " = " + target for src, target in hits) + "。"
        )
    # Ohara/Oozora is a documented GTA name correction; never force
    # standalone 大空 to a person's name outside this explicit contrast.
    if re.search(r"大原.{0,12}(?:じゃねえ|じゃない|ではない|でなく).{0,12}大空", japanese):
        parts.append("这句在纠正人名：大原 = Ohara；大空 = Oozora。")
    count = min(2, max(0, context_lines))
    if count:
        previous = [
            ja.strip().replace("\n", " ")[:64]
            for ja, _ in history[-count:]
            if ja.strip()
        ]
        if previous:
            parts.append(
                "之前的日语对话（仅用于理解语境，不要翻译或重复）：" +
                " / ".join(previous)[-135:] + "。"
            )
    if is_incomplete_fragment(japanese):
        parts.append("当前日语是句子片段，保持未说完的结尾。")
    if review_reason:
        # A *single* evidence-driven retry. The model never receives the
        # previous English candidate, avoiding anchoring on a bad translation.
        parts.append(
            "请重新核对本句的否定、问句、专有名词与动作方向，"
            "不要凭空添加人物或事件。"
        )
    parts.append("请只翻译下一行的日语，不要翻译参考对话。")
    return "\n".join(parts) + "\n\n" + japanese.strip()


def _clean(text: str) -> str:
    """Strip thinking, translation prefixes, and leaked meta-instructions.

    Truncate at the *first* prompt marker. Never allow the instructions
    following it to be shown in MPV or printed as translation output.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    text = text.strip().removeprefix("<think>").strip()
    text = re.sub(r"^(?:English|Translation|EN)\s*:\s*", "", text, flags=re.I)
    leak = _LEAK_MARKERS.search(text)
    if leak:
        log.warning("Index generated prompt instructions; truncating subtitle")
        text = text[:leak.start()]
    return " ".join(text.strip(' \t\r\n\"“”').splitlines()).strip()



def _model_ids(base_url: str) -> list[str] | None:
    """None if no reachable, ready llama.cpp server on this port."""
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


class IndexTranslator:
    """One local Index-Translate-2B GGUF, managed for the entire session."""

    def __init__(self, cfg: TranslateConfig, glossary: Glossary, *, vod_title: str = "") -> None:
        self.cfg = cfg
        self.glossary = glossary
        self.memory_lines = max(0, cfg.memory_lines)
        self.vod_title = " ".join(vod_title.split())[:200]
        self.base_url = cfg.index_base_url.rstrip("/")
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
        models = _model_ids(self.base_url)
        if models is not None:
            # Never adopt or kill a process started outside this session.
            raise RuntimeError(
                "An existing llama.cpp server occupies " + self.base_url
                + " (" + repr(models) + "). Stop it or change index_base_url. "
                "This branch starts and stops only its own Index server."
            )
        executable = shutil.which("llama-server")
        if not executable:
            raise RuntimeError(
                "Index-Translate-2B requires a recent CUDA-enabled llama-server "
                "with Qwen3.5 GGUF support in PATH."
            )
        parsed = urllib.parse.urlparse(self.base_url)
        host, port = parsed.hostname or "127.0.0.1", parsed.port or 8002
        command = [
            executable, "-hf", self.cfg.index_hf_model,
            "-c", str(self.cfg.index_context),
            "--host", host, "--port", str(port),
            "--alias", self.cfg.index_model,
            "--jinja", "--reasoning", "off",
            "--no-mmproj",  # text only; don't allocate a vision projector
            "-ngl", str(self.cfg.index_gpu_layers),
            "-np", "1",
        ]
        logs = Path("logs")
        logs.mkdir(exist_ok=True, parents=True)
        self._server_log = (logs / "index-server.log").open("a", encoding="utf-8")
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

        log.info("Started managed Index-Translate-2B server (PID %s)", self.process.pid)
        deadline = time.monotonic() + self.cfg.load_timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.close()
                raise RuntimeError(
                    "Index llama-server exited during startup; inspect "
                    "logs/index-server.log and update llama.cpp if necessary."
                )
            available = _model_ids(self.base_url)
            if available is not None:
                if self.cfg.index_model not in available:
                    self.close()
                    raise RuntimeError("Unexpected model on Index server: " + repr(available))
                log.info("Index-Translate-2B server ready at %s", self.base_url)
                return
            time.sleep(0.5)
        self.close()
        raise TimeoutError(
            "Index server startup timed out after "
            + str(self.cfg.load_timeout_s) + " seconds. Check logs/index-server.log."
        )

    def _warmup(self) -> None:
        deadline = time.monotonic() + self.cfg.load_timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Index warm-up timed out; inspect logs/index-server.log.")
            try:
                result = self._chat("こんにちは", [], timeout=min(20, remaining))
                if result.strip():
                    return
            except urllib.error.HTTPError as exc:
                if exc.code != 503:
                    raise
            except (TimeoutError, urllib.error.URLError):
                if self.process is not None and self.process.poll() is not None:
                    raise RuntimeError("Index server stopped during warm-up.")
            time.sleep(min(1, remaining))

    def _chat(
        self, japanese: str, history: list[tuple[str, str]], *,
        timeout: float, review_reason: str = "",
    ) -> str:
        prompt = build_index_prompt(
            japanese, history, self.glossary, vod_title=self.vod_title,
            context_lines=self.cfg.context_lines, review_reason=review_reason,
        )
        # The official Index client uses one USER message and greedy decoding.
        body = {
            "model": self.cfg.index_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": self.cfg.max_tokens,
            # Prevent the known continuation into translated prompt metadata.
            "stop": ["〖", "【约束", "Constraint Requirements", "Hard Requirement"],
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        return str(result["choices"][0]["message"].get("content") or "")

    def _review_reason(self, japanese: str, english: str) -> str | None:
        """Only retry errors with a cheap, concrete failure indicator."""
        if not english or is_skip_token(english):
            return "empty_english"
        reason = translation_review_reason(japanese, english)
        if reason in {
            "untranslated_japanese", "romanized_syllable_loop",
            "phonetic_transliteration", "japanese_man_unit_wrong",
            "lost_sight_as_lost_way", "duration_extension_one_minute",
            "positive_appraisal_reversed", "affirmative_ability_reversed",
            "exclusive_necessity_reversed", "loading_performance_reversed",
            "rejection_reversed", "rhetorical_egunai_reversed",
            "possible_missing_negation",
        }:
            return reason
        # These source-side terms have unambiguous technical meanings in
        # GTA/Hololive roleplay. Guard only these particular anchors; avoid
        # forcing every nickname or ambiguous single-kanji glossary entry.
        protected = (
            ("オリ曲", r"\boriginal\b"),
            ("現行犯", r"\b(?:caught|red.hand|in the act|arrest(?:ed)?)\b"),
            ("指名手配", r"\bwanted\b"),
            ("張り込み", r"\b(?:stakeout|stake.out|stak(?:e|ing)\s+out|surveillance|watching)\b"),
            ("税金の使い込み", r"\b(?:misus|embezz|squander|misappropriat)\w*"),
            ("大空部長", r"\boozora\b"),
        )
        for term, pattern in protected:
            if term in japanese and not re.search(pattern, english, re.I):
                return "missing_priority_term:" + term
        return None

    def translate(self, ja: str, history: list[tuple[str, str]]) -> str:
        if not ja.strip():
            return SKIP_TOKEN
        if is_obvious_nonverbal_japanese(ja) and not self.glossary.find(ja):
            return SKIP_TOKEN
        if self.memory_lines:
            reused = reuse_recent_short_term(ja, history)
            if reused is not None:
                return reused

        # Short, exact glossary interjections do not need a language model.
        # The constraint is scoped to the entire utterance, never a substring.
        stripped = ja.strip().strip(" 　。、!?！？…")
        if stripped in {"オリ曲", "オリジナル曲"}:
            target = "Original song"
            if ja.rstrip().endswith(("!", "！")):
                target += "!"
            elif ja.rstrip().endswith(("?", "？")):
                target += "?"
            return target

        english = _clean(self._chat(ja, history, timeout=self.cfg.timeout_s))
        reason = self._review_reason(ja, english)
        if self.cfg.quality_retry_enabled and reason:
            log.info("Index quality review: %s (Japanese=%r)", reason, ja[:100])
            try:
                alternative = _clean(self._chat(
                    ja, history,
                    timeout=min(self.cfg.quality_retry_timeout_s, self.cfg.timeout_s),
                    review_reason=reason,
                ))
                # Never replace a usable subtitle with another suspect line.
                # A failed/empty first pass may be replaced by any clean,
                # meaningful second pass.
                if (alternative and not is_skip_token(alternative)
                    and (
                        self._review_reason(ja, alternative) is None
                        or not english
                        or is_skip_token(english)
                    )
                ):
                    english = alternative
                    log.info("Index quality review accepted corrected translation")
            except (TimeoutError, OSError, ValueError, KeyError, IndexError) as exc:
                log.warning("Index quality review unavailable; keeping first result: %s", exc)

        if not english or is_skip_token(english):
            log.warning("Index returned no translation for meaningful speech: %r", ja)
            return "[translation failed]"

        # Shared conservative output safeguards, not another model.
        english = repair_high_confidence_meaning(ja, english)
        recent_ja = [j for j, _ in history[-8:]]
        english = repair_valorant_translation(ja, english, recent_ja, False)
        english = repair_barotrauma_translation(ja, english)
        english = repair_ark_tier_translation(ja, english, self.vod_title, recent_ja)
        english = correct_translation(ja, english)
        english = neutralize_unsupported_subjects(ja, english)
        return english or "[translation failed]"

    def close(self) -> None:
        """Stop only the server we launched; release Windows job cleanup."""
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
