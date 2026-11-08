"""Index-Translate-2B Q4_K_M: sole offline Japanese-to-English translator.

Uses the Index team's one-message instTrans prompt and a managed llama.cpp
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
)

log = logging.getLogger(__name__)

# These are Index-Translate's documented instTrans constraints, not a second
# model's system prompt. Context is labeled as reference, never source.
_SOFT_STYLE = (
    "使用自然、简洁、口语化的英语游戏直播字幕风格，保留说话者的情绪、玩笑、"
    "俚语和粗口，不要写成生硬的正式书面语。忠实保留否定、疑问、动作方向、"
    "人物关系、语气和不确定性。日语省略主语时不要凭空指定说话者或动作主体。"
    "识别文本可能有误，不要擅自补齐没有说出的内容；未完的句子仍保持未完。"
)

def build_index_prompt(
    japanese: str,
    history: list[tuple[str, str]],
    glossary: Glossary,
    *,
    vod_title: str = "",
    context_lines: int = 4,
) -> str:
    """Official-style instTrans prompt; ONLY japanese is source text."""
    hits = glossary.find(japanese)[:8]
    hard = (
        "只翻译〖源文〗中的日语为英语；保留原有的否定、数字、明确的专名、"
        "游戏术语和语义。不要添加源文没有说出的人物、行为、对象或信息。"
    )
    if hits:
        hard += " 专名/术语对照（仅在源文确实出现时使用）: " + "、".join(
            src + "→" + target for src, target in hits
        )
    guidance = []
    if vod_title:
        guidance.append("视频背景（不是要翻译的文字）: " + vod_title[:140])
    recent = [ja.strip()[:70] for ja, _ in history[-max(0, context_lines):]
              if ja.strip()]
    if recent:
        guidance.append(
            "先前的日语对话（仅用于理解上下文，不可重复翻译）: "
            + " / ".join(recent)[-300:]
        )
    if is_incomplete_fragment(japanese):
        guidance.append("当前源文是不完整的片段，只翻译听到的部分。")
    note = _SOFT_STYLE
    if guidance:
        note += " " + " ".join(guidance)
    return (
        "请将以下日语游戏直播对话翻译成英语，并且严格遵守所有约束要求。\n\n"
        "〖源文〗\n" + japanese.strip() + "\n\n"
        "〖约束要求〗\n"
        "1. 〖硬性要求〗" + hard + "\n"
        "2. 〖注意〗" + note + "\n\n"
        "只输出译文，不要有任何额外说明。"
    )


def _clean(text: str) -> str:
    """Remove reasoning tags and response prefixes, but keep spoken content."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    text = text.strip().removeprefix("<think>").strip()
    text = re.sub(r"^(?:English|Translation|EN)\s*:\s*", "", text, flags=re.I)
    return " ".join(text.strip(' "“”').splitlines()).strip()


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

    def _chat(self, japanese: str, history: list[tuple[str, str]], *, timeout: float) -> str:
        prompt = build_index_prompt(
            japanese, history, self.glossary, vod_title=self.vod_title,
            context_lines=self.cfg.context_lines,
        )
        # The official Index client uses one USER message and greedy decoding.
        body = {
            "model": self.cfg.index_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
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
            result = json.load(response)
        return str(result["choices"][0]["message"].get("content") or "")

    def translate(self, ja: str, history: list[tuple[str, str]]) -> str:
        if not ja.strip():
            return SKIP_TOKEN
        if is_obvious_nonverbal_japanese(ja) and not self.glossary.find(ja):
            return SKIP_TOKEN
        if self.memory_lines:
            reused = reuse_recent_short_term(ja, history)
            if reused is not None:
                return reused

        english = _clean(self._chat(ja, history, timeout=self.cfg.timeout_s))
        if not english or is_skip_token(english):
            log.warning("Index returned no translation for meaningful speech: %r", ja)
            return "[translation failed]"

        # Shared conservative output safeguards, not a second translation model.
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
