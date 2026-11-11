"""PLaMo 3 Translate with an automatically managed local llama.cpp server."""

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
from ..context_memory import build_recent_context, reuse_recent_short_term
from ..glossary import Glossary
from ..subject_guard import (
    neutralize_unsupported_subjects,
    unsupported_person_reference,
    neutralize_unwarranted_gender,
    neutralize_unwarranted_first_person,
    neutralize_unwarranted_possessives,
)
from ..translation_guard import suspicious_translation
from ..malay_translation_guard import suspicious_malay_translation
from ..watame import correct_translation

log = logging.getLogger(__name__)


# PLaMo 3 GGUF ignores system messages; reference guidance belongs in
# chat_template_kwargs["context"], alongside the video title.
_NEUTRAL_SUBJECT_CONTEXT = (
    "Translate the CURRENT Japanese audio transcript faithfully into concise, natural English subtitles. "
    "Japanese often omits subjects, gender and possessors; do not invent them. "
    "When the source leaves the actor unspecified, prefer a natural impersonal or subjectless "
    "sentence (e.g. 運転が下手 -> 'The driving is bad'; 分からない -> 'Not sure'; "
    "事故だ -> 'There's been an accident'). "
    "Do NOT automatically assume the streamer is speaking as 'I', or that another player "
    "is 'he' or 'she'. If the Japanese clearly indicates a third person but not their gender, "
    "prefer the stated role/name; use 'they/them' only if a third-person referent is needed. "
    "Do not insert 'their', 'his', 'her', 'my', 'our', or 'your' before a noun "
    "unless the Japanese really identifies whose thing/person it is. "
    "For example, 最初のお客さん -> 'First customer!' (NOT 'Their first customer' "
    "or 'Your first customer'); お前の最初のお客さん -> 'Your first customer'. "
    "In full sentences without an owner, use 'the' or 'a' rather than an invented possessor. "
    "Preserve explicit or strongly implied first-person agency (私, 俺, 僕, ～と思う, ～したい), "
    "explicit gendered references (彼, 彼女), addresses and imperatives. "
    "Do not shift first person to third person when the source is first person. "
    "Never infer pronouns from the video title, streamer identity, or past English subtitles. "
    "Preserve negation, speaker intent, factual details, names and uncertainty; "
    "do not invent unheard dialogue."
)
_MALAY_CONTEXT = (
    "Terjemahkan transkrip audio bahasa Jepun SEMASA ke Bahasa Melayu standard Malaysia "
    "(ms), bukan bahasa Indonesia atau dialek tempatan. "
    "Gunakan gaya sari kata percakapan yang ringkas dan lancar. "
    "Kekalkan makna, penafian, nombor, nama Hololive, konteks permainan dan keraguan. "
    "Jangan teka penutur, jantina, subjek atau pemilik yang tidak dinyatakan dalam ayat Jepun. "
    "Jangan tambah maklumat yang tidak didengar. Jangan tambah awalan MS atau komen. "
    "Hasilkan terjemahan sahaja."
)
_STRICT_NEUTRAL_RETRY_CONTEXT = (
    "The previous translation assigned an unsupported speaker/person or gender. "
    "Rewrite ONLY the current Japanese as natural English without unsupported I/he/she/him/her, "
    "without inventing a new actor, and without changing viewpoint or polarity. "
    "Prefer subjectless statements or the stated name/role; use singular they only when a "
    "third-person human is supported by the Japanese. "
    "Avoid assumed possession: say 'First customer' or 'the first customer', "
    "not 'their first customer' or 'your first customer' without source ownership. "
    "Output only the translation."
)


# Scene-specific terms are supplied only when the video metadata identifies
# HoloGTA, and only if that term appears in the *current* Japanese utterance.
_HOLOGTA_TITLE = re.compile(r"ホロ(?:ライブ)?GTA|ホロスサントス|holo.?gta", re.I)
_HOLOGTA_TERMS = {
    "ホロライブGTA": "Hololive GTA",
    "ホロGTA": "HoloGTA",
    "ホロスサントス": "Holo Santos",
    "ミリモン": "Million Monsters",
    "副署長": "Deputy Police Chief",
    "署長": "Police Chief",
    "救急隊": "EMS",
    "ギャング": "gang",
    "警察": "police",
    "人質": "hostage",
    "パン屋": "bakery",
}


_HOLOGTA_TERMS_MS = {
    "ホロライブGTA": "Hololive GTA",
    "ホロGTA": "HoloGTA",
    "ホロスサントス": "Holo Santos",
    "ミリモン": "Million Monsters",
    "副署長": "timbalan ketua polis",
    "署長": "ketua polis",
    "救急隊": "pasukan paramedik",
    "ギャング": "geng",
    "警察": "polis",
    "人質": "tebusan",
    "パン屋": "kedai roti",
}


def _current_glossary_matches(
    ja: str, vod_title: str, glossary: Glossary, *, target_lang: str = "en",
) -> list[tuple[str, str]]:
    """Longest-first, non-overlapping current-source matches (never future guesses)."""
    scene_terms = _HOLOGTA_TERMS_MS if target_lang == "ms" else _HOLOGTA_TERMS
    scene = (
        [(source, target) for source, target in scene_terms.items() if source in ja]
        if _HOLOGTA_TITLE.search(vod_title) else []
    )
    candidates = sorted(scene + glossary.find(ja), key=lambda item: len(item[0]), reverse=True)
    remaining = ja
    selected: list[tuple[str, str]] = []
    for source, target in candidates:
        if source in remaining:
            selected.append((source, target))
            remaining = remaining.replace(source, "\0")
        if len(selected) >= 16:
            break
    return selected


def _translation_context(vod_title: str, *, neutral_retry: bool = False, target_lang: str = "en") -> str:
    context = _MALAY_CONTEXT if target_lang == "ms" else _NEUTRAL_SUBJECT_CONTEXT
    if neutral_retry and target_lang == "en":
        context += " " + _STRICT_NEUTRAL_RETRY_CONTEXT
    if vod_title:
        return f"Video title: {vod_title}\n{context}"
    return context


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
        if getattr(self, "memory_lines", 0) and self.cfg.target_lang == "en":
            reused = reuse_recent_short_term(ja, history)
            if reused is not None:
                log.debug("Reusing recent short term: JP=%r EN=%r", ja, reused)
                return reused

        def normalized(raw: str) -> str:
            translated = correct_translation(ja, _clean(raw))
            return neutralize_unsupported_subjects(ja, translated) if self.cfg.target_lang == "en" else translated

        english = normalized(self._chat(ja, history, timeout=self.cfg.timeout_s))
        names = _current_glossary_matches(ja, self.vod_title, self.glossary, target_lang=self.cfg.target_lang)
        pronoun_reason = (
            unsupported_person_reference(ja, english)
            if self.cfg.neutral_subjects and self.cfg.target_lang == "en" else None
        )
        guard = suspicious_malay_translation if self.cfg.target_lang == "ms" else suspicious_translation
        reason = pronoun_reason or guard(ja, english, names=names)
        if reason and self.cfg.retry_suspicious:
            log.info("PLaMo suspicious translation (%s): JP=%r EN=%r; retry without prior chat",
                     reason, ja[:150], english[:160])
            try:
                # Extra neutral instruction only when the first pass introduced
                # unsupported person/gender references. No new model is loaded.
                candidate = normalized(self._chat(
                    ja, [], timeout=min(self.cfg.timeout_s, self.cfg.retry_timeout_s),
                    neutral_retry=bool(pronoun_reason),
                ))
            except Exception as exc:
                log.warning("PLaMo validation retry unavailable (%s); using first result", exc)
            else:
                retry_reason = (
                    unsupported_person_reference(ja, candidate)
                    if self.cfg.neutral_subjects and self.cfg.target_lang == "en" else None
                ) or guard(ja, candidate, names=names)
                if retry_reason is None:
                    english = candidate
                    log.info("PLaMo validation retry accepted (trigger=%s)", reason)
                else:
                    log.warning(
                        "PLaMo retry still suspicious (%s); keeping first result with safe fallbacks",
                        retry_reason,
                    )
        if self.cfg.neutral_subjects and self.cfg.target_lang == "en":
            # Prefer owner-free noun phrases BEFORE gender fallback converts
            # "his/her customer" into an invented "their customer".
            english = neutralize_unwarranted_possessives(ja, english)
            # Restrict remaining fallbacks to grammar-safe forms. A source
            # that explicitly states a person/gender retains that information.
            english = neutralize_unwarranted_gender(ja, english)
            english = neutralize_unwarranted_first_person(ja, english)
        return english

    def _chat(
        self,
        ja: str,
        history: list[tuple[str, str]],
        timeout: float,
        *,
        neutral_retry: bool = False,
    ) -> str:
        messages: list[dict[str, str]] = []
        dialogue_history = history[-self.cfg.context_lines:] if self.cfg.context_lines > 0 else []
        for source, translation in dialogue_history:
            messages.append({"role": "user", "content": source})
            # Prior inferred pronouns must not contaminate new translations.
            messages.append({
                "role": "assistant",
                "content": neutralize_unsupported_subjects(source, translation) if self.cfg.target_lang == "en" else translation,
            })
        messages.append({"role": "user", "content": ja})

        memory = build_recent_context(
            history, self.glossary,
            context_lines=self.cfg.context_lines,
            memory_lines=getattr(self.cfg, "memory_lines", 24),
        )
        options: dict = {
            "source_lang": "ja",
            "target_lang": self.cfg.target_lang,
            "context": _translation_context(self.vod_title, neutral_retry=neutral_retry, target_lang=self.cfg.target_lang) + (
                "\n" + memory if memory else ""
            ),
        }
        if hits := _current_glossary_matches(ja, self.vod_title, self.glossary, target_lang=self.cfg.target_lang):
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
    if line.upper().startswith(("EN:", "ENGLISH:", "MS:", "BM:", "MALAY:", "BAHASA MELAYU:")):
        line = line.split(":", 1)[1].strip()
    return line.strip('"“”「」')
