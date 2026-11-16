"""PLaMo 3 Translate with an automatically managed local llama.cpp server."""

import atexit
import httpx
from collections import Counter
import json
import logging
import re
import shutil
import socket
import subprocess
import threading
from ..windows_subprocess import owned_popen, check_running
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .base import (TranslatorUnavailable, TranslationCancelled,
                   TranslationDeadlineExceeded, TruncatedTranslation)
from ..config import TranslateConfig
from ..context_memory import build_recent_context, reuse_recent_short_term, split_completed_japanese_sentences
from ..glossary import Glossary
from ..name_hints import canonicalize_english_names, matsuri_matches, preserve_spoken_names
from ..name_context import is_miko_target
from ..subject_guard import (
    neutralize_unsupported_subjects,
    unsupported_person_reference,
    neutralize_unwarranted_gender,
    neutralize_unwarranted_first_person,
    neutralize_unwarranted_possessives,
)
from ..translation_guard import proper_name_mismatch, suspicious_translation
from ..watame import correct_translation

log = logging.getLogger(__name__)


# PLaMo 3 GGUF ignores system messages; reference guidance belongs in
# chat_template_kwargs["context"], alongside the video title.
_NEUTRAL_SUBJECT_CONTEXT = (
    "Translate the CURRENT Japanese audio transcript faithfully into concise, natural English subtitles. "
    "Japanese often omits subjects, gender and possessors; do not invent them. "
    "When the source leaves the actor unspecified, prefer a natural impersonal or subjectless "
    "sentence rather than an invented 'that person' or an assumed pronoun. "
    "For short gaming speech, omit the English subject whenever the meaning survives: "
    "もう出た -> 'Already left' (not 'He already left'); "
    "戻ってきた -> 'Back again' (not 'She came back'). "
    "Examples: 運転が下手 -> 'The driving is bad'; 分からない -> 'Not sure'; "
    "事故だ -> 'There's been an accident'. "
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
    "retain quantities and units. Translate the supplied spelling; do not repair ASR or complete missing dialogue."
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
    "無線機": "radio",
    "パトカー": "patrol car",
    "救急車": "ambulance",
    "防弾チョッキ": "bulletproof vest",
    "支給品": "issued equipment",
    "人質": "hostage",
    "パン屋": "bakery",
}


def _needs_fragment_context(ja: str) -> bool:
    """Only short, demonstrably context-dependent Japanese speech fragments."""
    current = ja.strip().rstrip("。！？!? ")
    if not current or len(current) > 26:
        return False
    return bool(
        re.match(r"^(?:それ|これ|あれ|その|あの|だから|でも|で、|じゃあ)", current)
        or re.search(r"(?:けど|けれど|ので|だから|って|とか|から|なのに)$", current)
    )


# Limit the extra instruction to the ambiguous word, excluding compounds
# such as 草原 and sound-alikes such as くさい or 面倒くさい.
_KUSA = re.compile(r"(?<![一-鿿])草(?![一-鿿])|(?:くさ|クサ)(?=$|[。、！？!?\s]|[をがはにもの])")


def _current_glossary_matches(ja: str, vod_title: str, glossary: Glossary, previous_ja: str = "", *,
                              name_context_ja: str | None = None) -> list[tuple[str, str]]:
    """Longest-first, non-overlapping current-source matches (never future guesses)."""
    scene = (
        [(source, target) for source, target in _HOLOGTA_TERMS.items() if source in ja]
        if _HOLOGTA_TITLE.search(vod_title) else []
    )
    terms = glossary.find(ja, previous_ja=previous_ja, video_context=vod_title)
    if name_context_ja is not None and name_context_ja != previous_ja:
        # Extend Miko identity evidence without broadening the established
        # immediately-previous-cue policy for Marine or Matsuri role names.
        terms = [pair for pair in terms if not is_miko_target(pair[1])] + [
            pair for pair in glossary.find(ja, previous_ja=name_context_ja, video_context=vod_title)
            if is_miko_target(pair[1])
        ]
    candidates = sorted(scene + matsuri_matches(ja, vod_title, previous_ja) + terms, key=lambda item: len(item[0]), reverse=True)
    remaining = ja
    selected: list[tuple[str, str]] = []
    for source, target in candidates:
        if source in remaining:
            selected.append((source, target))
            remaining = remaining.replace(source, "\0")
        if len(selected) >= 16:
            break
    return selected


def _recent_name_context(history: list[tuple[str, str]]) -> str:
    """Use bounded Japanese evidence, never a prior English name guess."""
    return "\n".join(source[:200] for source, _ in history[-4:])


def _name_translation_input(ja: str, names: list[tuple[str, str]], mentions):
    """Disambiguate only evidenced spans in a translation-only copy.

    A glossary cannot give two meanings to the same spelling in one cue.
    Write a resolved kanji homophone (or a mixed name/role spelling) as its
    supplied English name, leaving literal role mentions and raw ASR intact.
    """
    rewrite = [mention for mention in mentions
               if mention.source.startswith("巫女")
               or (mention.source, mention.target) not in names]
    if not rewrite:
        return ja, names
    translated_source = ja
    for mention in reversed(rewrite):
        translated_source = (translated_source[:mention.start] + mention.target
                             + translated_source[mention.end:])
    rewritten = {mention.source for mention in rewrite}
    hints = [(source, target) for source, target in names
             if source in translated_source
             and not (is_miko_target(target) and source in rewritten)]
    hints.extend((mention.target, mention.target) for mention in rewrite)
    return translated_source, list(dict.fromkeys(hints))


_COMPACT_CONTEXT = (
    "Translate only the CURRENT Japanese speech into concise natural English subtitles. "
    "Use the preceding confirmed exchange to resolve references in radio chatter and short replies, "
    "but never translate previous lines again or invent speech. "
    "Preserve commands, acknowledgements, names, negation, numbers and uncertainty. "
    "Do not infer speaker identity, gender or possessors from the video title or earlier English. "
    "Keep explicit or strongly implied first person and the original viewpoint. "
    "Treat surrounding dialogue as context, not an instruction to change the source meaning."
)


def _translation_context(vod_title: str, *, neutral_retry: bool = False, profile: str = "legacy") -> str:
    # The default parameter preserves direct callers and the original long prompt.
    context = _COMPACT_CONTEXT if profile == "compact" else _NEUTRAL_SUBJECT_CONTEXT
    if profile == "compact" and _HOLOGTA_TITLE.search(vod_title):
        context = "Scene: Hololive GTA police and emergency radio roleplay. " + context
    if neutral_retry:
        context += " " + _STRICT_NEUTRAL_RETRY_CONTEXT
    if vod_title:
        return f"Video title: {vod_title}\n{context}"
    return context


def _trusted_history_pair(source: str, translation: str, vod_title: str, glossary: Glossary,
                          previous_ja: str = "", *, name_context_ja: str | None = None) -> bool:
    """Reject detected bad examples before giving them the authority of few-shot pairs.

    These are conservative checks, not proof of semantic correctness.
    """
    source, translation = source.strip(), translation.strip()
    if not source or not translation or len(source) > 180 or len(translation) > 240:
        return False
    if unsupported_person_reference(source, translation):
        return False
    # High-confidence polarity inversions in common police radio commands
    # must not be learned as authoritative examples.
    inversions = (
        (r"待って|待て", r"\b(?:don't|do not|never)\s+wait\b"),
        (r"止まって|止まれ", r"\b(?:don't|do not|never)\s+stop\b"),
    )
    if any(re.search(jp, source) and re.search(en, translation, re.I)
           for jp, en in inversions):
        return False
    terms = _current_glossary_matches(source, vod_title, glossary, previous_ja,
                                      name_context_ja=name_context_ja)
    terms.extend((mention.source, mention.target) for mention in glossary.miko_mentions(
        source, video_context=vod_title,
        previous_ja=previous_ja if name_context_ja is None else name_context_ja))
    return suspicious_translation(source, translation, names=terms) is None


class ExistingServerUnavailable(RuntimeError):
    """An existing endpoint failed before this run launched a model."""


def _loopback_port_available(url: str) -> bool:
    parsed = urllib.parse.urlsplit(url)
    if parsed.hostname not in ("127.0.0.1", "::1"):
        return False
    family = socket.AF_INET6 if parsed.hostname == "::1" else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            probe.bind((parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)))
        except OSError:
            return False
        return True


class PlamoTranslator:
    """Japanese-to-English PLaMo translator; inference stays on the local machine."""

    def __init__(self, cfg: TranslateConfig, glossary: Glossary, *, vod_title: str = "") -> None:
        self.cfg = cfg
        self.memory_lines = max(0, getattr(cfg, "memory_lines", 24))
        self.glossary = glossary
        self.validation_stats: Counter[str] = Counter()
        self.vod_title = " ".join(vod_title.split())[:200]
        self.url = cfg.plamo_base_url.rstrip("/") + "/chat/completions"
        self.base_url = cfg.plamo_base_url.rstrip("/")
        self.process: subprocess.Popen | None = None
        self._server_log = None
        self._close_lock = threading.Lock()
        self._closed = False
        self._restart_used = False
        self.client = httpx.Client(trust_env=False)
        try:
            self._start_server()
            # Warm up before ingest starts so first subtitles do not pay load cost.
            log.info("Warming up PLaMo translation")
            self._warmup()
        except Exception:
            self.close()
            raise

    def _start_server(self) -> None:
        """Reuse a responding local server; otherwise launch and own one."""
        deadline = self._operation_deadline(self.cfg.load_timeout_s)
        log.info("Checking PLaMo endpoint %s", self.base_url)
        next_progress = 0.0
        while True:
            self._check_cue()
            check_running()
            state = _server_state(self.base_url, self.cfg.plamo_model)
            if state in ("ready", "absent"):
                break
            if state == "unresponsive" and _loopback_port_available(self.base_url):
                log.info("PLaMo probe timed out but the local port is free; launching llama-server")
                break
            if time.monotonic() >= next_progress:
                log.info("Waiting for existing PLaMo endpoint %s (%s)", self.base_url, state)
                next_progress = time.monotonic() + 10
            if time.monotonic() >= deadline:
                raise ExistingServerUnavailable(f"Existing PLaMo endpoint {self.base_url} is {state} after {self.cfg.load_timeout_s:g}s")
            time.sleep(0.5)
        if state == "ready":
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
        log.info("Resolving PLaMo model %s for llama-server", self.cfg.plamo_hf_model)
        from ..plamo_model import model_arguments
        model_args = model_arguments(self.cfg.plamo_hf_model)
        command = [
            executable,
            *model_args,
            "-c", str(self.cfg.plamo_context),
            "--host", host,
            "--port", str(port),
            "--alias", self.cfg.plamo_model,
            "--jinja",
            "--no-prefill-assistant",
            "-ngl", str(self.cfg.plamo_gpu_layers),
            "-np", "1",  # translation is serialized; avoid 4 parallel KV slots
        ]

        if self.cfg.plamo_gpu_layers == 0:
            command.extend(["--device", "none", "--no-kv-offload"])

        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        with self._close_lock:
            if self._closed:
                raise TranslatorUnavailable("PLaMo session closed")
            self._server_log = (log_dir / "plamo-server.log").open(
                "a", encoding="utf-8"
            )
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            try:
                self.process = owned_popen(
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

            process = self.process
            atexit.register(self.close)
        log.info(
            "Started local PLaMo server (PID %s); loading model...",
            process.pid,
        )

        deadline = self._operation_deadline(self.cfg.load_timeout_s)
        while time.monotonic() < deadline:
            self._check_cue()
            check_running()
            if process.poll() is not None:
                self.close()
                raise RuntimeError(
                    "llama-server exited during startup. See logs\\plamo-server.log."
                )
            if _server_state(self.base_url, self.cfg.plamo_model) == "ready":
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
            check_running()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"PLaMo did not accept a warm-up request within "
                    f"{self.cfg.load_timeout_s:.0f}s. See logs\\plamo-server.log."
                )
            try:
                self._chat("こんにちは", [], timeout=min(10.0, remaining))
                return
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 503:
                    raise
                log.info("PLaMo model is still loading; retrying warm-up...")
            time.sleep(min(1.0, max(0.0, remaining)))

    def _count_validation(self, reason: str) -> None:
        stats = getattr(self, "validation_stats", None)
        if stats is not None:
            stats[reason] += 1

    def close(self) -> None:
        """Stop only the llama-server process started by this translator."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        stats = getattr(self, "validation_stats", None)
        if stats:
            log.info("PLaMo translation validation summary: %s", dict(stats))
            stats.clear()
        process = self.process
        self.process = None
        try:
            if process is not None and process.poll() is None:
                log.info("Stopping PLaMo server (PID %s)...", process.pid)
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        finally:
            if self._server_log is not None:
                self._server_log.close()
                self._server_log = None
            self.client.close()
            atexit.unregister(self.close)

    def _operation_deadline(self, timeout_s: float) -> float:
        return min(time.monotonic() + timeout_s,
                   getattr(self, "_cue_deadline", float("inf")))

    def _check_cue(self) -> None:
        cancelled = getattr(self, "_cue_cancelled", None)
        if cancelled is not None and cancelled():
            raise TranslationCancelled("Subtitle superseded or player stopped")
        deadline = getattr(self, "_cue_deadline", float("inf"))
        if deadline != float("inf") and time.monotonic() >= deadline:
            raise TranslationDeadlineExceeded("PLaMo cue translation time budget exhausted")
        heartbeat = getattr(self, "_cue_heartbeat", None)
        if heartbeat is not None:
            heartbeat()

    def _request_timeout(self, timeout_s: float) -> float:
        self._check_cue()
        remaining = getattr(self, "_cue_deadline", float("inf")) - time.monotonic()
        if remaining <= 0:
            raise TranslationDeadlineExceeded("PLaMo cue translation time budget exhausted")
        return min(timeout_s, remaining)

    def translate(self, ja: str, history: list[tuple[str, str]]) -> str:
        return self.translate_cancellable(ja, history)

    def translate_cancellable(self, ja: str, history: list[tuple[str, str]], *,
                              cancelled=None, heartbeat=None) -> str:
        """One shared deadline for sentence, transport and validation retries."""
        self._cue_deadline = time.monotonic() + getattr(self.cfg, "cue_timeout_s", 75.0)
        self._cue_cancelled = cancelled
        self._cue_heartbeat = heartbeat
        self.last_name_resolutions = []
        self.last_translation_flags = []
        try:
            self._check_cue()
            result = self._translate_with_recovery(ja, history)
            self._check_cue()
            return result
        finally:
            self._cue_deadline = float("inf")
            self._cue_cancelled = None
            self._cue_heartbeat = None

    def _translate_with_recovery(self, ja: str, history: list[tuple[str, str]]) -> str:
        resolution_count = len(self.last_name_resolutions)
        previous_flags = list(self.last_translation_flags)
        try:
            return self._translate_cue(ja, history)
        except httpx.TransportError as exc:
            with self._close_lock:
                if self._closed or self.process is None or self.process.poll() is None:
                    raise
                if self._restart_used:
                    raise TranslatorUnavailable("PLaMo exited again after recovery") from exc
                self._restart_used = True
                self.process = None
                if self._server_log is not None:
                    self._server_log.close()
                    self._server_log = None
            log.warning("Owned PLaMo server exited. Restarting and retrying the subtitle.")
            # The complete cue is retried. Diagnostics from the failed
            # attempt must not make the recovered result look unresolved.
            del self.last_name_resolutions[resolution_count:]
            self.last_translation_flags = previous_flags
            try:
                self._check_cue()
                self._start_server()
                self._check_cue()
                return self._translate_cue(ja, history)
            except (TranslationCancelled, TranslationDeadlineExceeded):
                raise
            except Exception as recovery_error:
                raise TranslatorUnavailable(f"PLaMo recovery failed: {recovery_error}") from recovery_error

    def _translate_cue(self, ja: str, history: list[tuple[str, str]]) -> str:
        """Translate complete exchanges without changing media timestamps.

        The segmented path runs only on clear punctuation boundaries; every
        finalized sentence informs the next one. If a pass fails, preserve the
        original whole-utterance behavior instead of emitting half a cue.
        """
        self._check_cue()
        segments = (split_completed_japanese_sentences(ja)
                    if getattr(self.cfg, "sentence_context", False) else [ja])
        if len(segments) < 2:
            return self._translate_one(ja, history)
        recent = list(history)
        english = []
        resolution_count = len(self.last_name_resolutions)
        previous_flags = list(self.last_translation_flags)
        try:
            for segment in segments:
                self._check_cue()
                translated = self._translate_one(segment, recent)
                if not translated.strip() or suspicious_translation(segment, translated):
                    raise ValueError("unreliable sentence translation")
                english.append(translated.strip())
                recent.append((segment, translated))
        except (httpx.TransportError, TranslationCancelled,
                TranslationDeadlineExceeded, TruncatedTranslation):
            raise
        except Exception as exc:
            log.warning("Sentence context fallback to whole utterance: %s", exc)
            del self.last_name_resolutions[resolution_count:]
            self.last_translation_flags = previous_flags
            return self._translate_one(ja, history)
        return " ".join(english)

    def _translate_one(self, ja: str, history: list[tuple[str, str]]) -> str:
        self._check_cue()
        previous = history[-1][0] if history else ""
        name_context = _recent_name_context(history)
        names = _current_glossary_matches(ja, self.vod_title, self.glossary, previous,
                                          name_context_ja=name_context)
        mentions = self.glossary.miko_mentions(
            ja, video_context=self.vod_title, previous_ja=name_context)
        translation_ja, translation_names = _name_translation_input(ja, names, mentions)
        # Retain original source spans for validation, including names that
        # cannot safely be represented as a cue-wide glossary entry.
        names = list(dict.fromkeys(names + [(m.source, m.target) for m in mentions]))
        for mention in mentions:
            self.last_name_resolutions.append(dict(
                japanese=ja, start=mention.start, end=mention.end,
                source=mention.source, target=mention.target, evidence=mention.evidence,
                translation_source=translation_ja if translation_ja != ja else None,
            ))

        def normalize_names(english: str) -> str:
            if any(target in ("Haachama", "Laplus") for _, target in names):
                english = canonicalize_english_names(ja, english, self.vod_title)
            return preserve_spoken_names(ja, english, names, miko_mentions=mentions)

        if getattr(self, "memory_lines", 0) and not _KUSA.search(ja) and "マツリ" not in ja:
            reused = reuse_recent_short_term(ja, history)
            if reused is not None:
                # Cached subtitles must obey current short-name policy too.
                reused = normalize_names(reused)
                if not proper_name_mismatch(ja, reused, names):
                    log.debug("Reusing recent short term: JP=%r EN=%r", ja, reused)
                    return reused

        def normalized(raw: str) -> str:
            translated = _clean(raw)
            if _KUSA.search(ja) and re.search(r"\blol\b", translated, re.I):
                literal = re.search(r"草[をがはもに]*(?:刈|食|抜|生やさ|集め)|(?:羊|牛|馬).*食|草むしり", ja)
                if not literal and ja.strip("。！？!? ") in ("草", "くさ", "クサ") and previous:
                    literal = re.search(r"庭|雑草|草刈|草食|(?:羊|牛|馬).*食", previous)
                translated = re.sub(r"\blol\b", "grass" if literal else "kusa", translated, flags=re.I)
            if any(target in ("Watame", "Watamee", "Tsunomaki Watame") for _, target in names):
                translated = correct_translation(ja, translated)
            translated = normalize_names(translated)
            return (neutralize_unsupported_subjects(ja, translated)
                    if self.cfg.neutral_subjects else translated)

        try:
            raw = self._chat(translation_ja, history, timeout=self.cfg.timeout_s, names=translation_names)
        except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                raise
            log.warning("Retrying transient PLaMo request failure: %s", exc)
            self._check_cue()
            raw = self._chat(translation_ja, history, timeout=min(self.cfg.timeout_s, self.cfg.retry_timeout_s), names=translation_names)
        english = normalized(raw)
        pronoun_reason = (
            unsupported_person_reference(ja, english)
            if self.cfg.neutral_subjects else None
        )
        reason = pronoun_reason or suspicious_translation(ja, english, names=names)
        if reason:
            self._count_validation("flag:" + reason)
        if reason and self.cfg.retry_suspicious:
            log.info("PLaMo suspicious translation (%s): JP=%r EN=%r; retry without prior chat",
                     reason, ja[:150], english[:160])
            try:
                # Extra neutral instruction only when the first pass introduced
                # unsupported person/gender references. No new model is loaded.
                self._check_cue()
                candidate = normalized(self._chat(
                    translation_ja, [], timeout=min(self.cfg.timeout_s, self.cfg.retry_timeout_s),
                    neutral_retry=bool(pronoun_reason), names=translation_names,
                ))
            except Exception as exc:
                if isinstance(exc, (TranslationCancelled, TranslationDeadlineExceeded)):
                    raise
                self._count_validation("retry_unavailable")
                log.warning("PLaMo validation retry unavailable (%s); using first result", exc)
            else:
                retry_reason = (
                    unsupported_person_reference(ja, candidate)
                    if self.cfg.neutral_subjects else None
                ) or suspicious_translation(ja, candidate, names=names)
                if retry_reason is None:
                    english = candidate
                    self._count_validation("retry_accepted")
                    log.info("PLaMo validation retry accepted (trigger=%s)", reason)
                else:
                    self._count_validation("retry_rejected:" + retry_reason)
                    log.warning(
                        "PLaMo retry still suspicious (%s); keeping first result with safe fallbacks",
                        retry_reason,
                    )
        if self.cfg.neutral_subjects:
            # Prefer owner-free noun phrases BEFORE gender fallback converts
            # "his/her customer" into an invented "their customer".
            english = neutralize_unwarranted_possessives(ja, english)
            # Restrict remaining fallbacks to grammar-safe forms. A source
            # that explicitly states a person/gender retains that information.
            english = neutralize_unwarranted_gender(ja, english)
            english = neutralize_unwarranted_first_person(ja, english)
        if proper_name_mismatch(ja, english, names):
            if "proper_name_mismatch" not in self.last_translation_flags:
                self.last_translation_flags.append("proper_name_mismatch")
        return english

    def _chat(
        self,
        ja: str,
        history: list[tuple[str, str]],
        timeout: float,
        *,
        names=None,
        neutral_retry: bool = False,
    ) -> str:
        messages: list[dict[str, str]] = []
        dialogue_history = history[-self.cfg.context_lines:] if self.cfg.context_lines > 0 else []
        compact = getattr(self.cfg, "context_profile", "legacy") == "compact"
        rejected_sources: list[str] = []
        history_offset = len(history) - len(dialogue_history)
        for index, (source, translation) in enumerate(dialogue_history):
            # The compact profile borrows OverlaySubs' paired PFML examples,
            # but only after basic quality and unsupported-pronoun checks.
            # Legacy mode preserves the old Japanese-only neutral prompt.
            prior = history[:history_offset + index]
            trusted = (not compact or _trusted_history_pair(
                source, translation, self.vod_title, self.glossary,
                prior[-1][0] if prior else "", name_context_ja=_recent_name_context(prior)))
            if trusted and (compact or not self.cfg.neutral_subjects):
                messages.append({"role": "user", "content": source})
                messages.append({
                    "role": "assistant",
                    "content": (neutralize_unsupported_subjects(source, translation)
                                if self.cfg.neutral_subjects else translation),
                })
            elif source.strip():
                rejected_sources.append(source.strip()[:100])
        messages.append({"role": "user", "content": ja})

        memory = build_recent_context(
            history, self.glossary,
            context_lines=self.cfg.context_lines,
            memory_lines=getattr(self.cfg, "memory_lines", 24),
        )
        source_hint = ""
        if self.cfg.neutral_subjects and dialogue_history:
            # Don't echo source text that's already present in trusted pairs.
            previous = (rejected_sources[-2:] if compact else
                        [source.strip()[:100] for source, _ in dialogue_history[-2:] if source.strip()])
            if previous:
                source_hint = ("\nRecent Japanese (context only, not speech to translate; "
                               "never infer speaker gender from it): " + " / ".join(previous))
        fragment_hint = ""
        if (self.cfg.fragment_context and dialogue_history
                and _needs_fragment_context(ja)):
            # Past JP is evidence for a short omitted referent; past English
            # may be an incorrect guess. Never translate the prior line again.
            prior_ja = dialogue_history[-1][0].strip()[:160]
            fragment_hint = (
                "\nImmediately preceding Japanese (reference only, NOT part of the "
                "current subtitle): " + prior_ja
            )
        options: dict = {
            "source_lang": "ja",
            "target_lang": "en",
            "context": _translation_context(
                self.vod_title, neutral_retry=neutral_retry,
                profile=getattr(self.cfg, "context_profile", "legacy")) + (
                "\n" + memory if memory else ""
            ) + source_hint + fragment_hint,
        }
        if _KUSA.search(ja):
            options["context"] += (
                " Resolve kusa from the current Japanese and preceding dialogue. "
                "Use grass or weeds for vegetation, mowing, gathering or feeding animals. "
                "For amusement retain the spoken slang kusa. Do not render kusa as lol."
            )
        hits = names if names is not None else _current_glossary_matches(
            ja, self.vod_title, self.glossary, history[-1][0] if history else "",
            name_context_ja=_recent_name_context(history))
        if any(is_miko_target(target) for _, target in hits):
            options["context"] += (
                " The supplied Miko/Mikochi glossary entries are person names for Sakura Miko. "
                "Preserve the supplied name and spoken honorific; do not translate it as shrine maiden. "
                "A separate literal occupation or role 巫女 still means shrine maiden."
            )
        if any(self.glossary.nicknames.get(source) == target for source, target in hits):
            options["context"] += (" Keep each spoken nickname at its supplied length, "
                                   "including spoken honorifics. Do not expand it to a stage name.")
        if hits:
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
        for attempt in range(2):
            response = self.client.post(self.url, json=body,
                                        timeout=self._request_timeout(timeout))
            response.raise_for_status()
            self._check_cue()
            choice = response.json()["choices"][0]
            if choice.get("finish_reason") != "length":
                content = choice["message"]["content"]
                if not isinstance(content, str):
                    raise ValueError("PLaMo returned non-text translation content")
                return content
            self._count_validation("token_limit")
            if attempt:
                raise TruncatedTranslation("PLaMo translation still truncated after one token-limit retry")
            # Regenerate the COMPLETE cue; concatenating a partial response
            # can duplicate or contradict earlier words. Stay inside both
            # the original cue deadline and model context budget.
            body["max_tokens"] = max(self.cfg.max_tokens, min(self.cfg.max_tokens * 2, 768))
            log.warning("PLaMo hit token limit; retrying complete cue with %d tokens", body["max_tokens"])


_LOCAL = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _server_state(base_url: str, alias: str) -> str:
    try:
        with _LOCAL.open(base_url + "/models", timeout=3.0) as response:
            models = json.load(response)["data"]
    except urllib.error.HTTPError as exc:
        if exc.code == 503:
            return "loading"
        raise RuntimeError(f"Unexpected model service at {base_url}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, ConnectionRefusedError) or (
            isinstance(reason, str) and ("connection refused" in reason.lower() or "10061" in reason)
        ):
            return "absent"
        if isinstance(reason, TimeoutError) or "timed out" in str(reason).lower():
            return "unresponsive"
        raise RuntimeError(f"Cannot probe PLaMo at {base_url}: {reason}") from exc
    except TimeoutError:
        return "unresponsive"
    if not any(model.get("id") == alias for model in models):
        raise RuntimeError(f"Service at {base_url} does not advertise required model {alias}")
    return "ready"


def _clean(text: str) -> str:
    """Remove empty lines, surrounding quotes, and accidental role prefixes."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    line = " ".join(lines)
    if line.upper().startswith(("EN:", "ENGLISH:")):
        line = line.split(":", 1)[1].strip()
    return line.strip('"“”「」')
