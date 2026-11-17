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

from .base import TranslatorUnavailable
from ..config import TranslateConfig
from ..context_memory import (
    build_recent_context, recurring_term_hints, reuse_recent_short_term,
    select_dialogue_history, split_completed_japanese_sentences,
)
from ..glossary import Glossary
from ..runtime_diagnostics import record_event, rotate_log
from ..name_hints import canonicalize_english_names, preserve_spoken_names
from ..proper_names import name_glossary, resolve_names
from ..subject_guard import (
    neutralize_unsupported_subjects,
    unsupported_person_reference,
    neutralize_unwarranted_gender,
    neutralize_unwarranted_first_person,
    neutralize_unwarranted_possessives,
)
from ..translation_guard import suspicious_translation, _mention_value
from ..watame import correct_translation

log = logging.getLogger(__name__)


def _empty_name_diagnostics() -> dict:
    return {"matched_glossary": [], "raw_english": "", "corrected_english": "",
            "retry": False, "abstained": False, "error": None, "mentions": []}


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


def _current_glossary_matches(ja: str, vod_title: str, glossary: Glossary, previous_ja: str = "", *, mentions=None) -> list[tuple[str, str]]:
    """Longest-first, non-overlapping current-source matches (never future guesses)."""
    scene = (
        [(source, target) for source, target in _HOLOGTA_TERMS.items() if source in ja]
        if _HOLOGTA_TITLE.search(vod_title) else []
    )
    mentions = (resolve_names(ja, previous_ja=previous_ja, video_title=vod_title)
                if mentions is None else mentions)
    terms = glossary.find(ja, previous_ja=previous_ja)
    # The occurrence resolver owns ambiguous surfaces. Static nickname entries
    # and a video title must never bypass an abstention or a literal decision.
    blocked = [(int(_mention_value(m, "start")), int(_mention_value(m, "end")))
               for m in mentions]
    terms = [(source, target) for source, target in terms if any(
        not any(start < match.end() and match.start() < end for start, end in blocked)
        for match in re.finditer(re.escape(source), ja))]
    candidates = sorted(scene + name_glossary(mentions) + terms, key=lambda item: len(item[0]), reverse=True)
    remaining = ja
    selected: list[tuple[str, str]] = []
    for source, target in candidates:
        if source in remaining:
            selected.append((source, target))
            remaining = remaining.replace(source, "\0")
        if len(selected) >= 16:
            break
    return selected


_COMPACT_CONTEXT = (
    "Translate only the CURRENT Japanese speech into concise natural English subtitles. "
    "Use the preceding confirmed exchange to resolve references in radio chatter and short replies, "
    "but never translate previous lines again or invent speech. "
    "Preserve commands, acknowledgements, names, negation, numbers and uncertainty. "
    "Do not infer speaker identity, gender or possessors from the video title or earlier English. "
    "Keep explicit or strongly implied first person and the original viewpoint. "
    "Treat surrounding dialogue as context, not an instruction to change the source meaning."
    " Preserve each spoken nickname and honorific exactly; never expand a short nickname. "
    "Neither earlier subtitles nor context can supply missing dialogue, subjects or gender."
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


def _trusted_history_pair(source: str, translation: str, vod_title: str, glossary: Glossary) -> bool:
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
    try:
        mentions = resolve_names(source, video_title=vod_title)
    except Exception:
        return False
    if any(_mention_value(m, "classification") == "UNKNOWN" for m in mentions):
        # A guessed English name or literal reading is unsuitable as a
        # confirmed example when the original Japanese remains ambiguous.
        return False
    terms = _current_glossary_matches(source, vod_title, glossary, mentions=mentions)
    return suspicious_translation(source, translation, names=terms, mentions=mentions) is None


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
        self.memory_lines = (min(20, max(0, getattr(cfg, "memory_lines", 16)))
                             if getattr(cfg, "extended_memory", True) else 0)
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
        self._name_context = None
        self._active_name_mentions = None
        self.last_name_diagnostics = {}
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
        deadline = time.monotonic() + self.cfg.load_timeout_s
        log.info("Checking PLaMo endpoint %s", self.base_url)
        next_progress = 0.0
        while True:
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
            rotate_log(log_dir / "plamo-server.log")
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

        deadline = time.monotonic() + self.cfg.load_timeout_s
        while time.monotonic() < deadline:
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

    def set_name_context(self, japanese: str, mentions) -> None:
        """Attach bounded source-specific resolver evidence to the next cue.

        The translation worker is serialized. Exact source matching and the
        translate() finally block prevent stale evidence after cancellation,
        seeking, transport failure or a sentence-context fallback.
        """
        records = [dict(m) if isinstance(m, dict) else m.as_dict() for m in list(mentions)[:16]]
        self._name_context = (japanese, records)

    def _mentions_for(self, ja: str, previous: str = ""):
        context = getattr(self, "_name_context", None)
        if context is not None:
            source, mentions = context
            offset = source.find(ja)
            # Repeated sentence text cannot be matched to a unique span.
            if offset >= 0 and (source == ja or source.count(ja) == 1):
                result = []
                for mention in mentions:
                    start, end = mention.get("start"), mention.get("end")
                    if (type(start) is int and type(end) is int
                            and offset <= start < end <= offset + len(ja)
                            and source[start:end] == mention.get("source")):
                        result.append(dict(mention, start=start-offset, end=end-offset))
                return result
        return resolve_names(ja, previous_ja=previous, video_title=self.vod_title)

    def translate(self, ja: str, history: list[tuple[str, str]]) -> str:
        self.last_name_diagnostics = _empty_name_diagnostics()
        try:
            return self._translate_with_recovery(ja, history)
        except Exception as exc:
            self.last_name_diagnostics["error"] = type(exc).__name__
            raise
        finally:
            self._name_context = None
            self._active_name_mentions = None
            self._name_retry_context = False

    def _translate_with_recovery(self, ja: str, history: list[tuple[str, str]]) -> str:
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
            try:
                self._start_server()
                return self._translate_cue(ja, history)
            except Exception as recovery_error:
                raise TranslatorUnavailable(f"PLaMo recovery failed: {recovery_error}") from recovery_error

    def _stop_timed_out_inference(self) -> None:
        """Retire only an owned timed-out server before releasing GPU admission.

        Closing an HTTP connection does not prove llama.cpp stopped generating.
        Keep the dead process reference so the existing session recovery can
        restart once. Never terminate an attached external endpoint's process.
        """
        self._safe_timeout_stopped = False
        with self._close_lock:
            process = self.process
            if process is None:
                log.warning("Safe Playback: external PLaMo request timed out; inference cancellation "
                            "cannot be verified and the request will not be replayed")
                record_event("inference_cancellation_unverified", worker="plamo", owned=False,
                             reason="external_server_timeout", automatic_retry=False)
                return
            try:
                if process.poll() is None:
                    try:
                        process.terminate()
                        process.wait(timeout=2)
                    except (OSError, subprocess.TimeoutExpired):
                        # A stalled generation or failed graceful stop needs a
                        # bounded forced stop before another model is admitted.
                        process.kill()
                        process.wait(timeout=2)
                if process.poll() is None:
                    raise RuntimeError("owned PLaMo server still running after stop")
            except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                log.error("Safe Playback could not verify owned PLaMo timeout cleanup: %s", exc)
                record_event("inference_cancellation_unverified", worker="plamo", owned=True,
                             reason=type(exc).__name__, automatic_retry=False)
                raise TranslatorUnavailable("Owned PLaMo timeout cleanup failed; stopping translation") from exc
            self._safe_timeout_stopped = True
            record_event("inference_timeout_stopped", worker="plamo", owned=True,
                         pid=process.pid, exit_code=process.returncode)
            log.warning("Safe Playback stopped timed-out owned PLaMo inference (PID %s)", process.pid)

    def _translate_cue(self, ja: str, history: list[tuple[str, str]]) -> str:
        """Translate complete exchanges without changing media timestamps.

        The segmented path runs only on clear punctuation boundaries; every
        finalized sentence informs the next one. If a pass fails, preserve the
        original whole-utterance behavior instead of emitting half a cue.
        """
        extended = getattr(self, "memory_lines", 0) if getattr(self.cfg, "extended_memory", True) else 0
        limit = max(min(20, max(0, self.cfg.context_lines)), extended)
        history = history[-limit:] if limit else []
        segments = (split_completed_japanese_sentences(ja)
                    if getattr(self.cfg, "sentence_context", False) else [ja])
        if len(segments) < 2:
            return self._translate_one(ja, history)
        recent = list(history)
        english = []
        segment_records = []
        try:
            for segment in segments:
                translated = self._translate_one(segment, recent)
                segment_records.append(dict(getattr(self, "last_name_diagnostics", {}), source=segment))
                if not translated.strip() and getattr(self, "last_name_diagnostics", {}).get("unresolved_name_issue"):
                    # Do not spend another whole-cue pass after the bounded
                    # name retry has already failed, or emit a partial cue.
                    self._combine_name_diagnostics(ja, segment_records, "")
                    return ""
                if not translated.strip() or suspicious_translation(segment, translated):
                    raise ValueError("unreliable sentence translation")
                english.append(translated.strip())
                recent.append((segment, translated))
        except httpx.TransportError:
            raise
        except Exception as exc:
            log.warning("Sentence context fallback to whole utterance: %s", exc)
            result = self._translate_one(ja, history)
            self.last_name_diagnostics["segments"] = segment_records
            self.last_name_diagnostics["sentence_fallback"] = True
            return result
        result = " ".join(english)
        self._combine_name_diagnostics(ja, segment_records, result)
        return result

    def _combine_name_diagnostics(self, ja: str, records: list[dict], english: str) -> None:
        combined = _empty_name_diagnostics()
        combined["segments"] = records[:3]
        combined["raw_english"] = " ".join(record.get("raw_english", "") for record in records).strip()
        combined["corrected_english"] = english
        offset = 0
        for record in records:
            combined["retry"] |= bool(record.get("retry"))
            combined["abstained"] |= bool(record.get("abstained"))
            if record.get("error") and not combined["error"]:
                combined["error"] = record["error"]
            for key in ("retry_reason", "retry_error", "unresolved_name_issue", "literal_noun_review"):
                if record.get(key):
                    combined[key] = record[key]
            for pair in record.get("matched_glossary", []):
                if pair not in combined["matched_glossary"]:
                    combined["matched_glossary"].append(pair)
            source = record.get("source", "")
            begin = ja.find(source, offset) if source else -1
            if begin >= 0:
                for mention in record.get("mentions", []):
                    combined["mentions"].append(dict(mention, start=mention["start"] + begin,
                                                      end=mention["end"] + begin))
                offset = begin + len(source)
        self.last_name_diagnostics = combined

    def _translate_one(self, ja: str, history: list[tuple[str, str]]) -> str:
        self.last_name_diagnostics = _empty_name_diagnostics()
        previous = history[-1][0] if history else ""
        try:
            mentions = self._mentions_for(ja, previous)
            names = _current_glossary_matches(ja, self.vod_title, self.glossary, previous, mentions=mentions)
        except Exception as exc:
            # Advisory name resolution must never lose later subtitles.
            log.warning("Proper-name resolution unavailable; translating unchanged Japanese: %s", exc)
            mentions, names = [], []
            self.last_name_diagnostics["error"] = "resolver:" + type(exc).__name__
            self.last_name_diagnostics["abstained"] = True
        self._active_name_mentions = (ja, mentions)
        diagnostics = getattr(self, "last_name_diagnostics", {})
        diagnostics.update(matched_glossary=[{"source": s, "target": t} for s, t in names],
                           mentions=[dict(m) if isinstance(m, dict) else m.as_dict() for m in mentions])
        diagnostics["abstained"] = diagnostics.get("abstained", False) or any(
            _mention_value(m, "classification") == "UNKNOWN" for m in mentions)

        def normalize_names(english: str) -> str:
            if any(target in ("Haachama", "Laplus") for _, target in names):
                english = canonicalize_english_names(ja, english, self.vod_title)
            return preserve_spoken_names(ja, english, names, mentions=mentions)

        if (getattr(self.cfg, "extended_memory", True) and getattr(self, "memory_lines", 0)
                and not _KUSA.search(ja) and not mentions):
            reused = reuse_recent_short_term(ja, history)
            consistent = recurring_term_hints(ja, history, self.glossary)
            reviewed = any(target == (reused or "").rstrip("!?。！？") for _, target in names)
            if (reused is not None and (consistent or reviewed)
                    and _trusted_history_pair(ja, reused, self.vod_title, self.glossary)):
                log.debug("Reusing recent short term: JP=%r EN=%r", ja, reused)
                # Cached subtitles must obey current short-name policy too.
                result = normalize_names(reused)
                diagnostics.update(raw_english=reused, corrected_english=result, reused=True)
                return result

        def normalized(raw: str) -> str:
            translated = _clean(raw)
            if _KUSA.search(ja) and re.search(r"\blol\b", translated, re.I):
                literal = re.search(r"草[をがはもに]*(?:刈|食|抜|生やさ|集め)|(?:羊|牛|馬).*食|草むしり", ja)
                if not literal and ja.strip("。！？!? ") in ("草", "くさ", "クサ") and previous:
                    literal = re.search(r"庭|雑草|草刈|草食|(?:羊|牛|馬).*食", previous)
                translated = re.sub(r"\blol\b", "grass" if literal else "kusa", translated, flags=re.I)
            if any(target in ("Watame", "Watamee", "Tsunomaki Watame") for _, target in names):
                translated = correct_translation(ja, translated)
            return neutralize_unsupported_subjects(ja, normalize_names(translated))

        try:
            raw = self._chat(ja, history, timeout=self.cfg.timeout_s, names=names)
        except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            if isinstance(exc, httpx.TimeoutException) and getattr(self, "safe_playback", False):
                # _chat already retired an owned timed-out process. Transport
                # recovery belongs to translate(); never replay against a
                # possibly still-generating endpoint here.
                raise
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                raise
            log.warning("Retrying transient PLaMo request failure: %s", exc)
            raw = self._chat(ja, history, timeout=min(self.cfg.timeout_s, self.cfg.retry_timeout_s), names=names)
        english = normalized(raw)
        diagnostics["raw_english"] = _clean(raw)
        pronoun_reason = (
            unsupported_person_reference(ja, english)
            if self.cfg.neutral_subjects else None
        )
        reason = pronoun_reason or suspicious_translation(ja, english, names=names, mentions=mentions)
        if reason:
            self._count_validation("flag:" + reason)
        if reason and self.cfg.retry_suspicious:
            diagnostics["retry"] = True
            diagnostics["retry_reason"] = reason
            self._name_retry_context = reason in ("proper_name_mismatch", "proper_name_expansion", "literal_noun_mismatch",
                                                  "unsupported_proper_name", "possible_literal_noun_loss")
            log.info("PLaMo suspicious translation (%s): JP=%r EN=%r; retry without prior chat",
                     reason, ja[:150], english[:160])
            try:
                # Extra neutral instruction only when the first pass introduced
                # unsupported person/gender references. No new model is loaded.
                candidate = normalized(self._chat(
                    ja, [], timeout=min(self.cfg.timeout_s, self.cfg.retry_timeout_s),
                    neutral_retry=bool(pronoun_reason), names=names,
                ))
            except Exception as exc:
                if (getattr(self, "safe_playback", False)
                        and (isinstance(exc, TranslatorUnavailable)
                             or (isinstance(exc, httpx.TimeoutException)
                                 and not getattr(self, "_safe_timeout_stopped", False)))):
                    raise
                self._count_validation("retry_unavailable")
                diagnostics["retry_error"] = type(exc).__name__
                log.warning("PLaMo validation retry unavailable (%s); using first result", exc)
            else:
                retry_reason = (
                    unsupported_person_reference(ja, candidate)
                    if self.cfg.neutral_subjects else None
                ) or suspicious_translation(ja, candidate, names=names, mentions=mentions)
                if retry_reason is None:
                    english = candidate
                    diagnostics["retry_accepted"] = True
                    self._count_validation("retry_accepted")
                    log.info("PLaMo validation retry accepted (trigger=%s)", reason)
                else:
                    self._count_validation("retry_rejected:" + retry_reason)
                    log.warning(
                        "PLaMo retry still suspicious (%s); keeping first result with safe fallbacks",
                        retry_reason,
                    )
                    diagnostics["retry_accepted"] = False
            finally:
                self._name_retry_context = False
        if self.cfg.neutral_subjects:
            # Prefer owner-free noun phrases BEFORE gender fallback converts
            # "his/her customer" into an invented "their customer".
            english = neutralize_unwarranted_possessives(ja, english)
            # Restrict remaining fallbacks to grammar-safe forms. A source
            # that explicitly states a person/gender retains that information.
            english = neutralize_unwarranted_gender(ja, english)
            english = neutralize_unwarranted_first_person(ja, english)
        unresolved = suspicious_translation(ja, english, names=names, mentions=mentions)
        if unresolved in ("proper_name_mismatch", "proper_name_expansion", "literal_noun_mismatch", "unsupported_proper_name"):
            diagnostics["abstained"] = True
            diagnostics["unresolved_name_issue"] = unresolved
            # A detected name error that survived the single retry is not a
            # usable English subtitle. Returning an empty cue uses the normal
            # pipeline suppression path while later cues continue normally.
            log.warning("Suppressing unresolved PLaMo name error (%s)", unresolved)
            english = ""
        elif unresolved == "possible_literal_noun_loss":
            diagnostics["literal_noun_review"] = unresolved
        diagnostics["corrected_english"] = english
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
        active = getattr(self, "_active_name_mentions", None)
        try:
            mentions = (active[1] if active is not None and active[0] == ja else
                        self._mentions_for(ja, history[-1][0] if history else ""))
            hits = names if names is not None else _current_glossary_matches(
                ja, self.vod_title, self.glossary, mentions=mentions)
        except Exception as exc:
            log.warning("Skipping unavailable name hints: %s", exc)
            mentions, hits = [], []
        memory_lines = (min(20, max(0, getattr(self, "memory_lines", 0)))
                        if getattr(self.cfg, "extended_memory", True) else 0)
        history_limit = max(min(20, max(0, self.cfg.context_lines)), memory_lines)
        history = history[-history_limit:] if history_limit else []
        # Reserve room for current speech, template guidance, and output. This
        # conservative character budget needs no tokenizer/model allocation.
        available = max(0, getattr(self.cfg, "plamo_context", 4096)
                        - self.cfg.max_tokens - 1536 - min(512, len(ja) * 3))
        history_chars = min(1024, available // 2)
        memory_chars = min(384, history_chars // 3) if memory_lines else 0
        dialogue_history = select_dialogue_history(
            history, ja, context_lines=self.cfg.context_lines,
            memory_lines=memory_lines, max_chars=history_chars - memory_chars,
        )
        compact = getattr(self.cfg, "context_profile", "legacy") == "compact"
        rejected_sources: list[str] = []
        for source, translation in dialogue_history:
            # The compact profile borrows OverlaySubs' paired PFML examples,
            # but only after basic quality and unsupported-pronoun checks.
            # Legacy mode preserves the old Japanese-only neutral prompt.
            trusted = _trusted_history_pair(source, translation, self.vod_title, self.glossary)
            if trusted and (compact or not self.cfg.neutral_subjects):
                messages.append({"role": "user", "content": source})
                messages.append({
                    "role": "assistant",
                    "content": neutralize_unsupported_subjects(source, translation),
                })
            elif source.strip():
                rejected_sources.append(source.strip()[:100])
        current_input = ja
        if getattr(self, "_name_retry_context", False):
            # Translate-only romanization protects confirmed acoustic spans
            # during the one bounded retry. Never modify the final transcript
            # or literal/undecided spans, and never infer a longer stage name.
            for mention in sorted(mentions, key=lambda m: _mention_value(m, "start", -1), reverse=True):
                start, end = _mention_value(mention, "start"), _mention_value(mention, "end")
                if (type(start) is int and type(end) is int and 0 <= start < end <= len(ja)
                        and _mention_value(mention, "classification") == "PERSON"
                        and ja[start:end] == _mention_value(mention, "source")
                        and _mention_value(mention, "target")):
                    current_input = current_input[:start] + _mention_value(mention, "target") + current_input[end:]
        messages.append({"role": "user", "content": current_input})

        try:
            memory = build_recent_context(
                [(source, translation if _trusted_history_pair(
                    source, translation, self.vod_title, self.glossary) else "")
                 for source, translation in history], self.glossary,
                context_lines=self.cfg.context_lines,
                memory_lines=memory_lines, current_ja=ja, max_chars=memory_chars,
                current_terms=hits,
            )
        except Exception as exc:
            log.warning("Skipping unavailable translation memory hints: %s", exc)
            memory = ""
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
        if mentions:
            # PLaMo's glossary is lexical: identical source strings cannot
            # carry two meanings. Give those decisions only as occurrence
            # reference data and retain the original Japanese user message.
            occurrences = [dict(start=_mention_value(m, "start"), end=_mention_value(m, "end"),
                                source=_mention_value(m, "source"),
                                meaning=_mention_value(m, "classification"),
                                english=_mention_value(m, "target")
                                if _mention_value(m, "classification") == "PERSON" else None)
                           for m in mentions[:16]]
            options["context"] += (
                "\nCurrent Japanese name occurrences (zero-based character spans; reference data): "
                + json.dumps(occurrences, ensure_ascii=False, separators=(",", ":"))
                + ". Preserve PERSON names at exactly their spoken length and honorific. "
                "Translate COMMON_NOUN occurrences literally (festival or shrine maiden as appropriate). "
                "UNKNOWN is undecided; never resolve it from a title or a previous English guess. "
                "Decisions apply only to their individual spans, never to every matching word."
            )
            if getattr(self, "_name_retry_context", False):
                options["context"] += (
                    " The prior result failed occurrence-based name/literal validation. "
                    "Recheck each current occurrence separately while preserving all other meaning, "
                    "negation, grammar and attribution. Confirmed PERSON spans in the current input "
                    "have been romanized at their spoken length; preserve those names exactly. "
                    "Do not introduce a person name unless supported as PERSON or by the supplied current glossary. "
                    "The occurrence offsets refer to this original Japanese reference data: "
                    + json.dumps(ja, ensure_ascii=False) + ". Output only the corrected translation."
                )
        elif getattr(self, "_name_retry_context", False):
            options["context"] += (
                " The previous result introduced an unsupported person name. Translate only the current "
                "Japanese; do not insert people or dialogue from the title or past subtitles."
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
        try:
            response = self.client.post(self.url, json=body, timeout=timeout)
        except httpx.TimeoutException:
            if getattr(self, "safe_playback", False):
                self._stop_timed_out_inference()
            raise
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


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
