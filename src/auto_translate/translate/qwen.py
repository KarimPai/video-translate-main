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
from ..context_memory import SceneMemory, reuse_recent_short_term
from ..glossary import Glossary
from ..process_cleanup import _create_kill_job, _close_job
from ..stream_context import (
    ark_glossary_hits, is_ark_context, is_gta_context,
    is_horse_racing_context, is_incomplete_fragment,
)
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
    "Preserve tone, humor, strong language, names, honorifics, numbers, "
    "negation, uncertainty, and meaning without adding or censoring information. "
    "When Arabic numerals occur together in Japanese ASR without separators, "
    "preserve their exact order and digit count: do not add thousands commas, "
    "insert zeros, or reinterpret an ambiguous number string as currency. "
    "Accurately convert the Japanese number unit 万 (10,000): 30万 is "
    "300,000, NOT thirty thousand; do not add yen unless spoken. "
    "見失う means losing sight of a person or thing, NOT losing one's way. "
    "The colloquial pattern verb + しかねえ/しかない means there is no choice "
    "BUT to do the verb, NOT an inability to do it. "
    "Do not reverse praise, rejection, or other sentiment. "
    "Spoken names ending with -chan, -san or -sama retain those suffixes in English. "
    "Do not expand a nickname to a full name or invent a member name based on context. "
    "Keep short ambiguous phonetic nicknames as spoken; never complete them "
    "into unrelated character or franchise names. "
    "猫ちゃん means kitty/Neko-chan, not Nekomata-chan; do not infer someone's "
    "identity from an ambiguous nickname. "
    "Never add insults, blame, accusations, or spoken words absent from Japanese. "
    "Japanese may omit the grammatical subject; avoid guessing who acted. "
    "The input is automatic speech recognition and may contain errors; "
    "use the provided terminology only when the current Japanese actually supports it. "
    "Previous lines and video title are background, not words spoken in the current line. "
    "For unfinished clauses translate the fragment without guessing the missing ending. "
    "Skip only if the WHOLE line is clearly nonverbal noise; then reply exactly [SKIP]. "
    "Otherwise always translate the meaningful Japanese, even if unclear."
)
_GIST_STYLE = (
    "Subtitle style: concise gist, not a word-for-word transcript. "
    "Express the main point in one short, natural English sentence or fragment, "
    "usually 5 to 16 words when the source allows. "
    "Remove hesitations, filler, duplicate words and redundant conversational framing. "
    "Compress long repetitions of the same short word into a brief emphatic phrase. "
    "If the transcript appears to include two replies, preserve each distinct "
    "statement in order; do not collapse them into one speaker's perspective. "
    "Preserve every meaningful negation, question, uncertainty, important number, "
    "explicit speaker reference, proper name, game term, and essential action. "
    "Do not soften strong language, reverse sentiment, invent details, or guess "
    "what garbled recognition was meant to say. "
    "Shorten phrasing, NEVER change what happened or who did it. "
    "An unfinished Japanese sentence must remain an unfinished English thought."
)
_FAITHFUL_STYLE = (
    "Subtitle style: faithful natural translation. Preserve the spoken details "
    "and conversational structure without unnecessary filler or embellishment."
)

_REVIEW_HINTS = {
    "positive_appraisal_reversed": "Japanese clearly praises something; do not turn it into criticism.",
    "loading_performance_reversed": "軽くなった means lighter or less demanding, NOT slower loading.",
    "rejection_reversed": "Repeated のん or ノン expresses rejection: no, no, no.",
    "rhetorical_egunai_reversed": (
        "Gaming slang えぐない is an emphatic rhetorical question: "
        "isn't that crazy/brutal? It does not mean things are not so bad."
    ),
    "possible_missing_negation": "Recheck explicit Japanese negation without inventing one.",
    "exclusive_necessity_reversed": (
        "Japanese しかねえ/しかない after a verb means there is no choice BUT to do it. "
        "The speaker is going to do it, not saying they cannot do it."
    ),
    "japanese_man_unit_wrong": "30万 is 300,000 (thirty times ten thousand), NOT 30,000.",
    "lost_sight_as_lost_way": "見失う is to lose sight of someone/something, NOT to lose one's way.",
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
    # 猫ちゃん is not evidence for the specific person Nekomata. Never infer
    # another person's name solely from the video title or a prior cue.
    if ("猫ちゃん" in japanese and not any(
            name in japanese for name in ("猫又", "ネコマタ", "ねこまた"))):
        english = re.sub(r"\bNekomata-chan\b", "Neko-chan", english, flags=re.I)
    return english


def _remove_unspoken_insult(japanese: str, english: str) -> str:
    """Remove only a known trailing Qwen addition if Japanese lacks an insult."""
    if re.search(r"バカ|ばか|馬鹿|アホ|阿呆|クソ|くそ|死ね|ふざけ", japanese):
        return english
    return re.sub(
        r",\s+you idiot(?P<punct>[.!?]?)\s*$",
        lambda match: match.group("punct"),
        english,
        flags=re.I,
    )


class QwenTranslator:
    """Only local Qwen3.5-4B; no external API or alternative translator."""

    def __init__(self, cfg: TranslateConfig, glossary: Glossary, *, vod_title: str = "") -> None:
        self.cfg = cfg
        self.glossary = glossary
        self.memory_lines = max(0, cfg.memory_lines)
        self.vod_title = " ".join(vod_title.split())[:200]
        self._scene_memory = SceneMemory(
            getattr(cfg, "scene_context_max_idle_lines", 32)
        )
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
        if process is None and job is None and self._server_log is None:
            return  # Already closed, or never owned a server.
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
            if job is not None:
                _close_job(job)
            if self._server_log is not None:
                self._server_log.close()
                self._server_log = None
            atexit.unregister(self.close)

    def reset_scene_context(self) -> None:
        """Discard scene evidence on a seek or timeline generation change."""
        scene = getattr(self, "_scene_memory", None)
        if scene is not None:
            scene.reset()

    def observe_scene(self, japanese: str) -> None:
        """Update scene evidence from the recognized Japanese, never English."""
        if not getattr(self.cfg, "scene_context_enabled", True):
            return
        if not hasattr(self, "_scene_memory"):
            self._scene_memory = SceneMemory(
                getattr(self.cfg, "scene_context_max_idle_lines", 32)
            )
        self._scene_memory.observe(japanese)

    def _scene_topic(self) -> str | None:
        if not getattr(self.cfg, "scene_context_enabled", True):
            return None
        memory = getattr(self, "_scene_memory", None)
        if memory is None:
            return None
        topic = memory.current_topic()
        if topic is None and memory.topic is not None:
            # An unconfirmed topic change must not resurrect stale
            # title-only/game-specific guidance from the previous scene.
            return "__transition__"
        return topic

    def _glossary_hits(self, ja: str, history: list[tuple[str, str]]) -> list[tuple[str, str]]:
        hits = self.glossary.find(ja)
        # Use an exact current-cue gaming term. Do not alter the original
        # 327-term glossary or mistake carjacking for kidnapping.
        if "カージャック" in ja and not any(term == "カージャック" for term, _ in hits):
            hits.append(("カージャック", "carjack / carjacking (vehicle theft, not kidnapping)"))
        recent_ja = [jp for jp, _ in history[-8:]]
        if (self._scene_topic() == "ark" or
                (self._scene_topic() is None and
                 is_ark_context(self.vod_title, ja, recent_ja))):
            existing = {source for source, _ in hits}
            hits.extend((src, target) for src, target in ark_glossary_hits(ja) if src not in existing)
        if (self._scene_topic() == "horse_racing" or
                (self._scene_topic() is None and
                 is_horse_racing_context(self.vod_title, ja, recent_ja))):
            racing_terms = {
                "ワイド": "wide bet (Japanese place quinella; NOT a straight win bet)",
                "軸馬": "key horse / anchor horse in betting combinations",
                "馬券": "horse-racing betting ticket",
                "三連複": "trio bet (three horses in top three, any order)",
                "複勝": "place bet",
                "単勝": "win bet",
                "払い戻し": "cash-out / payout for horse-racing bets (NOT a retail refund)",
                "払戻し": "cash-out / payout for horse-racing bets (NOT a retail refund)",
            }
            existing = {source for source, _ in hits}
            for src, target in racing_terms.items():
                if src in ja and src not in existing:
                    hits.append((src, target))
        return hits[:10]

    def translate(self, ja: str, history: list[tuple[str, str]]) -> str:
        # Observe once per current speech cue; no extra Qwen invocation.
        self.observe_scene(ja)
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
        english = _remove_unspoken_insult(ja, english)
        english = neutralize_unsupported_subjects(ja, english)
        # In Japanese horse racing, 払い戻し is cashing out a bet,
        # not returning purchased goods to a retailer.
        if ("払い戻し" in ja or "払戻し" in ja) and (
            self._scene_topic() == "horse_racing"
            or (self._scene_topic() is None and is_horse_racing_context(
                self.vod_title, ja, [source for source, _ in history[-8:]]
            ))
        ):
            english = re.sub(
                r"\brefund(s?)\b",
                lambda m: "payouts" if m.group(1) else "payout",
                english, flags=re.I,
            )
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
        scene = getattr(self, "_scene_memory", None)
        if getattr(self.cfg, "scene_context_enabled", True) and scene is not None:
            hint = scene.hint()
            if hint:
                blocks.append(
                    "Scene memory (prior recognized Japanese; NOT spoken in the "
                    "current line, do not add details): " + hint
                )
        hits = self._glossary_hits(ja, history)
        if hits:
            blocks.append(
                "Term mappings ONLY if present in the current Japanese:\n"
                + "\n".join(src + " => " + target for src, target in hits)
            )
        # A Hololive X期生 group is a generation, not a school year.
        # Only add this hint when a generation term was actually recognized;
        # never use the title to supply missing words or identities.
        if (
            re.search(r"(?:[一二三四五六七八九十0-9０-９]+期生)", ja)
            and re.search(r"ホロライブ|hololive", self.vod_title, re.I)
        ):
            blocks.append(
                "Hololive terminology: X期生 means Xth-generation members, "
                "not school-year students. Translate only the generation "
                "actually present in the current Japanese; do not infer "
                "member names or add absent speech."
            )
        if (
            self._scene_topic() == "ark"
            or (self._scene_topic() is None and is_ark_context(
                self.vod_title, ja, [jp for jp, _ in history[-8:]]
            ))
        ):
            blocks.append("Game context: ARK dinosaurs and survival gameplay. ギガノト is Giganotosaurus.")
        if (
            self._scene_topic() == "gta"
            or (self._scene_topic() is None and is_gta_context(self.vod_title))
        ):
            blocks.append(
                "Game context: GTA roleplay with car pursuits, store robberies, "
                "and radio communication only when supported by the Japanese. "
                "裏手 for a building is around the back / rear side, not a theater "
                "backstage. 見失った / 見失いました means lost sight of someone "
                "or something, not lost your way. アタック禁止 is no attacking; "
                "30秒 means 30 seconds, not minutes. Do not invent targets, "
                "actions, speaker identities or misunderstood names."
            )
        if (
            self._scene_topic() == "horse_racing"
            or (self._scene_topic() is None and is_horse_racing_context(
                self.vod_title, ja, [jp for jp, _ in history[-8:]]
            ))
        ):
            blocks.append(
                "Event context: Japanese horse racing and betting. Odds are "
                "betting odds; the paddock is where horses are inspected; "
                "推奨馬 is a recommended horse, never a bar. ワイド is a "
                "place-quinella bet where both selected horses finish in the "
                "top three; 軸 is a key or anchor horse, not a straight bet. "
                "払い戻し/払戻し in this betting context is collecting a payout, "
                "not refunding a purchased item. "
                "Unseparated digits may be spoken horse numbers: preserve "
                "their exact sequence, never turn 62468 into 62,468 or 468 "
                "into 4,680. Keep racehorse names as proper names. Do not "
                "invent an identity or repair unclear ASR without evidence."
            )
            if "ミルフィーユドール" in ja:
                blocks.append(
                    "ミルフィーユドール is a horse name; preserve its phonetic "
                    "name (Millefeuille d'Or), not 'Milky Way Doll'."
                )
        if is_incomplete_fragment(ja):
            blocks.append("Current sentence is unfinished; translate only the heard fragment.")
        elif history and is_incomplete_fragment(history[-1][0]):
            blocks.append("Current line may continue an unfinished prior sentence; do not repeat prior translation.")
        if review_hint:
            blocks.append("RECHECK: " + _REVIEW_HINTS.get(review_hint, "Preserve the Japanese meaning exactly."))
        blocks.append("CURRENT JAPANESE (translate only this line):\n" + ja)
        style = getattr(self.cfg, "subtitle_style", "gist")
        if style not in {"gist", "faithful"}:
            raise ValueError(f"Invalid subtitle_style {style!r}: expected 'gist' or 'faithful'")
        style_prompt = _GIST_STYLE if style == "gist" else _FAITHFUL_STYLE
        body = {
            "model": self.cfg.qwen_model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT + " " + style_prompt},
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
