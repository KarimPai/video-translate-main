"""Caption-only MPV playback for PLaMo 3 Translate 8B.

No audio extraction, VAD, ASR, DeepFilterNet, or diarization is started.
The player uses YouTube's original VOD clock, so native seekbar jumps never
require a second audio pipeline or a subtitle timestamp rebase.
"""
from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4

from .caption_feed import near_cues, prepare_caption_cues
from .captions import CaptionIndex
from .config import Config
from .contextualize import repair_nearby_terms
from .ingest import Source
from .output.mpv import (
    INPUT_CONF, JUMP_MESSAGE, OVERLAY_ID, MpvError, MpvIpc,
    _ass_many, find_mpv,
)
from .output.terminal import TerminalOutput
from .process_cleanup import _close_job, _create_kill_job
from .session_cache import SessionSubtitleCache
from .translate.base import Translator
from .types import Subtitle, Timings

log = logging.getLogger(__name__)

PREFETCH_SECONDS = 35.0
START_LEAD_SECONDS = 5.0
TICK_SECONDS = 0.12


class CaptionOnlyPlayer:
    """Translate nearby caption cues on demand while MPV plays the original VOD."""

    def __init__(
        self,
        source: Source,
        cfg: Config,
        captions: CaptionIndex,
        translator: Translator,
        *,
        vod_title: str = "",
    ) -> None:
        if source.live:
            raise RuntimeError(
                "Captions-only currently supports archived YouTube VODs. "
                "Live YouTube auto-captions cannot yet be retrieved and aligned "
                "reliably; no ASR fallback is running in this branch."
            )
        self.source = source
        self.cfg = cfg
        self.cues = prepare_caption_cues(
            captions, join_gap_s=cfg.translate.fragment_join_gap_s
        )
        if not self.cues:
            raise RuntimeError(
                "No usable Japanese captions were found. Captions-only mode "
                "cannot translate this video without a Japanese subtitle track."
            )
        self.translator = translator
        self.vod_title = vod_title
        self.cache = SessionSubtitleCache()
        self.terminal = TerminalOutput(simple=True)
        self._completed: set[int] = set()
        self._translated: dict[int, Subtitle] = {}
        self._condition = threading.Condition()
        self._target = source.start_s
        self._stop = threading.Event()
        self._error: Exception | None = None
        self._shown = ""
        self._proc: subprocess.Popen | None = None
        self._ipc: MpvIpc | None = None
        self._job = None
        self._ipc_path = ""
        self._worker: threading.Thread | None = None

    def _load_player(self) -> None:
        cfg = self.cfg.mpv
        self._ipc_path = (
            rf"\\.\pipe\auto-translate-captions-{uuid4().hex}"
            if sys.platform == "win32"
            else f"/tmp/auto-translate-captions-{uuid4().hex}.sock"
        )
        log_path = Path("logs/mpv.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        arguments = [
            find_mpv(cfg.path),
            "--pause",
            f"--input-ipc-server={self._ipc_path}",
            "--force-window=yes",
            "--title=auto-translate-captions",
            "--idle=yes",
            "--keep-open=yes",
            "--rebase-start-time=no",
            "--cache=yes",
            "--demuxer-readahead-secs=60",
            "--demuxer-max-bytes=256MiB",
            "--demuxer-max-back-bytes=256MiB",
            "--osc=yes",
            "--osd-on-seek=no",
            "--no-input-default-bindings",
            f"--input-conf={INPUT_CONF}",
            f"--hwdec={cfg.hwdec}",
            f"--log-file={log_path.resolve()}",
            "--msg-level=all=warn",
        ]
        if not Path(self.source.location).is_file():
            arguments.append(f"--ytdl-format={cfg.ytdlp_format}")
        self._proc = subprocess.Popen(
            arguments,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            self._job = _create_kill_job(self._proc)
        except (OSError, AttributeError):
            log.warning("Could not attach MPV to automatic process cleanup", exc_info=True)
        self._ipc = MpvIpc(self._ipc_path)
        self._ipc.command("loadfile", self.source.location, "replace")
        deadline = time.monotonic() + 45.0
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"MPV exited with code {self._proc.returncode}; see logs/mpv.log"
                )
            try:
                event = self._ipc.events.get(timeout=0.2)
            except queue.Empty:
                continue
            if event.get("event") == "file-loaded":
                break
            if event.get("event") == "end-file" and event.get("reason") == "error":
                raise RuntimeError("MPV could not load the video; see logs/mpv.log")
        else:
            raise TimeoutError("MPV did not load the original YouTube video within 45s")

        if self.source.start_s > 0:
            self._ipc.command("seek", self.source.start_s, "absolute+exact")
        log.info(
            "Caption-only MPV ready: %d Japanese cues, on-demand translation, "
            "no audio ingest/VAD/ASR", len(self.cues),
        )

    def _translate_worker(self) -> None:
        failures = 0
        try:
            while not self._stop.is_set():
                with self._condition:
                    candidates = near_cues(self.cues, self._target, PREFETCH_SECONDS)
                    index = next(
                        (i for i in candidates if i not in self._completed), None
                    )
                    if index is None:
                        self._condition.wait(timeout=0.3)
                        continue
                    cue = self.cues[index]
                    history = [
                        (self._translated[i].ja, self._translated[i].en)
                        for i in range(max(0, index - 12), index)
                        if i in self._translated
                    ]
                try:
                    nearby = [item[0] for item in history[-8:]]
                    future = [
                        following.text for following in self.cues[index + 1:index + 5]
                        if following.start_s <= cue.end_s + 2.0
                    ]
                    japanese = repair_nearby_terms(
                        cue.text, nearby, future, title=self.vod_title
                    )
                    started = time.monotonic()
                    english = self.translator.translate(japanese, history)
                    elapsed = time.monotonic() - started
                    if english and english.strip().upper() != "[SKIP]" and english != "[translation failed]":
                        subtitle = Subtitle(
                            cue.start_s, cue.end_s, japanese, english,
                            timings=Timings(mt_start=started, mt_end=time.monotonic()),
                        )
                        with self._condition:
                            self._translated[index] = subtitle
                            self.cache.add(subtitle)
                        self.terminal.show(subtitle)
                    log.debug(
                        "Caption translation %d/%d: %.2fs model time, %.2f-%.2fs",
                        index + 1, len(self.cues), elapsed, cue.start_s, cue.end_s,
                    )
                    failures = 0
                except Exception as exc:
                    failures += 1
                    log.exception(
                        "Caption %d translation failed (%d consecutive errors)",
                        index, failures,
                    )
                    if failures >= 3:
                        raise RuntimeError(
                            "PLaMo failed to translate three consecutive captions"
                        ) from exc
                finally:
                    with self._condition:
                        # Mark failed captions too, to avoid spinning forever
                        # on the same bad cue. Never report English success.
                        self._completed.add(index)
                        self._condition.notify_all()
        except Exception as exc:
            self._error = exc
            self._stop.set()

    def _set_target(self, position_s: float) -> None:
        with self._condition:
            self._target = position_s
            self._condition.notify_all()

    def _lead_ready(self, position_s: float) -> bool:
        with self._condition:
            horizon = min(
                position_s + START_LEAD_SECONDS,
                self.source.duration or float("inf"),
            )
            return all(
                i in self._completed
                for i in near_cues(self.cues, position_s, START_LEAD_SECONDS)
                if self.cues[i].start_s < horizon
            )

    def _render(self, position_s: float) -> None:
        assert self._ipc is not None
        active = self.cache.active_at(position_s, include_linger=True)
        active.sort(key=lambda sub: sub.start_s, reverse=True)
        active = active[:max(1, self.cfg.mpv.max_subtitle_rows)]
        text = _ass_many(active, self.cfg.mpv) if active else ""
        if text != self._shown:
            self._ipc.command(
                "osd-overlay",
                id=OVERLAY_ID,
                format="ass-events" if text else "none",
                data=text,
                res_x=0,
                res_y=720,
                z=10,
                hidden=False,
            )
            self._shown = text

    def _handle_keys(self) -> None:
        assert self._ipc is not None
        while True:
            try:
                event = self._ipc.events.get_nowait()
            except queue.Empty:
                break
            args = event.get("args") or []
            if event.get("event") != "client-message" or not args:
                continue
            if args[0] == JUMP_MESSAGE and len(args) >= 2:
                try:
                    delta = float(args[1])
                except (TypeError, ValueError):
                    continue
                self._ipc.command("seek", delta, "relative+exact")

    def run(self) -> None:
        """Block until MPV closes. Translation is driven by its real time-pos."""
        buffering = True
        resume_after_buffer = True
        last_pos: float | None = None
        last_at = time.monotonic()
        try:
            self._load_player()
            assert self._ipc is not None and self._proc is not None
            self._worker = threading.Thread(
                target=self._translate_worker,
                name="youtube-caption-translator",
                daemon=True,
            )
            self._worker.start()
            log.info("Preparing %.0fs caption translation lead", START_LEAD_SECONDS)

            while self._proc.poll() is None and not self._stop.is_set():
                self._handle_keys()
                reported = self._ipc.get("time-pos")
                position = float(reported) if reported is not None else self.source.start_s
                position = max(0.0, position)
                paused = self._ipc.get("pause")
                now = time.monotonic()
                if last_pos is not None and not buffering:
                    expected = last_pos + (0.0 if paused else now - last_at)
                    if abs(position - expected) > 2.0:
                        # MPV's native seekbar has moved the original VOD clock.
                        # Prior caption translations remain reusable by timestamp.
                        resume_after_buffer = paused is False
                        buffering = True
                        log.info("MPV seek detected at %.2fs; prioritizing local captions", position)
                        self._ipc.command("set_property", "pause", True)
                last_pos, last_at = position, now
                self._set_target(position)
                if buffering and self._lead_ready(position):
                    buffering = False
                    if resume_after_buffer:
                        self._ipc.command("set_property", "pause", False)
                    log.info("Captions ready near %.2fs; playback enabled", position)
                elif buffering and paused is False:
                    # Do not start ahead of untranslated nearby captions.
                    self._ipc.command("set_property", "pause", True)
                self._render(position)
                time.sleep(TICK_SECONDS)

            if self._error is not None:
                raise RuntimeError(str(self._error)) from self._error
            log.info("MPV closed; captions-only session finished")
        except KeyboardInterrupt:
            log.info("Stopping captions-only playback (Ctrl+C)")
        finally:
            self.close()

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None
        if self._ipc is not None:
            try:
                if self._proc is not None and self._proc.poll() is None:
                    self._ipc.command("quit")
            except (OSError, ConnectionError, MpvError):
                pass
        if self._proc is not None:
            try:
                self._proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=2.0)
        if self._ipc is not None:
            try:
                self._ipc.close()
            except OSError:
                pass
            self._ipc = None
        if self._job is not None:
            _close_job(self._job)
            self._job = None
        if self._ipc_path and sys.platform != "win32":
            try:
                Path(self._ipc_path).unlink(missing_ok=True)
            except OSError:
                pass
