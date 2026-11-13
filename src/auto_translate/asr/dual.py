import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
import urllib.error

import numpy as np

from ..text_filter import is_usable_transcript
from .whisper_turbo import WhisperTurboASR
from .language_router import classify_language
from .diagnostics import append_comparison
from .reconciler import Reconciler
from .selection import compare

log = logging.getLogger(__name__)
SECONDARY = "whisper-large-v3-turbo"


@dataclass(frozen=True)
class LanguageProbe:
    """Legacy optional language probe, unused by Japanese-only Whisper Turbo."""
    text: str
    language: str | None
    route: str
    elapsed_s: float


class DualASR:
    def __init__(self, cfg, budget, glossary, live):
        self.cfg = cfg
        self.budget = budget
        self.glossary = glossary
        self.live = live
        self.mode = cfg.dual_asr.secondary_live_mode if live else cfg.dual_asr.secondary_vod_mode
        self.worker = self.selector = None
        self.generation = None
        self.history = deque(maxlen=4)
        self.lock = threading.Lock()
        self.pending = None
        self.playback_position = lambda: None
        self.session_id = str(time.time_ns())
        self.language_routing_supported = True
        self.language_failures = 0
        self.previous_english_end_s = None
        self.previous_english_generation = None
        if not cfg.dual_asr.enabled or self.mode == "off":
            return
        try:
            # Admission includes MPV, VAD, model scratch and Windows headroom.
            if cfg.dual_asr.secondary_device == "cuda" and not budget.admits(1.9):
                raise RuntimeError("insufficient measured GPU headroom for secondary ASR")
            self.worker = WhisperTurboASR(
                model=cfg.dual_asr.secondary_model,
                device=cfg.dual_asr.secondary_device,
                compute_type=cfg.dual_asr.secondary_compute_type,
                beam_size=cfg.dual_asr.secondary_beam_size,
            )
            if cfg.dual_asr.secondary_device == "cuda" and not budget.admits(0.3):
                raise RuntimeError("secondary ASR exceeded its startup reservation")
            log.info("Secondary %s ready on %s (%s)", SECONDARY,
                     cfg.dual_asr.secondary_device, self.mode)
        except Exception as exc:
            log.warning("Haas + PLaMo fallback: %s", exc)
            self.release()
            return
        if cfg.reconciler.enabled:
            self._start_selector(cfg.reconciler.gpu_layers)

    def _start_selector(self, layers):
        # Decimal GB estimates for the 731 MB LFM GGUF, context/compute buffers
        # and MPV/VAD allocations that may arrive after model startup.
        playback_reserve = 0.65
        if layers and self.budget.admits(1.15 + playback_reserve):
            try:
                self.selector = Reconciler(layers)
                if self.budget.admits(playback_reserve):
                    log.info("LFM2.5-1.2B-JP transcript selector ready; GPU layers requested=%d", layers)
                    return
                self.selector.close()
                self.selector = None
                log.warning("Selector exceeded startup reservation; switching to CPU")
            except Exception as exc:
                log.warning("GPU selector failed to start; trying CPU: %s", exc)
        elif layers:
            log.info("Insufficient GPU headroom for LFM2.5-1.2B-JP; selecting on CPU")
        try:
            self.selector = Reconciler(0)
            log.info("LFM2.5-1.2B-JP transcript selector ready; CPU")
        except Exception as exc:
            self.selector = None
            log.warning("Selector unavailable; deterministic selection: %s", exc)

    def release_selector(self):
        """Reclaim the optional GPU selector first, retaining secondary ASR."""
        with self.lock:
            selector = self.selector
            # A CPU selector consumes no GPU VRAM: retain it at soft pressure.
            if selector is None or not getattr(selector, "gpu_layers", 0):
                return
            self.selector = None
        log.warning("GPU pressure: releasing optional GPU transcript selector")
        selector.close()

    def release(self):
        with self.lock:
            worker, selector = self.worker, self.selector
            self.worker = self.selector = None
        if worker or selector:
            log.warning("Releasing optional models for the rest of this session")
        try:
            if worker:
                worker.close()
        finally:
            if selector:
                selector.close()

    def deadline(self, utt):
        if not self.live:
            return float("inf")
        deadline = utt.timings.audio_read + self.cfg.mpv.delay_s - 2
        position = self.playback_position()
        if position is not None:
            deadline = min(deadline, time.monotonic() + utt.end_s - position - 2)
        return deadline

    def remember(self, generation, text):
        with self.lock:
            if generation != self.generation:
                self.history.clear()
                self.generation = generation
            self.history.append(text[:160])

    def preflight_language(self, utt, current=None):
        """Whisper Turbo JA is Japanese-only; do not attempt language detection."""
        return None

    def select_segment(self, utt, primary, title, backlog, current, primary_stop=None, primary_elapsed_s=None, prefetched=None):
        # Single-flight optional work. A blocked pipe or HTTP read cannot hold
        # the live ASR stage past its deadline or accumulate an executor queue.
        if self.worker is None or (self.pending and self.pending.is_alive()):
            return None
        # Backpressure must not extend an otherwise usable primary transcript.
        # Still allow rescue when Haas missed speech entirely.
        if backlog() and (self.live or is_usable_transcript(primary)):
            return None
        cancelled = threading.Event()
        finished = threading.Event()
        result = [None]

        def generation():
            if cancelled.is_set():
                return -1
            return current() if current else utt.generation

        def recognize():
            try:
                result[0] = self.choose(utt, primary, title, backlog, generation, primary_stop, primary_elapsed_s, prefetched)
            except Exception:
                log.exception("Optional recognition failed at %.3fs", utt.start_s)
            finally:
                finished.set()

        self.pending = threading.Thread(target=recognize, name="optional-asr", daemon=True)
        self.pending.start()
        timeout = (self.cfg.dual_asr.live_timeout_s + self.cfg.reconciler.live_timeout_s
                   if self.live else self.cfg.dual_asr.vod_timeout_s + self.cfg.reconciler.vod_timeout_s)
        deadline = time.monotonic() + timeout
        if self.live:
            deadline = min(deadline, self.deadline(utt))
        while not finished.wait(.05):
            if (time.monotonic() >= deadline or generation() != utt.generation
                    or (backlog() and (self.live or is_usable_transcript(primary)))):
                cancelled.set()
                return None
        self.pending.join()
        return result[0]

    def choose(self, utt, primary, title, backlog, current, primary_stop=None, primary_elapsed_s=None, prefetched=None):
        """Return a selected original, or None to use the existing Haas path."""
        worker, selector = self.worker, self.selector
        if worker is None:
            return None
        with self.lock:
            context = list(self.history) if self.generation == utt.generation else []
        if (utt.probe or utt.vad_confidence is None or utt.vad_confidence < self.cfg.vad.threshold
                or not len(utt.audio) or not np.isfinite(utt.audio).all()
                or np.max(np.abs(utt.audio)) < 1e-5):
            return ""
        if self.budget.pressure_level == "critical":
            return None
        end = self.deadline(utt)
        secondary_budget = (self.cfg.dual_asr.live_timeout_s if self.live
                            else self.cfg.dual_asr.vod_timeout_s)
        if self.live and (backlog() or time.monotonic() + secondary_budget > end):
            return None
        if not self.live and backlog() and is_usable_transcript(primary):
            return None
        if (self.mode == "selective" and primary_stop != "model_limit" and primary and len(primary) > 5
                and utt.vad_confidence >= 0.85 and utt.end_s - utt.start_s < 4
                and prefetched is None):
            return None
        try:
            if (prefetched is not None and prefetched.route == "japanese"
                    and is_usable_transcript(prefetched.text)):
                secondary = prefetched.text
                secondary_elapsed_s = prefetched.elapsed_s
                log.debug("Reusing Whisper Turbo auto-Japanese transcript at %.2fs", utt.start_s)
            else:
                started = time.perf_counter()
                secondary = worker.transcribe(utt.audio, timeout=secondary_budget)
                secondary_elapsed_s = time.perf_counter() - started
            log.info("Whisper Turbo ASR %.2f-%.2fs audio decoded in %.3fs",
                     utt.start_s, utt.end_s, secondary_elapsed_s)
        except Exception as exc:
            log.warning("Secondary ASR failed at %.2fs: %s", utt.start_s, exc)
            self.release()
            return None
        if current and current() != utt.generation:
            return ""
        log.debug("ASR generation=%d %.3f-%.3f Haas=%r Whisper Turbo=%r",
                  utt.generation, utt.start_s, utt.end_s, primary, secondary)
        def record(selected_text, selected_by, score=None, flags=None, selector_called=False):
            append_comparison({
                "session_id": self.session_id,
                "generation": utt.generation,
                "start_s": round(utt.start_s, 3),
                "end_s": round(utt.end_s, 3),
                "duration_s": round(utt.end_s - utt.start_s, 3),
                "haas": primary,
                "whisper_turbo": secondary,
                "detected_language": getattr(prefetched, "language", None),
                "language_route": ("japanese_reconciled" if getattr(prefetched, "route", None) == "japanese"
                                   else "uncertain_japanese_fallback" if prefetched is not None
                                   else "japanese_forced"),
                "chosen": selected_text,
                "chosen_by": selected_by,
                "haas_elapsed_s": (round(primary_elapsed_s, 3)
                                   if primary_elapsed_s is not None else None),
                "whisper_turbo_elapsed_s": round(secondary_elapsed_s, 3),
                "similarity": round(score, 4) if score is not None else None,
                "flags": flags or [],
                "selector_called": selector_called,
                "haas_stop": primary_stop,
                "whisper_turbo_stop": worker.last_stop,
                "vad_confidence": utt.vad_confidence,
            })

        if not is_usable_transcript(secondary):
            if not is_usable_transcript(primary):
                record("", "both_unusable")
                return ""
            record(primary, "haas_secondary_unusable")
            return None
        result = primary if is_usable_transcript(primary) else secondary
        selected_by = "haas" if is_usable_transcript(primary) else "whisper_turbo_rescue"
        selector_called = False
        if (primary_stop == "model_limit" and worker.last_stop == "complete"
                and secondary.startswith(primary) and len(secondary) > len(primary)):
            result = secondary
            selected_by = "whisper_turbo_completion"
        terms = self.glossary.find(primary + " " + secondary)
        score, flags = compare(primary, secondary, terms)
        if score < 1 and primary:
            log.info("ASR disagreement %.3fs similarity=%.3f flags=%s", utt.start_s, score, flags)
            timeout = min(end - time.monotonic(), self.cfg.reconciler.live_timeout_s
                          if self.live else self.cfg.reconciler.vod_timeout_s)
            if selector and timeout >= 0.25 and not (self.live and backlog()):
                try:
                    result = selector.choose(primary, secondary, {
                        "title_channel": title[:200], "recent_ja": context,
                        "glossary": terms[:12], "vad_probability": utt.vad_confidence,
                        "haas_stop": primary_stop, "secondary_stop": worker.last_stop,
                        "similarity": score, "disagreement": flags,
                    }, timeout)
                    selector_called = True
                    selected_by = getattr(selector, "last_choice", None)
                    if selected_by not in ("haas", "whisper_turbo", "uncertain"):
                        selected_by = "whisper_turbo" if result == secondary else "haas"
                except Exception as exc:
                    log.warning("Selector fallback at %.2fs: %s", utt.start_s, exc)
        from .quality_guard import transcript_problem, disagreement_unresolved
        issue = transcript_problem(result, utt.end_s - utt.start_s)
        if issue or disagreement_unresolved(primary, secondary, result, score, flags, selected_by):
            log.warning("Suppressing unreliable dual ASR %.2fs: %s", utt.start_s, issue or "unresolved_disagreement")
            record("", "abstain_" + (issue or "disagreement"), score, flags, selector_called)
            return ""
        record(result, selected_by, score, flags, selector_called)
        return result
