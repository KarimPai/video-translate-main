import logging
import threading
import time
from collections import deque

import numpy as np

from ..text_filter import is_usable_transcript
from .nemotron import NemotronASR
from .reconciler import Reconciler
from .selection import compare

log = logging.getLogger(__name__)
SECONDARY = "nemotron-3.5"


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
        if not cfg.dual_asr.enabled or self.mode == "off":
            return
        try:
            # Admission includes MPV, VAD, model scratch and Windows headroom.
            if not budget.admits(1.7):
                raise RuntimeError("insufficient measured GPU headroom for secondary ASR")
            self.worker = NemotronASR(device="cuda:0")
            if not budget.admits(0.5):
                raise RuntimeError("secondary ASR exceeded its startup reservation")
            log.info("Secondary %s ready on CUDA0 (%s)", SECONDARY, self.mode)
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

    def select_segment(self, utt, primary, title, backlog, current, primary_stop=None):
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
                result[0] = self.choose(utt, primary, title, backlog, generation, primary_stop)
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

    def choose(self, utt, primary, title, backlog, current, primary_stop=None):
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
        used = self.budget.used
        if used is None or used >= self.budget.cfg.optimization_threshold_gb:
            return None
        end = self.deadline(utt)
        secondary_budget = (self.cfg.dual_asr.live_timeout_s if self.live
                            else self.cfg.dual_asr.vod_timeout_s)
        if self.live and (backlog() or time.monotonic() + secondary_budget > end):
            return None
        if not self.live and backlog() and is_usable_transcript(primary):
            return None
        if (self.mode == "selective" and primary_stop != "model_limit" and primary and len(primary) > 5
                and utt.vad_confidence >= 0.85 and utt.end_s - utt.start_s < 4):
            return None
        try:
            secondary = worker.transcribe(utt.audio, timeout=secondary_budget)
        except Exception as exc:
            log.warning("Secondary ASR failed at %.2fs: %s", utt.start_s, exc)
            self.release()
            return None
        if current and current() != utt.generation:
            return ""
        log.debug("ASR generation=%d %.3f-%.3f Haas=%r Nemotron=%r",
                  utt.generation, utt.start_s, utt.end_s, primary, secondary)
        if not is_usable_transcript(secondary):
            return None
        result = primary if is_usable_transcript(primary) else secondary
        if (primary_stop == "model_limit" and worker.last_stop == "complete"
                and secondary.startswith(primary) and len(secondary) > len(primary)):
            result = secondary
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
                except Exception as exc:
                    log.warning("Selector fallback at %.2fs: %s", utt.start_s, exc)
        return result
