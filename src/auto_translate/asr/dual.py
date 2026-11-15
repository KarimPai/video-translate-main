import logging
import re
import threading
import time
from collections import deque
from concurrent.futures import CancelledError

import numpy as np

from ..windows_subprocess import stopping
from ..asr_correction import lap_spelling_from_peer
from ..text_filter import is_usable_transcript
from .whisper_turbo import WhisperTurboASR
from .diagnostics import append_comparison
from .reconciler import Reconciler
from .selection import clause_differences, compare, scene_repair
from .quality_guard import (transcript_problem, disagreement_unresolved,
                            suspicious_phonetics, uncorroborated_full_name, weak_outro)

log = logging.getLogger(__name__)
SECONDARY = "whisper-large-v3-turbo"


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
        self.closed = False
        self._selector_stop = threading.Event()
        self.failures = 0
        self.retry_at = 0.0
        self._restart_thread = None
        if not cfg.dual_asr.enabled or self.mode == "off":
            return
        try:
            self.worker = self._create_worker()
        except Exception as exc:
            # A cold-start OOM or transient CUDA failure is not permanent.
            self._defer_restart(exc)
            return
        log.info("Secondary %s ready on %s (%s)", SECONDARY,
                 cfg.dual_asr.secondary_device, self.mode)
        if cfg.reconciler.enabled:
            self._start_selector(cfg.reconciler.gpu_layers)

    def _create_worker(self):
        # Admission includes MPV, VAD, model scratch and Windows headroom.
        if self.cfg.dual_asr.secondary_device == "cuda" and not self.budget.admits(1.9):
            raise RuntimeError("insufficient measured GPU headroom for secondary ASR")
        worker = WhisperTurboASR(
            model=self.cfg.dual_asr.secondary_model,
            device=self.cfg.dual_asr.secondary_device,
            compute_type=self.cfg.dual_asr.secondary_compute_type,
            beam_size=self.cfg.dual_asr.secondary_beam_size,
        )
        if self.cfg.dual_asr.secondary_device == "cuda" and not self.budget.admits(0.3):
            worker.close()
            raise RuntimeError("secondary ASR exceeded its startup reservation")
        return worker

    def _defer_restart(self, reason):
        with self.lock:
            if self.closed:
                return
            self.failures += 1
            delay = min(2 ** min(self.failures, 6), 60)
            self.retry_at = time.monotonic() + delay
        log.warning("Secondary ASR unavailable (%s); will retry after %ss (failure %d)",
                    reason, delay, self.failures)

    def _retire_worker(self, worker, reason):
        """Retire only the failed Whisper process, not the selector."""
        with self.lock:
            if self.closed or self.worker is not worker:
                return
            self.worker = None
        # Defer before close so no new worker races the old CUDA teardown.
        self._defer_restart(reason)
        # Close outside lock: multiprocessing join can take seconds on Windows.
        worker.close()

    def _maybe_restart(self):
        """Restart asynchronously: model startup must never block live playback."""
        with self.lock:
            if (self.closed or stopping.is_set() or self.worker is not None or
                    (self._restart_thread and self._restart_thread.is_alive()) or
                    time.monotonic() < self.retry_at):
                return
            self._restart_thread = threading.Thread(
                target=self._restart_worker, name="whisper-turbo-restart", daemon=True)
            self._restart_thread.start()

    def _restart_worker(self):
        try:
            worker = self._create_worker()
        except Exception as exc:
            self._defer_restart(exc)
            return
        with self.lock:
            closed = self.closed
            if not closed:
                self.worker = worker
                self.failures = 0
                self.retry_at = 0.0
        if closed:
            worker.close()
            return
        log.info("Secondary Whisper Turbo recovered; dual ASR enabled")
        if self.cfg.reconciler.enabled and self.selector is None and not self.closed:
            self._start_selector(self.cfg.reconciler.gpu_layers)

    def _start_selector(self, layers):
        # Decimal GB estimates for the 731 MB LFM GGUF, context/compute buffers
        # and MPV/VAD allocations that may arrive after model startup.
        playback_reserve = 0.65
        placements = [0]
        if layers and self.budget.admits(1.15 + playback_reserve):
            placements.insert(0, layers)
        for placement in placements:
            if self._selector_stop.is_set():
                return
            try:
                selector = Reconciler(placement, cancel_event=self._selector_stop)
            except CancelledError:
                return
            except Exception as exc:
                log.warning("Selector startup failed (GPU layers=%d): %s", placement, exc)
                continue
            try:
                if placement and not self.budget.admits(playback_reserve):
                    log.warning("Selector exceeded startup reservation; switching to CPU")
                    continue
                with self.lock:
                    if self.closed:
                        return
                    self.selector = selector
                    selector = None  # Ownership transferred to DualASR.
                log.info("LFM2.5-1.2B-JP selector ready; GPU layers requested=%d", placement)
                return
            finally:
                if selector is not None:
                    selector.close()

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
        """Permanent shutdown, reserved for GPU pressure or session exit."""
        with self.lock:
            self.closed = True
            self._selector_stop.set()
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

    def close(self):
        """Standard lifecycle close alias."""
        self.release()

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

    def log_fallback(self, utt, primary, elapsed, reason):
        append_comparison(dict(stage="asr_fallback", session_id=self.session_id,
                               generation=utt.generation, start_s=round(utt.start_s, 3),
                               end_s=round(utt.end_s, 3), haas=primary,
                               haas_elapsed_s=elapsed, fallback_reason=reason,
                               vad_confidence=utt.vad_confidence))

    def select_segment(self, utt, primary, title, backlog, current, primary_stop=None, primary_elapsed_s=None):
        # Single-flight optional work. A blocked pipe or HTTP read cannot hold
        # the live ASR stage past its deadline or accumulate an executor queue.
        if self.worker is None:
            self._maybe_restart()
            self.log_fallback(utt, primary, primary_elapsed_s, "secondary_restarting")
            return None
        if self.pending and self.pending.is_alive():
            self.log_fallback(utt, primary, primary_elapsed_s, "worker_busy")
            return None
        # Backpressure must not extend an otherwise usable primary transcript.
        # Still allow rescue when Haas missed speech entirely.
        if backlog() and (self.live or is_usable_transcript(primary)):
            self.log_fallback(utt, primary, primary_elapsed_s, "backlog")
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
                result[0] = self.choose(utt, primary, title, backlog, generation, primary_stop, primary_elapsed_s, cancel_event=cancelled)
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
                self.log_fallback(utt, primary, primary_elapsed_s, "deadline_generation_or_backlog")
                return None
        self.pending.join()
        return result[0] if generation() == utt.generation else None

    def choose(self, utt, primary, title, backlog, current, primary_stop=None, primary_elapsed_s=None, cancel_event=None):
        """Return corroborated Japanese, an abstention, or None for the Haas path."""
        worker, selector = self.worker, self.selector
        duration = utt.end_s - utt.start_s

        def fallback(reason):
            self.log_fallback(utt, primary, primary_elapsed_s, reason)
            return None

        if worker is None:
            return fallback("secondary_unavailable")
        with self.lock:
            context = list(self.history) if self.generation == utt.generation else []
        if (utt.probe or utt.vad_confidence is None or utt.vad_confidence < self.cfg.vad.threshold
                or not len(utt.audio) or not np.isfinite(utt.audio).all()
                or np.max(np.abs(utt.audio)) < 1e-5):
            self.log_fallback(utt, primary, primary_elapsed_s, "acoustic_guard")
            return ""
        if self.budget.pressure_level == "critical":
            return fallback("gpu_pressure")
        end = self.deadline(utt)
        secondary_budget = (self.cfg.dual_asr.live_timeout_s if self.live
                            else self.cfg.dual_asr.vod_timeout_s)
        if self.live and (backlog() or time.monotonic() + secondary_budget > end):
            return fallback("live_budget")
        if not self.live and backlog() and is_usable_transcript(primary):
            return fallback("backlog")
        if (self.mode == "selective" and primary_stop != "model_limit" and primary and len(primary) > 5
                and utt.vad_confidence >= 0.85 and utt.end_s - utt.start_s < 4):
            return fallback("selective_mode")
        try:
            started = time.perf_counter()
            secondary = worker.transcribe(utt.audio, timeout=secondary_budget, cancel_event=cancel_event)
            secondary_elapsed_s = time.perf_counter() - started
            log.info("Whisper Turbo ASR %.2f-%.2fs audio decoded in %.3fs",
                     utt.start_s, utt.end_s, secondary_elapsed_s)
        except CancelledError:
            log.debug("Discarded Whisper cue generation=%d", utt.generation)
            return None
        except Exception as exc:
            log.warning("Secondary ASR failed at %.2fs: %s", utt.start_s, exc)
            self._retire_worker(worker, exc)
            return fallback("secondary_failure")
        if (cancel_event is not None and cancel_event.is_set()) or (
                current and current() != utt.generation):
            return ""
        log.debug("ASR generation=%d %.3f-%.3f Haas=%r Whisper Turbo=%r",
                  utt.generation, utt.start_s, utt.end_s, primary, secondary)
        lfm_text = None
        lfm_choice = None
        edits = []
        selector_elapsed = None
        fallback_reason = None

        def record(selected_text, selected_by, score=None, flags=None, selector_called=False):
            append_comparison({
                "stage": "reconciliation",
                "lfm_selected": lfm_text,
                "lfm_choice": lfm_choice,
                "corrected_ja": selected_text if edits else None,
                "correction_evidence": edits,
                "selector_elapsed_s": selector_elapsed,
                "fallback_reason": fallback_reason,
                "session_id": self.session_id,
                "generation": utt.generation,
                "start_s": round(utt.start_s, 3),
                "end_s": round(utt.end_s, 3),
                "duration_s": round(utt.end_s - utt.start_s, 3),
                "haas": primary,
                "whisper_turbo": secondary,
                "detected_language": None,
                "language_route": "japanese_forced",
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

        # A short English credit can be a real name/quote. Prefer a usable
        # Japanese peer only on long audio with this specific discrepancy.
        def suspect_credit(text, peer):
            return (duration >= 3 and re.fullmatch(r"by [a-z]\.", text.strip(), re.I)
                    and re.search(r"[ぁ-ヺ一-龯]", peer) and not transcript_problem(peer, duration))

        for candidate, peer in ((primary, secondary), (secondary, primary)):
            if weak_outro(candidate, peer, duration, utt.vad_confidence, utt.audio):
                mode = self.cfg.dual_asr.japanese_outro_mode
                log.info("Weak Japanese outro at %.2fs, mode=%s, VAD=%.3f", utt.start_s, mode, utt.vad_confidence)
                if mode == "suppress":
                    replacement = peer if not transcript_problem(peer, duration) else ""
                    record(replacement, "weak_japanese_outro")
                    return replacement

        if transcript_problem(secondary, duration) or suspect_credit(secondary, primary):
            if transcript_problem(primary, duration):
                record("", "both_unusable")
                return ""
            fallback_reason = transcript_problem(secondary, duration) or "short_credit_with_japanese_peer"
            record(primary, "haas_secondary_unusable")
            return None
        primary_ok = not transcript_problem(primary, duration) and not suspect_credit(primary, secondary)
        result = primary if primary_ok else secondary
        selected_by = "haas" if primary_ok else "whisper_turbo_rescue"
        selector_called = False
        if (primary_stop == "model_limit" and worker.last_stop == "complete"
                and secondary.startswith(primary) and len(secondary) > len(primary)):
            result = secondary
            selected_by = "whisper_turbo_completion"
        terms = list(dict.fromkeys(self.glossary.find(primary) + self.glossary.find(secondary)))
        score, flags = compare(primary, secondary, terms, self.glossary.scene_pairs)
        if score < 1 and primary_ok and selected_by != "whisper_turbo_completion":
            log.info("ASR disagreement %.3fs similarity=%.3f flags=%s", utt.start_s, score, flags)
            timeout = min(end - time.monotonic(), self.cfg.reconciler.live_timeout_s
                          if self.live else self.cfg.reconciler.vod_timeout_s)
            fallback_reason = "selector_unavailable_or_budget"
            if selector and timeout >= 0.25 and not (self.live and backlog()):
                started = time.perf_counter()
                try:
                    selector_called = True
                    lap_evidence = {}
                    for heard, peer in ((primary, secondary), (secondary, primary)):
                        if lap_spelling_from_peer(heard, peer, recent=context) != heard:
                            lap_evidence["corroborated_nickname_reading"] = peer
                            break
                    selected = selector.choose(primary, secondary, {
                        **lap_evidence, "title_channel": title[:200], "recent_ja": context,
                        "glossary": terms[:12], "vad_probability": utt.vad_confidence,
                        "haas_stop": primary_stop, "secondary_stop": worker.last_stop,
                        "similarity": score, "disagreement": flags,
                        "different_clauses": clause_differences(primary, secondary),
                    }, timeout)
                    if selected not in (primary, secondary):
                        raise ValueError("selector returned an unsupported hypothesis")
                    result = selected
                    fallback_reason = None
                    selected_by = getattr(selector, "last_choice", None)
                    if selected_by not in ("haas", "whisper_turbo", "uncertain"):
                        selected_by = "whisper_turbo" if result == secondary else "haas"
                    lfm_choice = selected_by
                    lfm_text = None if selected_by == "uncertain" else result
                except Exception as exc:
                    fallback_reason = type(exc).__name__
                    log.warning("Selector fallback at %.2fs: %s", utt.start_s, exc)
                finally:
                    selector_elapsed = round(time.perf_counter() - started, 3)
        if current and current() != utt.generation:
            return ""
        if score < 1 and primary_ok and result in (primary, secondary):
            other = secondary if result == primary else primary
            result, edits = scene_repair(result, other, title, context, self.glossary.scene_pairs)
        # A title/glossary match is not acoustic evidence. In a strong dual-ASR
        # disagreement, reject an isolated *full* stage name that only one
        # recognizer heard. Keep valid short-name calls and supported names.
        other = secondary if result == primary else primary
        if (primary_ok and score < 0.45
                and uncorroborated_full_name(result, other, self.glossary.name_guards)):
            flags.append("uncorroborated_full_name")
            fallback_reason = "isolated_full_name_only_one_asr"
            log.warning("Suppressing one-sided uncorroborated full name at %.2fs", utt.start_s)
            record("", "abstain_uncorroborated_full_name", score, flags, selector_called)
            return ""
        # A nickname heard on just one side of a strong disagreement
        # must not disable the weak-phonetics abstention.
        corroborating_terms = any(source not in self.glossary.nicknames
                                  or (source in primary and source in secondary)
                                  for source, _ in terms)
        if suspicious_phonetics(result) and not corroborating_terms and score < .45:
            flags.append("uncorroborated_phonetics")
            if selected_by == "uncertain" and utt.vad_confidence < .6:
                record("", "abstain_weak_phonetics", score, flags, selector_called)
                return ""
        if duration >= 5 and len(result.strip()) <= 3:
            flags.append("short_for_duration")
        issue = transcript_problem(result, utt.end_s - utt.start_s)
        if issue or disagreement_unresolved(primary, secondary, result, score, flags, selected_by):
            log.warning("Suppressing unreliable dual ASR %.2fs: %s", utt.start_s, issue or "unresolved_disagreement")
            record("", "abstain_" + (issue or "disagreement"), score, flags, selector_called)
            return ""
        record(result, selected_by, score, flags, selector_called)
        return result
