import logging
import re
import threading
import time
from collections import deque
from concurrent.futures import CancelledError
from contextlib import nullcontext

import numpy as np

from ..windows_subprocess import stopping
from ..runtime_diagnostics import update_metrics, record_event
from ..asr_correction import lap_spelling_from_peer
from ..text_filter import is_usable_transcript
from .whisper_turbo import WhisperTurboASR
from .diagnostics import append_comparison
from .reconciler import Reconciler
from .selection import (clause_differences, compare, scene_repair, normalize,
                        disagreement_details, targeted_retry_window, retry_is_supported)
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
        self.chat_context = None
        self.session_id = str(time.time_ns())
        self.closed = False
        self._selector_stop = threading.Event()
        self.failures = 0
        self.retry_at = 0.0
        self._restart_thread = None
        self._selector_restart_thread = None
        self._selector_retry_at = 0.
        self._selector_failures = 0
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
        if cfg.reconciler.enabled and getattr(cfg.dual_asr, "smart_correction", True):
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
        self._worker_metrics(worker)
        return worker

    @staticmethod
    def _worker_metrics(worker):
        process = getattr(worker, "process", None) if worker is not None else None
        pid = getattr(process, "pid", None)
        update_metrics(workers={"whisper": {"pid": pid if type(pid) is int else None,
                                           "running": worker is not None,
                                           "owned": True}})

    def _has_gate(self):
        return (callable(getattr(type(self.budget), "operation", None))
                or callable(getattr(self.budget, "__dict__", {}).get("operation")))

    def _defer_restart(self, reason):
        with self.lock:
            if self.closed:
                return
            self.failures += 1
            delay = min(2 ** min(self.failures, 6), 60)
            self.retry_at = time.monotonic() + delay
        log.warning("Secondary ASR unavailable (%s); will retry after %ss (failure %d)",
                    reason, delay, self.failures)
        record_event("worker_restart_deferred", worker="whisper", reason=type(reason).__name__,
                     retry_after_s=delay, failures=self.failures)

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
        self._worker_metrics(None)

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
            operation = (self.budget.operation("whisper_load", cancel_event=self._selector_stop)
                         if self._has_gate() else nullcontext())
            with operation:
                worker = self._create_worker()
        except CancelledError:
            return
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
        record_event("worker_recovered", worker="whisper")
        if (self.cfg.reconciler.enabled and getattr(self.cfg.dual_asr, "smart_correction", True)
                and self.selector is None and not self.closed):
            self._maybe_restart_selector()

    def _maybe_restart_selector(self):
        with self.lock:
            if (self.closed or stopping.is_set() or self.selector is not None
                    or not self.cfg.reconciler.enabled or self.worker is None
                    or not getattr(self.cfg.dual_asr, "smart_correction", True)
                    or self.budget.pressure_level != "normal"
                    or time.monotonic() < self._selector_retry_at
                    or (self._selector_restart_thread and self._selector_restart_thread.is_alive())):
                return

            def restart():
                try:
                    operation = (self.budget.operation("lfm_load", cancel_event=self._selector_stop)
                                 if self._has_gate() else nullcontext())
                    with operation:
                        self._start_selector(self.cfg.reconciler.gpu_layers)
                except CancelledError:
                    return
                except Exception:
                    log.exception("Selector recovery failed; primary ASR continues")
                finally:
                    with self.lock:
                        if self.selector is None:
                            self._selector_failures += 1
                            self._selector_retry_at = time.monotonic() + min(60, 2 ** min(self._selector_failures, 6))
                        else:
                            self._selector_failures = 0
            self._selector_restart_thread = threading.Thread(target=restart, name="lfm-restart", daemon=True)
            self._selector_restart_thread.start()

    def _retire_selector(self, selector, reason):
        with self.lock:
            if self.selector is not selector:
                return
            self.selector = None
            self._selector_failures += 1
            self._selector_retry_at = time.monotonic() + min(60, 2 ** min(self._selector_failures, 6))
        log.warning("Retiring stalled LFM selector (%s); primary ASR continues", reason)
        record_event("worker_timeout", worker="lfm", reason=type(reason).__name__)
        selector.close()

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
            self._worker_metrics(None)
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

    def _operation(self, name, utt, current, cancel_event, deadline):
        # Older integrations and lightweight test budgets have no scheduler.
        if self._has_gate():
            return self.budget.operation(name, cancel_event=cancel_event, current=current,
                                         generation=utt.generation, deadline=deadline)
        return nullcontext()

    def _transcribe_guarded(self, worker, audio, timeout, cancel_event):
        """Keep the GPU admission lease until a timed-out child is stopped."""
        try:
            return worker.transcribe(audio, timeout=timeout, cancel_event=cancel_event)
        except CancelledError:
            if getattr(getattr(self.cfg, "safe_playback", None), "enabled", False):
                self._retire_worker(worker, "cancelled_inflight_safe_playback")
            raise
        except Exception as exc:
            self._retire_worker(worker, exc)
            raise

    def _retry_disagreement(self, utt, primary, secondary, confidence, backlog,
                            current, cancel_event, deadline):
        """Recheck only a timed, low-confidence disputed clause, at most twice.

        Retries cannot invent a third transcript: corroborate one original
        candidate, otherwise abstain and leave existing selection unchanged.
        """
        policy = self.cfg.dual_asr
        if not getattr(policy, "smart_correction", True):
            return None, []
        count = getattr(policy, "correction_live_retries" if self.live else "correction_vod_retries", 0)
        allowance = getattr(policy, "correction_live_budget_s" if self.live else "correction_vod_budget_s", 0.)
        if not count or allowance <= 0 or backlog():
            return None, []
        window = targeted_retry_window(primary, secondary, confidence, len(utt.audio) / 16000)
        if window is None:
            return None, []
        until = min(deadline, time.monotonic() + allowance)
        audio = utt.audio[int(window["start_s"] * 16000):int(window["end_s"] * 16000)]
        attempts = []
        worker = self.worker
        for attempt in range(min(2, count)):
            if (worker is None or self.closed or stopping.is_set() or backlog()
                    or (cancel_event is not None and cancel_event.is_set())
                    or (current is not None and current() != utt.generation)):
                break
            remaining = until - time.monotonic()
            if remaining < .25:
                break
            record = {"attempt": attempt + 1, "start_s": window["start_s"],
                      "end_s": window["end_s"], "reason": "low_confidence_disputed_clause"}
            try:
                with self._operation("whisper_retry", utt, current, cancel_event, until):
                    remaining = until - time.monotonic()
                    if remaining <= 0:
                        raise CancelledError("correction budget exhausted awaiting GPU admission")
                    heard = self._transcribe_guarded(worker, audio, remaining, cancel_event)
                if ((cancel_event is not None and cancel_event.is_set())
                        or (current is not None and current() != utt.generation)
                        or time.monotonic() >= until):
                    record["result"] = "obsolete"
                    attempts.append(record)
                    break
                candidate = None
                if retry_is_supported(getattr(worker, "last_confidence", None)):
                    for label, clause in (("haas", window["haas_clause"]),
                                          ("whisper_turbo", window["whisper_clause"])):
                        if normalize(heard) == normalize(clause):
                            candidate = label
                record["result"] = candidate or "insufficient_acoustic_support"
                attempts.append(record)
                if candidate is not None:
                    return candidate, attempts
            except CancelledError:
                record["result"] = "cancelled"
                attempts.append(record)
                break
            except Exception as exc:
                record["result"] = type(exc).__name__
                attempts.append(record)
                # A failed correction is optional; retain primary translation
                # while retiring a broken worker for bounded async recovery.
                self._retire_worker(worker, exc)
                break
        return None, attempts

    def _chat_terms(self, utt, primary, secondary):
        """Only candidate-present terms reach LFM; raw chat never reaches it."""
        if not callable(self.chat_context):
            return []
        try:
            values = self.chat_context(utt, primary, secondary)
            if not isinstance(values, (list, tuple)):
                return []
            terms = []
            for value in values[:12]:
                term = value.get("term") if isinstance(value, dict) else value
                if (isinstance(term, str) and 2 <= len(term) <= 32
                        and (term in primary or term in secondary) and term not in terms):
                    terms.append(term)
            return terms[:6]
        except Exception:
            log.warning("Chat context unavailable for ASR; continuing without it")
            return []

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
        if self.selector is None:
            self._maybe_restart_selector()
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
        if getattr(self.cfg.dual_asr, "smart_correction", True):
            timeout += getattr(self.cfg.dual_asr, "correction_live_budget_s" if self.live
                               else "correction_vod_budget_s", 0.)
        deadline = time.monotonic() + timeout
        if self.live:
            deadline = min(deadline, self.deadline(utt))
        while not finished.wait(.05):
            if (self.closed or stopping.is_set() or time.monotonic() >= deadline
                    or generation() != utt.generation
                    or (backlog() and (self.live or is_usable_transcript(primary)))):
                cancelled.set()
                self.log_fallback(utt, primary, primary_elapsed_s, "deadline_generation_or_backlog")
                return None
        self.pending.join()
        if time.monotonic() >= deadline:
            cancelled.set()
            self.log_fallback(utt, primary, primary_elapsed_s, "completed_after_deadline")
            return None
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
            request_deadline = min(end, time.monotonic() + secondary_budget)
            with self._operation("whisper", utt, current, cancel_event, request_deadline):
                remaining = request_deadline - time.monotonic()
                if remaining <= 0:
                    raise CancelledError("Whisper deadline exhausted awaiting GPU admission")
                started = time.perf_counter()
                try:
                    secondary = self._transcribe_guarded(worker, utt.audio, remaining, cancel_event)
                finally:
                    secondary_elapsed_s = time.perf_counter() - started
                    update_metrics(stages={"whisper_s": secondary_elapsed_s})
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
        retry_attempts = []
        confidence = getattr(worker, "last_confidence", {})
        confidence = confidence if isinstance(confidence, dict) else {}
        differences = disagreement_details(primary, secondary)

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
                "disagreement_details": differences,
                "whisper_confidence": {key: value for key, value in confidence.items()
                                       if key in ("avg_logprob", "no_speech_prob", "compression_ratio")
                                       and type(value) in (float, int)},
                "targeted_retries": retry_attempts,
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
        retry_choice = None
        if score < 1 and primary_ok and selected_by != "whisper_turbo_completion":
            retry_choice, retry_attempts = self._retry_disagreement(
                utt, primary, secondary, confidence, backlog, current, cancel_event, end)
            if retry_choice is not None:
                result = primary if retry_choice == "haas" else secondary
                selected_by = "targeted_audio_" + retry_choice
        if (score < 1 and primary_ok and selected_by != "whisper_turbo_completion"
                and retry_choice is None and getattr(self.cfg.dual_asr, "smart_correction", True)):
            log.info("ASR disagreement %.3fs similarity=%.3f flags=%s", utt.start_s, score, flags)
            timeout = min(end - time.monotonic(), self.cfg.reconciler.live_timeout_s
                          if self.live else self.cfg.reconciler.vod_timeout_s)
            fallback_reason = "selector_unavailable_or_budget"
            if selector and timeout >= 0.25 and not (self.live and backlog()):
                selector_started = False
                try:
                    lap_evidence = {}
                    for heard, peer in ((primary, secondary), (secondary, primary)):
                        if lap_spelling_from_peer(heard, peer, recent=context) != heard:
                            lap_evidence["corroborated_nickname_reading"] = peer
                            break
                    evidence = {
                        **lap_evidence, "title_channel": title[:200], "recent_ja": context,
                        "glossary": terms[:12], "vad_probability": utt.vad_confidence,
                        "whisper_signals_uncalibrated": {key: value for key, value in confidence.items()
                                                         if key != "segments" and type(value) in (float, int)},
                        "weak_chat_terms": ([] if retry_is_supported(confidence)
                                            else self._chat_terms(utt, primary, secondary)),
                        "haas_stop": primary_stop, "secondary_stop": worker.last_stop,
                        "similarity": score, "disagreement": flags,
                        "different_clauses": clause_differences(primary, secondary),
                    }
                    selector_deadline = min(end, time.monotonic() + timeout)
                    with self._operation("lfm", utt, current, cancel_event, selector_deadline):
                        timeout = selector_deadline - time.monotonic()
                        if timeout <= 0:
                            raise TimeoutError("LFM deadline exhausted awaiting GPU admission")
                        selector_started = True
                        selector_called = True
                        started = time.perf_counter()
                        try:
                            selected = selector.choose(primary, secondary, evidence, timeout,
                                                       cancel_event=cancel_event)
                        except (TimeoutError, OSError, CancelledError) as exc:
                            self._retire_selector(selector, exc)
                            raise
                        finally:
                            selector_elapsed = round(time.perf_counter() - started, 3)
                            update_metrics(stages={"lfm_s": selector_elapsed})
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
                    if selector_started and isinstance(exc, (TimeoutError, OSError)):
                        self._retire_selector(selector, exc)
        if ((current and current() != utt.generation)
                or (cancel_event is not None and cancel_event.is_set())
                or time.monotonic() >= end):
            return ""
        if (score < 1 and primary_ok and result in (primary, secondary)
                and retry_choice is None and getattr(self.cfg.dual_asr, "smart_correction", True)):
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
