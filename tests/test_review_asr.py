"""Regression coverage for model recovery, cancellation and optional startup."""
import threading
import time
import unittest
from concurrent.futures import CancelledError
from unittest.mock import Mock, patch

import numpy as np

from auto_translate.asr import PrimaryASR
from auto_translate.asr.dual import DualASR
from auto_translate.asr.whisper_turbo import WhisperTurboASR
from auto_translate.config import Config
from auto_translate.glossary import Glossary
from auto_translate.types import Utterance


class OptionalStartupTests(unittest.TestCase):
    def test_background_startup_waits_for_loaded_playback_and_cancels_cleanly(self):
        for cancel in (False, True):
            ready, created = threading.Event(), threading.Event()
            worker = Mock()
            cfg = Config()
            cfg.reconciler.enabled = False
            with patch.object(DualASR, "_create_worker", side_effect=lambda: (created.set(), worker)[1]):
                dual = DualASR(cfg, Mock(), Glossary({}), True,
                               start_background=True, startup_event=ready)
                self.assertFalse(created.wait(.03))
                if cancel:
                    dual.close()
                else:
                    ready.set()
                    self.assertTrue(created.wait(1))
                dual._restart_thread.join(1)
                self.assertFalse(dual._restart_thread.is_alive())
                if cancel:
                    self.assertFalse(created.is_set())
                dual.close()

    def test_disabled_and_off_never_restart_on_first_cue(self):
        for enabled, mode in ((False, "all"), (True, "off")):
            with self.subTest(enabled=enabled, mode=mode):
                cfg = Config()
                cfg.dual_asr.enabled = enabled
                cfg.dual_asr.secondary_vod_mode = mode
                with patch.object(DualASR, "_create_worker") as create:
                    dual = DualASR(cfg, Mock(), Glossary({}), False,
                                   start_background=True)
                    self.assertIsNone(dual.select_segment(None, "", "", None, None))
                    dual.start_background()
                    create.assert_not_called()
                    self.assertIsNone(dual._restart_thread)
                    dual.close()

    def test_background_startup_returns_and_shutdown_retires_unpublished_worker(self):
        cfg = Config()
        started, finish = threading.Event(), threading.Event()
        worker = Mock()

        def create(_dual):
            started.set()
            self.assertTrue(finish.wait(2))
            return worker

        with patch.object(DualASR, "_create_worker", create):
            dual = DualASR(cfg, Mock(), Glossary({}), False, start_background=True)
            self.assertTrue(started.wait(1))
            self.assertIsNone(dual.worker)
            dual.close()
            finish.set()
            dual._restart_thread.join(2)
            self.assertFalse(dual._restart_thread.is_alive())
            self.assertIsNone(dual.worker)
            self.assertIsNone(dual.selector)
            worker.close.assert_called_once()

    def test_dead_selector_retires_then_restarts_on_cpu_after_backoff(self):
        cfg = Config()
        cfg.dual_asr.enabled = False
        dual = DualASR(cfg, Mock(pressure_level="normal"), Glossary({}), False)
        cfg.dual_asr.enabled = True
        worker = Mock(last_stop="complete")
        worker.transcribe.return_value = "まだ行かない"
        dead = Mock()
        dead.choose.side_effect = RuntimeError("selector exited")
        dual.worker, dual.selector = worker, dead
        utt = Utterance(1, 2, np.ones(16000, np.float32), vad_confidence=.9)
        with patch("auto_translate.asr.dual.append_comparison"):
            dual.choose(utt, "行く", "", lambda: False, None)
        self.assertIsNone(dual.selector)
        self.assertGreater(dual.selector_retry_at, time.monotonic())
        dead.close.assert_called_once()
        replacement = Mock(gpu_layers=0)
        with patch("auto_translate.asr.dual.Reconciler", return_value=replacement) as create:
            dual._maybe_start_selector()
            create.assert_not_called()
            dual.selector_retry_at = 0
            dual._maybe_start_selector()
            dual._selector_thread.join(2)
            create.assert_called_once_with(0, cancel_event=dual._selector_stop)
            self.assertIs(dual.selector, replacement)
        dual.close()

    def test_preview_skips_when_final_optional_work_is_pending(self):
        cfg = Config()
        cfg.dual_asr.enabled = False
        dual = DualASR(cfg, Mock(), Glossary({}), False)
        cfg.dual_asr.enabled = True
        dual.worker = Mock()
        dual.pending = Mock()
        dual.pending.is_alive.return_value = True
        self.assertIsNone(dual.try_preview(np.ones(100), 0))
        dual.worker.transcribe.assert_not_called()
        dual.close()

    def test_final_cue_cancels_preview_without_retiring_worker(self):
        cfg = Config()
        cfg.dual_asr.enabled = False
        cfg.reconciler.enabled = False
        dual = DualASR(cfg, Mock(pressure_level="normal"), Glossary({}), False)
        cfg.dual_asr.enabled = True
        started = threading.Event()
        worker = Mock(last_stop="complete")

        def decode(_audio, *, timeout, cancel_event):
            if threading.current_thread().name == "preview-test":
                started.set()
                until = time.monotonic() + 1
                while not cancel_event.is_set() and time.monotonic() < until:
                    time.sleep(.005)
                raise CancelledError("final cue requested")
            return "待って"

        worker.transcribe.side_effect = decode
        dual.worker = worker
        preview_result = []
        preview = threading.Thread(target=lambda: preview_result.append(
            dual.try_preview(np.ones(100), 0, timeout=1)), name="preview-test")
        preview.start()
        self.assertTrue(started.wait(1))
        utt = Utterance(1, 2, np.ones(16000, np.float32), vad_confidence=.9)
        with patch("auto_translate.asr.dual.append_comparison"):
            self.assertEqual(dual.select_segment(utt, "待って", "", lambda: False, None), "待って")
        preview.join(1)
        self.assertFalse(preview.is_alive())
        self.assertEqual(preview_result, [None])
        self.assertIs(dual.worker, worker)
        worker.close.assert_not_called()
        dual.close()


class PrimaryRecoveryTests(unittest.TestCase):
    @patch("auto_translate.asr.HaasASR")
    def test_constructor_failure_counts_toward_cpu_fallback(self, create):
        original, cpu = Mock(), Mock(last_stop="complete")
        original.transcribe.side_effect = TimeoutError("decode timeout")
        cpu.transcribe.return_value = "行こう"
        create.side_effect = [original, RuntimeError("Vulkan init failed"), cpu]
        primary = PrimaryASR(Config().asr)
        self.assertEqual(primary.transcribe(None), "")
        self.assertEqual(primary.transcribe(None), "")
        self.assertEqual(primary.failures, 2)
        self.assertEqual(primary.cfg.haas_device, "cpu")
        self.assertEqual(primary.transcribe(None), "行こう")
        self.assertEqual(create.call_args.args[0].haas_device, "cpu")
        self.assertEqual(primary.failures, 0)
        primary.close()


class OutstandingWhisperTests(unittest.TestCase):
    def test_final_waits_boundedly_for_canceled_preview_without_sending(self):
        worker = self.make_worker()
        worker._inflight = {"id": 1, "watchdog": Mock()}
        worker.connection.poll.side_effect = [False, True]
        self.assertTrue(worker.wait_until_idle(.5))
        worker.connection.send.assert_not_called()
        self.assertIsNotNone(worker._inflight)  # caller drains under single-flight lock
        worker.connection.poll.side_effect = None
        worker.connection.poll.return_value = False
        self.assertFalse(worker.wait_until_idle(.02))
        worker.connection.send.assert_not_called()
        worker.close()

    def make_worker(self):
        worker = object.__new__(WhisperTurboASR)
        worker._lock = threading.Lock()
        worker._decode_lock = threading.Lock()
        worker._request_id = 0
        worker._inflight = None
        worker.process, worker.connection = Mock(), Mock()
        worker.process.is_alive.return_value = True
        return worker

    def cancel_during_poll(self, worker, cancel):
        def poll(_timeout):
            cancel.set()
            return False
        worker.connection.poll.side_effect = poll
        with self.assertRaises(CancelledError):
            worker.transcribe(np.ones(100), timeout=5, cancel_event=cancel)

    @patch("auto_translate.asr.whisper_turbo.threading.Timer")
    def test_canceled_job_keeps_watchdog_and_is_drained_before_next_send(self, timer):
        worker = self.make_worker()
        cancel = threading.Event()
        events = []
        worker.connection.send.side_effect = lambda c: events.append(("send", c["id"]))
        self.cancel_during_poll(worker, cancel)
        watchdog = timer.return_value
        watchdog.cancel.assert_not_called()
        self.assertEqual(worker._inflight["id"], 1)
        cancel.clear()
        worker.connection.poll.side_effect = None
        worker.connection.poll.return_value = False
        with self.assertRaises(CancelledError):
            worker.transcribe(np.ones(100), cancel_event=cancel)
        self.assertEqual(worker.connection.send.call_count, 1)
        worker.connection.poll.return_value = True
        replies = iter([dict(type="error", id=1, message="obsolete"),
                        dict(type="transcript", id=2, text="次")])

        def recv():
            result = next(replies)
            events.append(("recv", result["id"]))
            return result

        worker.connection.recv.side_effect = recv
        self.assertEqual(worker.transcribe(np.ones(100)), "次")
        self.assertEqual(events, [("send", 1), ("recv", 1), ("send", 2), ("recv", 2)])
        self.assertIsNone(worker._inflight)
        worker.close()

    @patch("auto_translate.asr.whisper_turbo.threading.Timer")
    def test_canceled_hung_job_is_still_killed_by_hard_deadline(self, timer):
        worker = self.make_worker()
        process, connection = worker.process, worker.connection
        self.cancel_during_poll(worker, threading.Event())
        _, callback = timer.call_args.args
        callback(*timer.call_args.kwargs["args"])
        process.terminate.assert_called_once()
        connection.close.assert_called_once()
        self.assertIsNone(worker.process)
        self.assertIsNone(worker._inflight)
        worker.close()

    def test_busy_worker_does_not_fail_or_retire_other_decode(self):
        worker = self.make_worker()
        worker._decode_lock.acquire()
        try:
            with self.assertRaises(CancelledError):
                worker.transcribe(np.ones(100))
            worker.connection.send.assert_not_called()
            worker.process.terminate.assert_not_called()
        finally:
            worker._decode_lock.release()
            worker.close()


if __name__ == "__main__":
    unittest.main()
