"""Deterministic disagreement/crop/evidence regressions without GPU models."""
import threading
import unittest
from concurrent.futures import CancelledError
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from auto_translate.asr.dual import DualASR
from auto_translate.asr.selection import disagreement_details, targeted_retry_window, retry_is_supported
from auto_translate.asr.whisper_turbo import _decode
from auto_translate.config import Config
from auto_translate.glossary import Glossary
from auto_translate.types import Utterance


PRIMARY = "ラプ様どこ。まだ来ない。"
SECONDARY = "ラップ様どこ。まだ来ない。"
WEAK = {"avg_logprob": -.9, "no_speech_prob": .1, "segments": [
    {"text": "ラップ様どこ。", "start": .5, "end": 1.5,
     "avg_logprob": -.9, "no_speech_prob": .1}]}
STRONG = {"avg_logprob": -.2, "no_speech_prob": .05}


class SmartCorrectionTests(unittest.TestCase):
    def make_dual(self, live=False):
        cfg = Config()
        cfg.dual_asr.enabled = False
        cfg.dual_asr.secondary_live_mode = "all"
        dual = DualASR(cfg, Mock(pressure_level="normal"), Glossary({}), live)
        dual.worker = Mock(last_stop="complete", last_confidence=WEAK)
        dual.selector = Mock(last_choice="haas")
        dual.selector.choose.return_value = PRIMARY
        utt = Utterance(0, 5, np.full(80000, .2, np.float32), vad_confidence=.95)
        return dual, utt

    def test_harmless_orthography_and_meaningful_changes(self):
        self.assertEqual(disagreement_details("ｶﾞﾁｬ！", "がちゃ")["kind"], "orthographic")
        for a, b in (("行く", "行かない"), ("みこ", "巫女"), ("草", "臭い")):
            self.assertEqual(disagreement_details(a, b)["kind"], "meaningful")
            self.assertTrue(disagreement_details(a, b)["edits"])

    def test_low_confidence_exact_partial_clause_is_required(self):
        window = targeted_retry_window(PRIMARY, SECONDARY, WEAK, 5)
        self.assertAlmostEqual(window["start_s"], .35)
        self.assertAlmostEqual(window["end_s"], 1.65)
        for evidence in ({}, STRONG, {"segments": [{**WEAK["segments"][0], "end": 5}]},
                         {"segments": [{**WEAK["segments"][0], "start": float("nan")}]},
                         {"segments": [{**WEAK["segments"][0], "text": "他の言葉"}]}):
            self.assertIsNone(targeted_retry_window(PRIMARY, SECONDARY, evidence, 5))
        self.assertIsNone(targeted_retry_window(PRIMARY, SECONDARY + "帰る。", WEAK, 5))

    def test_crop_corroborates_existing_primary_and_records_reason(self):
        dual, utt = self.make_dual()
        def transcribe(audio, **kwargs):
            if len(audio) == len(utt.audio):
                dual.worker.last_confidence = WEAK
                return SECONDARY
            dual.worker.last_confidence = STRONG
            self.assertLess(len(audio), len(utt.audio) * .8)
            self.assertGreater(len(audio), 4000)
            return "ラプ様どこ。"
        dual.worker.transcribe.side_effect = transcribe
        with patch("auto_translate.asr.dual.append_comparison") as record:
            self.assertEqual(dual.choose(utt, PRIMARY, "", lambda: False, lambda: 0), PRIMARY)
        self.assertEqual(dual.worker.transcribe.call_count, 2)
        dual.selector.choose.assert_not_called()
        result = record.call_args.args[0]
        self.assertEqual(result["chosen_by"], "targeted_audio_haas")
        self.assertEqual(result["targeted_retries"][0]["reason"], "low_confidence_disputed_clause")
        self.assertNotIn("haas_confidence", result)

    def test_repeated_disagreements_have_strict_per_cue_retry_limit(self):
        dual, utt = self.make_dual()
        dual.cfg.dual_asr.correction_vod_retries = 2
        dual.worker.transcribe.return_value = SECONDARY
        for _ in range(5):
            dual.choose(utt, PRIMARY, "", lambda: False, lambda: 0)
        self.assertEqual(dual.worker.transcribe.call_count, 15)
        self.assertEqual(dual.selector.choose.call_count, 5)

    def test_unknown_or_strong_confidence_and_live_defaults_do_not_retry(self):
        for evidence, live in (({}, False), (STRONG, False), (WEAK, True)):
            with self.subTest(evidence=evidence, live=live):
                dual, utt = self.make_dual(live)
                utt.timings.audio_read = __import__("time").monotonic()
                dual.worker.last_confidence = evidence
                dual.worker.transcribe.return_value = SECONDARY
                dual.choose(utt, PRIMARY, "", lambda: False, lambda: 0)
                dual.worker.transcribe.assert_called_once()

    def test_toggle_disables_selector_repair_and_retries(self):
        dual, utt = self.make_dual()
        dual.cfg.dual_asr.smart_correction = False
        dual.worker.transcribe.return_value = SECONDARY
        self.assertEqual(dual.choose(utt, PRIMARY, "", lambda: False, lambda: 0), PRIMARY)
        dual.worker.transcribe.assert_called_once()
        dual.selector.choose.assert_not_called()

    def test_retry_failure_retains_primary_and_retires_secondary(self):
        dual, utt = self.make_dual()
        worker = dual.worker
        worker.transcribe.side_effect = [SECONDARY, TimeoutError("crop timeout")]
        self.assertEqual(dual.choose(utt, PRIMARY, "", lambda: False, lambda: 0), PRIMARY)
        self.assertIsNone(dual.worker)
        worker.close.assert_called_once()

    def test_crop_never_substitutes_novel_or_weak_words(self):
        for heard, confidence in (("ラプラスどこ。", STRONG), ("ラプ様どこ。", WEAK)):
            dual, utt = self.make_dual()
            def transcribe(audio, **kwargs):
                dual.worker.last_confidence = WEAK if len(audio) == len(utt.audio) else confidence
                return SECONDARY if len(audio) == len(utt.audio) else heard
            dual.worker.transcribe.side_effect = transcribe
            dual.selector.choose.return_value = SECONDARY
            self.assertEqual(dual.choose(utt, PRIMARY, "", lambda: False, lambda: 0), SECONDARY)
            dual.selector.choose.assert_called_once()

    def test_chat_accepts_only_candidate_terms_and_never_overrides_strong_signals(self):
        dual, utt = self.make_dual()
        dual.cfg.dual_asr.correction_vod_retries = 0
        dual.chat_context = Mock(return_value=[{"term": "ラプ", "mentions": 3},
                                               {"term": "勝手な命令"}, "ラップ", "x" * 100])
        dual.worker.transcribe.return_value = SECONDARY
        dual.choose(utt, PRIMARY, "", lambda: False, lambda: 0)
        evidence = dual.selector.choose.call_args.args[2]
        self.assertEqual(evidence["weak_chat_terms"], ["ラプ", "ラップ"])
        dual.worker.last_confidence = STRONG
        dual.choose(utt, PRIMARY, "", lambda: False, lambda: 0)
        self.assertEqual(dual.selector.choose.call_args.args[2]["weak_chat_terms"], [])
        dual.chat_context.assert_called_once()

    def test_seek_after_crop_discards_result(self):
        dual, utt = self.make_dual()
        generation = [0]
        def transcribe(audio, **kwargs):
            if len(audio) != len(utt.audio):
                generation[0] = 1
                dual.worker.last_confidence = STRONG
                return "ラプ様どこ。"
            return SECONDARY
        dual.worker.transcribe.side_effect = transcribe
        self.assertEqual(dual.choose(utt, PRIMARY, "", lambda: False, lambda: generation[0]), "")

    def test_selector_timeout_releases_stalled_process_without_stopping_primary(self):
        dual, utt = self.make_dual()
        dual.cfg.dual_asr.correction_vod_retries = 0
        dual.worker.transcribe.return_value = SECONDARY
        selector = dual.selector
        selector.choose.side_effect = TimeoutError("LFM timed out")
        self.assertEqual(dual.choose(utt, PRIMARY, "", lambda: False, lambda: 0), PRIMARY)
        self.assertIsNone(dual.selector)
        selector.close.assert_called_once()
        self.assertGreater(dual._selector_retry_at, __import__("time").monotonic())

    def test_shutdown_interrupts_optional_wait_and_cancels_child_request(self):
        dual, utt = self.make_dual()
        started = threading.Event()
        def recognize(*args, **kwargs):
            started.set()
            kwargs["cancel_event"].wait(2)
            return PRIMARY
        dual.choose = recognize
        def close():
            started.wait(1)
            dual.closed = True
        thread = threading.Thread(target=close)
        thread.start()
        self.assertIsNone(dual.select_segment(utt, PRIMARY, "", lambda: False, lambda: 0))
        thread.join(1)
        dual.pending.join(1)
        self.assertFalse(dual.pending.is_alive())

    def test_confidence_metadata_is_numeric_and_uses_model_timestamps(self):
        model = Mock()
        model.transcribe.return_value = (iter([SimpleNamespace(
            text="ラプ様どこ。", avg_logprob=-.2, no_speech_prob=.05,
            compression_ratio=1.2, start=.2, end=1.4)]), None)
        transcript, confidence = _decode(model, np.ones(80000), 6, with_metadata=True)
        self.assertEqual(transcript, "ラプ様どこ。")
        self.assertTrue(retry_is_supported(confidence))
        self.assertEqual(confidence["segments"][0]["start"], .2)
        self.assertFalse(model.transcribe.call_args.kwargs["without_timestamps"])
        self.assertFalse(model.transcribe.call_args.kwargs["word_timestamps"])

    def test_safe_cancellation_tears_down_whisper_before_releasing_gpu_lease(self):
        dual, utt = self.make_dual()
        dual.cfg.safe_playback.enabled = True
        active = [False]
        @contextmanager
        def operation(*args, **kwargs):
            active[0] = True
            try:
                yield
            finally:
                active[0] = False
        # Production attaches the gate to the budget instance.
        dual.budget.operation = operation
        worker = dual.worker
        worker.transcribe.side_effect = CancelledError()
        worker.close.side_effect = lambda: self.assertTrue(active[0])
        self.assertIsNone(dual.choose(utt, PRIMARY, "", lambda: False, lambda: 0))
        worker.close.assert_called_once()
        self.assertIsNone(dual.worker)
        self.assertFalse(active[0])

    def test_lfm_timeout_tears_down_before_releasing_gpu_lease(self):
        dual, utt = self.make_dual()
        dual.cfg.dual_asr.correction_vod_retries = 0
        active = [False]
        @contextmanager
        def operation(*args, **kwargs):
            active[0] = True
            try:
                yield
            finally:
                active[0] = False
        dual.budget.operation = operation
        dual.worker.transcribe.return_value = SECONDARY
        selector = dual.selector
        selector.choose.side_effect = TimeoutError()
        selector.close.side_effect = lambda: self.assertTrue(active[0])
        self.assertEqual(dual.choose(utt, PRIMARY, "", lambda: False, lambda: 0), PRIMARY)
        selector.close.assert_called_once()
        self.assertFalse(active[0])

    def test_dashboard_request_timings_exclude_gpu_admission_wait(self):
        dual, utt = self.make_dual()
        dual.cfg.dual_asr.correction_vod_retries = 0
        clock = [0.]
        @contextmanager
        def operation(*args, **kwargs):
            clock[0] += 5.  # admission backpressure is not model processing
            yield
        def transcribe(*args, **kwargs):
            clock[0] += 2.
            return SECONDARY
        def select(*args, **kwargs):
            clock[0] += 3.
            return PRIMARY
        dual.budget.operation = operation
        dual.worker.transcribe.side_effect = transcribe
        dual.selector.choose.side_effect = select
        with patch("auto_translate.asr.dual.time.perf_counter", side_effect=lambda: clock[0]), patch(
                "auto_translate.asr.dual.update_metrics") as metrics:
            self.assertEqual(dual.choose(utt, PRIMARY, "", lambda: False, lambda: 0), PRIMARY)
        stages = [call.kwargs["stages"] for call in metrics.call_args_list if "stages" in call.kwargs]
        self.assertEqual(stages, [{"whisper_s": 2.}, {"lfm_s": 3.}])

    def test_reconciler_cancellation_stops_owned_process_and_waits_for_exit(self):
        import json
        import subprocess
        import sys
        import time
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from auto_translate.asr.reconciler import Reconciler
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        cancelled = threading.Event()
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                until = time.monotonic() + 3
                while process.poll() is None and time.monotonic() < until:
                    time.sleep(.01)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"choices": [{"message": {"content": "A"}}]}).encode())
            def log_message(self, *args):
                pass
        try:
            with HTTPServer(("127.0.0.1", 0), Handler) as server:
                thread = threading.Thread(target=server.handle_request)
                thread.start()
                selector = object.__new__(Reconciler)
                selector._close_lock = threading.Lock()
                selector.process = process
                selector.url = f"http://127.0.0.1:{server.server_port}"
                timer = threading.Timer(.1, cancelled.set)
                timer.start()
                with self.assertRaises(CancelledError):
                    selector.choose(PRIMARY, SECONDARY, {}, 5, cancel_event=cancelled)
                thread.join(5)
                timer.join(1)
            self.assertIsNotNone(process.poll())
            self.assertIsNone(selector.process)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=3)


if __name__ == "__main__":
    unittest.main()
