import json
import queue
import subprocess
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from auto_translate.asr.dual import DualASR
from auto_translate.asr.selection import compare, normalize, selected
from auto_translate.config import Config, GpuBudgetConfig, ReconcilerConfig, DualAsrConfig
from auto_translate.glossary import Glossary
from auto_translate.gpu_budget import GpuBudget, used_gb
from auto_translate.pipeline import run_asr
from auto_translate.types import Utterance


class ComparisonTests(unittest.TestCase):
    def test_width_kana_punctuation(self):
        self.assertEqual(normalize("ｶﾞｯﾂ、３回！"), normalize("がっつ 3回"))
        self.assertNotEqual(normalize("3.5回"), normalize("35回"))

    def test_semantic_disagreement(self):
        for a, b, flag in [("行く", "行かない", "negation"), ("２回", "３回", "number"),
                           ("わため", "わたなべ", "glossary"), ("待って", "待ってくれ", "missing phrase")]:
            ratio, flags = compare(a, b, [("わため", "Watame")])
            self.assertLess(ratio, 1)
            self.assertIn(flag, flags)
        for a, b in [("橋", "箸"), ("私が", "私を"), ("え", "ええ")]:
            self.assertLess(compare(a, b, [])[0], 1)

    def test_contract(self):
        good = dict(selection="qwen", selected_japanese="行け", uncertain=False, reason="文脈")
        self.assertEqual(selected(json.dumps(good), "行く", "行け"), "行け")
        for change in [dict(selected_japanese="行こう"), dict(uncertain=0), dict(reason=[]),
                       dict(extra=1), dict(selection="other"), dict(uncertain=True)]:
            with self.assertRaises(ValueError):
                selected(json.dumps(good | change), "行く", "行け")

    def test_thresholds(self):
        for cls, kwargs in [(GpuBudgetConfig, {"maximum_gb": 7.6}),
                            (ReconcilerConfig, {"enabled": "false"}),
                            (DualAsrConfig, {"secondary_live_mode": "often"})]:
            with self.assertRaises(ValueError):
                cls(**kwargs)

    @patch("auto_translate.gpu_budget.subprocess.check_output")
    def test_device_memory(self, output):
        output.return_value = "7000\n"
        self.assertAlmostEqual(used_gb(), 7.340032)
        budget = GpuBudget(GpuBudgetConfig())
        self.assertFalse(budget.admits(.1))
        output.side_effect = subprocess.TimeoutExpired("nvidia-smi", 2)
        self.assertFalse(budget.admits(0))


class PipelineSelectionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.cfg.dual_asr.enabled = False
        self.budget = Mock(used=4., cfg=self.cfg.gpu_budget)
        self.dual = DualASR(self.cfg, self.budget, Glossary({"わため": "Watame"}), False)
        self.dual.worker = Mock()
        self.dual.selector = Mock()
        self.utt = Utterance(2, 3, np.full(16000, .1, np.float32), vad_confidence=.91)
        self.utt.timings.audio_read = time.monotonic()

    def run_stage(self, a, b, generation=None):
        self.dual.worker.transcribe.return_value = b
        primary = Mock()
        primary.transcribe.return_value = a
        iq, oq = queue.Queue(), queue.Queue()
        iq.put(self.utt)
        iq.put(None)
        run_asr(primary, iq, oq, generation, dual=self.dual)
        results = list(oq.queue)
        self.assertIsNone(results.pop())
        return results

    def test_selected_one_cue_unchanged_timestamps_audio(self):
        self.dual.selector.choose.return_value = "まだ行かない"
        result, = self.run_stage("まだ行く", "まだ行かない")
        self.assertEqual((result.ja, result.start_s, result.end_s), ("まだ行かない", 2, 3))
        self.assertIs(self.dual.worker.transcribe.call_args.args[0], self.utt.audio)

    def test_agree_and_silence(self):
        self.assertEqual(self.run_stage("ガチャ！", "がちゃ")[0].ja, "ガチャ！")
        self.dual.selector.choose.assert_not_called()
        self.utt.audio.fill(0)
        self.assertEqual(self.run_stage("ありがとう", "ありがとう"), [])

    def test_probe_never_validates_music(self):
        self.utt.probe = True
        self.assertEqual(self.run_stage("ありがとう", "ありがとう"), [])
        self.dual.worker.transcribe.assert_not_called()

    def test_seek_during_decode(self):
        self.assertEqual(self.run_stage("待て", "待って", Mock(side_effect=[0, 1])), [])
        self.dual.selector.choose.assert_not_called()

    def test_timeout_and_cleanup(self):
        self.dual.worker.transcribe.side_effect = TimeoutError()
        worker, selector = self.dual.worker, self.dual.selector
        self.assertEqual(self.run_stage("待て", "")[0].ja, "待て")
        self.dual.release()
        worker.close.assert_called_once()
        selector.close.assert_called_once()

    def test_live_deadline_and_pressure(self):
        self.dual.live = True
        self.utt.timings.audio_read -= 60
        self.assertEqual(self.run_stage("待て", "待って")[0].ja, "待て")
        self.dual.live = False
        self.budget.used = 7.15
        self.assertEqual(self.run_stage("待て", "待って")[0].ja, "待て")
        self.dual.worker.transcribe.assert_not_called()

    def test_selector_timeout(self):
        self.dual.selector.choose.side_effect = TimeoutError()
        self.assertEqual(self.run_stage("待て", "待って")[0].ja, "待て")


class TransportTests(unittest.TestCase):
    def test_speech_worker_protocol(self):
        import base64
        import sys
        from auto_translate.asr.haas import HaasASR

        script = """
import json, sys
print(json.dumps({'type': 'ready'}), flush=True)
for line in sys.stdin:
    item = json.loads(line)
    if item['type'] == 'chunk':
        pcm = item['pcm']
    elif item['type'] == 'transcribe':
        assert item['sample_rate'] == 16000
        print(json.dumps({'type':'end','id':item['id'],'text':pcm}), flush=True)
"""
        spawn = subprocess.Popen
        def worker_process(cmd, **kwargs):
            self.assertEqual(cmd[-2:], ["--language", "ja"])
            return spawn([sys.executable, "-u", "-c", script], **kwargs)
        with patch("auto_translate.asr.haas.shutil.which", return_value="speech"), patch(
                "auto_translate.asr.haas.subprocess.Popen", side_effect=worker_process):
            worker = HaasASR(Config().asr, language="ja")
            child = worker.process
            try:
                audio = np.array([0., -.5, .5], dtype=np.float32)
                encoded = worker.transcribe(audio)
                self.assertEqual(base64.b64decode(encoded), (audio * 32767).astype("<i2").tobytes())
            finally:
                worker.close()
            self.assertIsNotNone(child.poll())

    def test_selector_http_round_trip(self):
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from auto_translate.asr.reconciler import Reconciler

        requests = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200)
                self.end_headers()
                selection = dict(selection="qwen", selected_japanese="まだ行かない",
                                 uncertain=False, reason="negation")
                self.wfile.write(json.dumps({"choices": [{"message": {
                    "content": json.dumps(selection)}}]}).encode())

            def log_message(self, *args):
                pass

        with HTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.handle_request)
            thread.start()
            selector = object.__new__(Reconciler)
            selector.url = f"http://127.0.0.1:{server.server_port}"
            selector.process = Mock()
            selector.process.poll.return_value = None
            self.assertEqual(selector.choose("行く", "まだ行かない", {}, 2), "まだ行かない")
            thread.join(2)
        self.assertEqual(requests[0]["response_format"]["type"], "json_schema")


class RecoveryTests(unittest.TestCase):
    @patch("auto_translate.asr.HaasASR")
    def test_primary_recovers_without_reading_stale_response(self, constructor):
        from auto_translate.asr import PrimaryASR
        first, second = Mock(), Mock()
        first.transcribe.side_effect = TimeoutError()
        second.transcribe.return_value = "行こう"
        constructor.side_effect = [first, second]
        primary = PrimaryASR(Config().asr)
        self.assertEqual(primary.transcribe(None), "")
        self.assertEqual(primary.transcribe(None), "行こう")
        self.assertEqual(constructor.call_args.args[0].haas_device, "cpu")
        first.close.assert_called_once()
        primary.close()
        second.close.assert_called_once()

    def test_live_backlog_detaches_stalled_request(self):
        import threading
        cfg = Config()
        cfg.dual_asr.enabled = False
        dual = DualASR(cfg, Mock(used=3., cfg=cfg.gpu_budget), Glossary({}), True)
        dual.worker = Mock()
        started, resume = threading.Event(), threading.Event()

        def stalled(audio, timeout):
            started.set()
            resume.wait(2)
            return "まだ行かない"
        dual.worker.transcribe.side_effect = stalled
        utt = Utterance(10, 12, np.ones(32000), vad_confidence=.8)
        utt.timings.audio_read = time.monotonic()
        try:
            self.assertIsNone(dual.select_segment(utt, "行く", "", started.is_set, lambda: 0))
            self.assertTrue(started.is_set())
            self.assertTrue(dual.pending.is_alive())
            self.assertIsNone(dual.select_segment(utt, "行く", "", lambda: False, lambda: 0))
            dual.worker.transcribe.assert_called_once()
        finally:
            resume.set()
            dual.pending.join(2)
        dual.playback_position = lambda: 13.
        self.assertLess(dual.deadline(utt), time.monotonic())
