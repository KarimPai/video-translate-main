import json
import tempfile
import threading
from concurrent.futures import CancelledError
from pathlib import Path
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
        for letter, label, expected in [("A", "haas", "行く"), ("B", "whisper_turbo", "行け"),
                                        ("U", "uncertain", "行く")]:
            record = selected(letter, "行く", "行け")
            self.assertEqual(record["selection"], label)
            self.assertEqual(record["selected_japanese"], expected)
            self.assertEqual(record["uncertain"], letter == "U")
        for answer in ["", "AB", "A because", "行こう", None, {}]:
            with self.assertRaises(ValueError):
                selected(answer, "行く", "行け")

    def test_thresholds(self):
        for cls, kwargs in [(GpuBudgetConfig, {"startup_headroom_gb": 0.1}),
                            (ReconcilerConfig, {"enabled": "false"}),
                            (DualAsrConfig, {"secondary_live_mode": "often"})]:
            with self.assertRaises(ValueError):
                cls(**kwargs)

    @patch("auto_translate.gpu_budget.subprocess.check_output")
    def test_device_memory(self, output):
        output.return_value = "7000,8192\n"
        self.assertAlmostEqual(used_gb(), 7.340032)
        budget = GpuBudget(GpuBudgetConfig())
        self.assertAlmostEqual(budget.total, 8192 * 1048576 / 1e9)
        self.assertTrue(budget.admits(.1))  # No absolute 7.5 GB cutoff.
        self.assertFalse(budget.admits(1.2))  # Leaves startup headroom.
        output.side_effect = subprocess.TimeoutExpired("nvidia-smi", 2)
        self.assertFalse(budget.admits(0))


class PipelineSelectionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.cfg.dual_asr.enabled = False
        self.budget = Mock(used=4., cfg=self.cfg.gpu_budget)
        self.dual = DualASR(self.cfg, self.budget, Glossary.load(Path(__file__).resolve().parents[1] / "glossary.toml"), False)
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

    def test_legacy_name_hints_never_prompt_whisper(self):
        self.cfg.dual_asr.whisper_name_hints = True
        self.dual.worker.transcribe.return_value = "ラプラス"
        self.assertEqual(
            self.dual.choose(self.utt, "ラプラス", "ホロGTA 3日目",
                             lambda: False, lambda: self.utt.generation),
            "ラプラス",
        )
        self.dual.worker.transcribe.assert_called_once()
        self.assertNotIn("initial_prompt", self.dual.worker.transcribe.call_args.kwargs)

    def test_lap_peer_hint_preserves_uncertain_selection(self):
        self.dual.selector.last_choice = "uncertain"
        self.dual.selector.choose.return_value = "ラップ様どこ"
        result, = self.run_stage("ラップ様どこ", "ラプ様どこ")
        evidence = self.dual.selector.choose.call_args.args[2]
        self.assertEqual(evidence["corroborated_nickname_reading"], "ラプ様どこ")
        self.assertIn(("ラプ様", "Lap-sama"), evidence["glossary"])
        self.assertEqual(result.ja, "ラップ様どこ")

    def test_one_sided_isolated_formal_laplus_is_suppressed(self):
        self.dual.selector.choose.return_value = "ラプラス・ダークネス。"
        self.assertEqual(self.run_stage("無線を確認して", "ラプラス・ダークネス。"), [])
        self.dual.selector.choose.assert_called_once()

    def test_corroborated_short_laplus_is_preserved(self):
        self.dual.selector.choose.return_value = "ラプラス・ダークネス。"
        result, = self.run_stage("ラプラスだよ", "ラプラス・ダークネス。")
        self.assertEqual(result.ja, "ラプラス・ダークネス。")

    def test_whisper_loop_preserves_valid_haas_without_selector(self):
        primary = "大丈夫だろこれのろまって意味なの"
        result, = self.run_stage(primary, "ダイジョブ" * 8)
        self.assertEqual(result.ja, primary)
        self.dual.selector.choose.assert_not_called()

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
        self.budget.pressure_level = "critical"
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
                self.wfile.write(json.dumps({"choices": [{"message": {"content": "B"}}]}).encode())

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
        self.assertEqual(requests[0]["grammar"], "root ::= [ABU]")
        self.assertEqual(requests[0]["max_tokens"], 4)
        self.assertNotIn("chat_template_kwargs", requests[0])
        self.assertEqual(__import__("auto_translate.asr.reconciler", fromlist=["MODEL"]).MODEL,
                         "LiquidAI/LFM2.5-1.2B-JP-202606-GGUF:Q4_K_M")


class RecoveryTests(unittest.TestCase):
    @patch("auto_translate.asr.HaasASR")
    def test_primary_recovers_without_reading_stale_response(self, constructor):
        from auto_translate.asr import PrimaryASR
        first, second, third = Mock(), Mock(), Mock()
        first.transcribe.side_effect = TimeoutError()
        second.transcribe.side_effect = TimeoutError()
        third.transcribe.return_value = "行こう"
        constructor.side_effect = [first, second, third]
        primary = PrimaryASR(Config().asr)
        self.assertEqual(primary.transcribe(None), "")
        self.assertEqual(constructor.call_args.args[0].haas_device, "Vulkan0")
        self.assertEqual(primary.transcribe(None), "")
        self.assertEqual(primary.transcribe(None), "行こう")
        self.assertEqual(constructor.call_args.args[0].haas_device, "cpu")
        first.close.assert_called_once()
        second.close.assert_called_once()
        primary.close()
        third.close.assert_called_once()

    def test_live_backlog_detaches_stalled_request(self):
        import threading
        cfg = Config()
        cfg.dual_asr.enabled = False
        dual = DualASR(cfg, Mock(used=3., cfg=cfg.gpu_budget), Glossary({}), True)
        dual.worker = Mock()
        original_worker = dual.worker
        started, resume = threading.Event(), threading.Event()

        def stalled(audio, timeout, cancel_event=None):
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
            original_worker.transcribe.assert_called_once()
        finally:
            resume.set()
            dual.pending.join(2)
        dual.playback_position = lambda: 13.
        self.assertLess(dual.deadline(utt), time.monotonic())


    @patch("auto_translate.asr.dual.WhisperTurboASR")
    def test_secondary_failure_recovers_without_releasing_selector(self, constructor):
        cfg = Config()
        cfg.dual_asr.enabled = False
        budget = Mock()
        budget.pressure_level = "normal"
        budget.admits.return_value = True
        dual = DualASR(cfg, budget, Glossary({}), False)
        first, replacement, selector = Mock(), Mock(), Mock()
        first.transcribe.side_effect = TimeoutError("transient CUDA hiccup")
        dual.worker, dual.selector = first, selector
        utt = Utterance(1, 2, np.ones(16000, np.float32), vad_confidence=.95)
        self.assertIsNone(dual.choose(utt, "行こう", "", lambda: False, lambda: 0))
        self.assertIsNone(dual.worker)
        self.assertIs(dual.selector, selector)
        first.close.assert_called_once()
        selector.close.assert_not_called()
        self.assertGreater(dual.retry_at, time.monotonic())
        constructor.return_value = replacement
        dual.retry_at = 0
        dual._maybe_restart()
        dual._restart_thread.join(2)
        self.assertIs(dual.worker, replacement)
        self.assertEqual(dual.failures, 0)
        self.assertIs(dual.selector, selector)
        dual.release()
        replacement.close.assert_called_once()
        selector.close.assert_called_once()

    def test_cancelled_reply_is_drained_before_reusing_worker(self):
        from auto_translate.asr.whisper_turbo import WhisperTurboASR
        worker = object.__new__(WhisperTurboASR)
        worker._lock = threading.Lock()
        worker._decode_lock = threading.Lock()
        worker.last_stop = "unknown"
        process, connection = Mock(), Mock()
        worker._request_id = 0
        process.is_alive.return_value = True
        connection.recv.side_effect = [{"type": "transcript", "id": 1, "text": "old"},
                                       {"type": "transcript", "id": 2, "text": "new"}]
        cancelled = threading.Event()
        def poll(_):
            cancelled.set()
            return True
        connection.poll.side_effect = poll
        worker.process, worker.connection = process, connection
        with self.assertRaises(CancelledError):
            worker.transcribe(np.ones(100, np.float32), timeout=25,
                              cancel_event=cancelled)
        self.assertIs(worker.process, process)
        self.assertEqual(worker.transcribe(np.ones(100)), "new")
        process.terminate.assert_not_called()
        connection.close.assert_not_called()

    def test_stale_decode_error_after_cancel_does_not_fail_next_cue(self):
        from auto_translate.asr.whisper_turbo import WhisperTurboASR
        worker = object.__new__(WhisperTurboASR)
        worker._lock = threading.Lock()
        worker._decode_lock = threading.Lock()
        worker._request_id = 0
        worker.process, worker.connection = Mock(), Mock()
        cancelled = threading.Event()
        def cancel_while_polling(_):
            cancelled.set()
            return False
        worker.connection.poll.side_effect = cancel_while_polling
        with self.assertRaises(CancelledError):
            worker.transcribe(np.ones(100), cancel_event=cancelled)
        worker.connection.poll.side_effect = None
        worker.connection.poll.return_value = True
        worker.connection.recv.side_effect = [
            {"type": "error", "id": 1, "message": "obsolete decode failure"},
            {"type": "transcript", "id": 2, "text": "次"},
        ]
        self.assertEqual(worker.transcribe(np.ones(100)), "次")

    def test_diagnostics_shutdown_drains_and_exits(self):
        import subprocess
        import sys
        script = """
from auto_translate.asr import diagnostics as d
from pathlib import Path
import sys
path = Path(sys.argv[1])
for i in range(75):
    d.append_comparison({"cue": i}, path)
d._shutdown_writer()
assert not d._WRITER.is_alive()
assert d._QUEUE.unfinished_tasks == 0
"""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cues.jsonl"
            subprocess.run([sys.executable, "-c", script, str(path)], check=True, timeout=10)
            self.assertEqual([json.loads(line)["cue"] for line in path.read_text().splitlines()],
                             list(range(75)))

    def test_diagnostics_flushes_between_cues(self):
        from auto_translate.asr.diagnostics import append_comparison, flush_comparisons
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "asr-comparison.jsonl"
            append_comparison({"stage": "asr_fallback", "chosen": "行く"}, path)
            self.assertTrue(flush_comparisons(3))
            append_comparison({"stage": "translation", "chosen": "go"}, path)
            self.assertTrue(flush_comparisons(3))
            records = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]
            self.assertEqual([r["stage"] for r in records], ["asr_fallback", "translation"])

class SelectorPlacementTests(unittest.TestCase):
    @patch("auto_translate.asr.dual.Reconciler")
    def test_release_during_startup_closes_unpublished_selector(self, constructor):
        cfg = Config()
        cfg.dual_asr.enabled = False
        dual = DualASR(cfg, Mock(), Glossary({}), False)
        selector = Mock()
        def finish_after_release(layers, *, cancel_event):
            dual.close()
            self.assertTrue(cancel_event.is_set())
            return selector
        constructor.side_effect = finish_after_release
        dual._start_selector(0)
        self.assertIsNone(dual.selector)
        selector.close.assert_called_once()

    @patch("auto_translate.asr.dual.Reconciler")
    def test_full_offload_and_fallback(self, constructor):
        dual = object.__new__(DualASR)
        dual._selector_stop = threading.Event()
        dual.lock = threading.Lock()
        dual.closed = False
        dual.budget = Mock()
        for checks, layers in [([True, True], [99]), ([False], [0]), ([True, False], [99, 0])]:
            with self.subTest(checks=checks):
                constructor.reset_mock()
                dual.budget.admits.side_effect = checks
                dual._start_selector(ReconcilerConfig().gpu_layers)
                self.assertEqual([call.args[0] for call in constructor.call_args_list], layers)
                if len(layers) == 2:
                    constructor.return_value.close.assert_called_once()

    @patch("auto_translate.asr.dual.Reconciler")
    def test_gpu_startup_failure_retries_on_cpu(self, constructor):
        dual = object.__new__(DualASR)
        dual._selector_stop = threading.Event()
        dual.lock = threading.Lock()
        dual.closed = False
        dual.budget = Mock()
        cpu = Mock()
        constructor.side_effect = [RuntimeError("allocation failed"), cpu]
        dual._start_selector(99)
        self.assertIs(dual.selector, cpu)
        self.assertEqual(constructor.call_args.args, (0,))


def _fake_whisper_pipe(connection, cancelled):
    for i in range(4):
        command = connection.recv()
        if i < 3:
            cancelled.set()
        connection.send({"type": "transcript", "id": command["id"], "text": str(i)})
    connection.recv()


class WorkerProcessTests(unittest.TestCase):
    def test_repeated_cancellations_preserve_subprocess_pid(self):
        import multiprocessing
        from auto_translate.asr.whisper_turbo import WhisperTurboASR
        ctx = multiprocessing.get_context("spawn")
        parent, child = ctx.Pipe()
        cancel = ctx.Event()
        adapter = object.__new__(WhisperTurboASR)
        adapter._request_id = 0
        adapter._lock = threading.Lock()
        adapter._decode_lock = threading.Lock()
        adapter.connection = parent
        adapter.process = ctx.Process(target=_fake_whisper_pipe, args=(child, cancel), daemon=True)
        adapter.process.start()
        child.close()
        pid = adapter.process.pid
        try:
            for _ in range(3):
                cancel.clear()
                with self.assertRaises(CancelledError):
                    adapter.transcribe(np.ones(100), timeout=10, cancel_event=cancel)
                self.assertEqual(adapter.process.pid, pid)
                self.assertTrue(adapter.process.is_alive())
            self.assertEqual(adapter.transcribe(np.ones(100)), "3")
        finally:
            adapter.close()
