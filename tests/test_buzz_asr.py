from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from auto_translate.asr import create_asr
from auto_translate.config import AsrConfig


class BuzzAdapterTests(unittest.TestCase):
    """Exercise the real worker transport with a deterministic stand-in decoder."""

    def start(self, reply, *, replies=None, **options):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        Path(folder.name, "model.safetensors").touch()
        program = (
            "import sys, json, base64, time\n"
            "print(json.dumps({'ready': True}), flush=True)\n"
            "for line in sys.stdin.buffer:\n"
            "    pcm = base64.b64decode(line)\n"
        )
        scripted = iter(replies or [reply])
        def launch(args, **kwargs):
            return subprocess.Popen([sys.executable, "-u", "-c", program + next(scripted)], **kwargs)
        cfg = AsrConfig(model_path=folder.name, **options)
        self.enterContext(patch("auto_translate.asr.buzz.owned_popen", side_effect=launch))
        engine = create_asr(cfg)
        self.addCleanup(engine.close)
        return engine

    def test_pcm_transport_and_accepted_japanese(self):
        audio = np.linspace(-0.2, 0.2, 16000, dtype="<f4")
        # Compare PCM byte-for-byte inside the worker without passing a large command line.
        import hashlib
        digest = hashlib.sha256(audio.tobytes()).hexdigest()
        engine = self.start(
            f"    assert __import__('hashlib').sha256(pcm).hexdigest() == '{digest}'\n"
            "    print(json.dumps({'text': '\\u3084\\u3070\\u3044'}), flush=True)\n"
        )
        self.assertEqual(engine.transcribe(audio), "\u3084\u3070\u3044")
        self.assertEqual(engine.last_raw_transcript, "\u3084\u3070\u3044")
        self.assertIsNone(engine.last_rejection)

    def test_rejections_and_validation(self):
        engine = self.start("    print(json.dumps({'text':'blocked'}),flush=True)\n",
                            blocklist=["blocked"])
        audio = np.ones(16000, np.float32)
        for sample, probe, reason in [(audio, True, "probe_unconfirmed"),
                                      (np.zeros_like(audio), False, "no_speech"),
                                      (audio, False, "blocklist")]:
            self.assertEqual(engine.transcribe(sample, probe), "")
            self.assertEqual(engine.last_rejection, reason)
        for invalid in ([float("nan")], [[0.1]], np.ones(960001)):
            with self.assertRaises(ValueError):
                engine.transcribe(invalid)

    def test_seek_discards_inflight_result(self):
        engine = self.start("    print(json.dumps({'text':'old'}),flush=True)\n")
        receive = engine._receive
        waiting, release = threading.Event(), threading.Event()
        def pause(timeout):
            waiting.set()
            self.assertTrue(release.wait(3))
            return receive(timeout)
        with patch.object(engine, "_receive", side_effect=pause), ThreadPoolExecutor(1) as pool:
            result = pool.submit(engine.transcribe, np.ones(16000))
            self.assertTrue(waiting.wait(3))
            engine.cancel_pending()
            release.set()
            self.assertEqual(result.result(3), "")
            self.assertEqual(engine.last_raw_transcript, "")

    def test_timeout_reaps_worker(self):
        engine = self.start("    time.sleep(10)\n", response_timeout_s=0.05)
        with self.assertRaises(TimeoutError):
            engine.transcribe(np.ones(1600))
        self.assertIsNotNone(engine._proc.poll())

    def test_missing_weights_fail_before_spawn(self):
        with tempfile.TemporaryDirectory() as folder, patch("auto_translate.asr.buzz.owned_popen") as spawn:
            with self.assertRaisesRegex(FileNotFoundError, "Setup.bat"):
                create_asr(AsrConfig(model_path=folder))
            spawn.assert_not_called()

    def test_artifact_cleanup_preserves_raw_and_cleaned_transcripts(self):
        engine = self.start("    print(json.dumps({'text':'*SGigh* ありがとう'}),flush=True)\n")
        self.assertEqual(engine.transcribe(np.ones(16000)), 'ありがとう')
        self.assertEqual(engine.last_raw_transcript, '*SGigh* ありがとう')
        self.assertEqual(engine.last_cleaned_transcript, 'ありがとう')
        engine.cancel_pending()
        self.assertEqual(engine.last_cleaned_transcript, '')

    def test_cuda_error_restarts_in_fresh_process_and_retries_audio(self):
        engine = self.start("", replies=[
            "    print(json.dumps({'error':'CUDA error: illegal memory access'}),flush=True)\n"
            "    time.sleep(10)\n",
            "    print(json.dumps({'text':'やばい'}),flush=True)\n",
        ])
        previous = engine._proc
        self.assertEqual(engine.transcribe(np.ones(1600)), "やばい")
        self.assertIsNotNone(previous.poll())
        self.assertNotEqual(engine._proc.pid, previous.pid)
        self.assertEqual(engine._restarts, 1)

    def test_repeated_cuda_error_is_bounded_and_preserves_cause(self):
        error = "    print(json.dumps({'error':'CUDA error: illegal memory access'}),flush=True)\n"
        engine = self.start("", replies=[error, error])
        with self.assertRaisesRegex(RuntimeError, "CUDA error: illegal memory access"):
            engine.transcribe(np.ones(1600))
        self.assertEqual(engine._restarts, 1)

    def test_non_cuda_worker_error_is_not_retried_or_hidden_by_timeout(self):
        engine = self.start("    print(json.dumps({'error':'ValueError: bad model input'}),flush=True)\n")
        with self.assertRaisesRegex(RuntimeError, "ValueError: bad model input"):
            engine.transcribe(np.ones(1600))
        self.assertEqual(engine._restarts, 0)

    def test_seek_during_cuda_failure_recovers_without_replaying_stale_audio(self):
        engine = self.start("", replies=[
            "    print(json.dumps({'error':'CUDA error: illegal memory access'}),flush=True)\n",
            "    print(json.dumps({'text':'やばい'}),flush=True)\n",
        ])
        receive = engine._receive
        waiting, release = threading.Event(), threading.Event()
        def pause_once(timeout):
            if not waiting.is_set():
                waiting.set()
                self.assertTrue(release.wait(3))
            return receive(timeout)
        with patch.object(engine, "_receive", side_effect=pause_once), ThreadPoolExecutor(1) as pool:
            result = pool.submit(engine.transcribe, np.ones(1600))
            self.assertTrue(waiting.wait(3))
            engine.cancel_pending()
            release.set()
            self.assertEqual(result.result(3), "")
        self.assertEqual(engine._restarts, 1)
        self.assertEqual(engine.transcribe(np.ones(1600)), "やばい")

    def test_final_worker_error_reaps_poisoned_process(self):
        # A non-CUDA worker error is not retried; the poisoned worker must be
        # stopped before the error surfaces, not left orphaned until shutdown.
        engine = self.start("    print(json.dumps({'error':'ValueError: bad model input'}),flush=True)\n")
        with self.assertRaisesRegex(RuntimeError, "ValueError: bad model input"):
            engine.transcribe(np.ones(1600))
        self.assertIsNotNone(engine._proc.poll())
        self.assertEqual(engine._restarts, 0)

    def test_repeated_cuda_error_reaps_both_workers(self):
        # After one bounded restart, a second CUDA error must stop the replacement
        # worker instead of leaving it orphaned on the GPU.
        error = "    print(json.dumps({'error':'CUDA error: illegal memory access'}),flush=True)\n"
        engine = self.start("", replies=[error, error])
        with self.assertRaisesRegex(RuntimeError, "CUDA error: illegal memory access"):
            engine.transcribe(np.ones(1600))
        self.assertEqual(engine._restarts, 1)
        self.assertIsNotNone(engine._proc.poll())
