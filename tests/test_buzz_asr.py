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

    def start(self, reply, **options):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        Path(folder.name, "model.safetensors").touch()
        program = (
            "import sys, json, base64, time\n"
            "print(json.dumps({'ready': True}), flush=True)\n"
            "for line in sys.stdin.buffer:\n"
            "    pcm = base64.b64decode(line)\n" + reply
        )
        def launch(args, **kwargs):
            return subprocess.Popen([sys.executable, "-u", "-c", program], **kwargs)
        cfg = AsrConfig(model_path=folder.name, **options)
        with patch("auto_translate.asr.buzz.owned_popen", side_effect=launch):
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
