"""Mocked speech.cpp Haas worker tests; do not require GPU or model downloads."""
import base64
import io
import json
import queue
import unittest
from unittest.mock import Mock, patch

import numpy as np

from auto_translate.asr import create_asr
from auto_translate.asr.haas import HaasASR
from auto_translate.config import AsrConfig


def stream(*items):
    proc = Mock()
    proc.poll.return_value = None
    proc.stdin = io.StringIO()
    proc.stdout = io.StringIO("".join(
        json.dumps(item, ensure_ascii=False) + "\n"
        for item in ({"type": "ready", "version": "0.10.1"}, *items)
    ))
    return proc


class HaasWorkerTests(unittest.TestCase):
    def setUp(self):
        job = patch("auto_translate.asr.haas._create_kill_job", return_value=None)
        job.start()
        self.addCleanup(job.stop)

    def make(self, proc, cfg=None):
        exe = patch("auto_translate.asr.haas.shutil.which", return_value="speech.exe")
        pop = patch("auto_translate.asr.haas.subprocess.Popen", return_value=proc)
        exe.start()
        pop.start()
        self.addCleanup(exe.stop)
        self.addCleanup(pop.stop)
        return create_asr(cfg or AsrConfig()), pop

    def test_worker_uses_q8_vulkan_and_cpu_decode(self):
        proc = stream({"type": "end", "id": "1", "text": "わため、ありがとう"})
        asr, pop = self.make(proc)
        self.assertIsInstance(asr, HaasASR)
        args = pop.mock_calls[0].args[0]
        self.assertEqual(args[:3], ["speech.exe", "worker", "haas-ja-v1.0:q8_0"])
        self.assertEqual(args[args.index("--device") + 1], "Vulkan0")
        self.assertIn("--decode-on-cpu=true", args)
        self.assertEqual(asr.transcribe(np.ones(16000, dtype=np.float32)), "わため、ありがとう")
        messages = [json.loads(line) for line in proc.stdin.getvalue().splitlines()]
        self.assertEqual([m["type"] for m in messages], ["chunk", "transcribe"])
        self.assertEqual(len(base64.b64decode(messages[0]["pcm"])), 32000)
        self.assertEqual(messages[1]["sample_rate"], 16000)
        self.assertNotIn("prompt", messages[1])
        asr.close()
        asr.close()
        proc.terminate.assert_called_once()

    def test_progress_then_end(self):
        proc = stream({"type": "progress", "id": "1"}, {"type": "end", "id": "1", "text": "大丈夫"})
        asr, _ = self.make(proc)
        self.assertEqual(asr.transcribe(np.ones(8000, dtype=np.float32)), "大丈夫")
        asr.close()

    def test_timeout_is_reported(self):
        asr, _ = self.make(stream(), AsrConfig(response_timeout_s=1))
        with patch.object(asr._messages, "get", side_effect=queue.Empty):
            with self.assertRaisesRegex(TimeoutError, "Haas ASR response timeout"):
                asr.transcribe(np.ones(8000, dtype=np.float32))
        asr.close()

    def test_stdout_closed_is_reported(self):
        asr, _ = self.make(stream())
        with self.assertRaisesRegex(RuntimeError, "closed the stdout pipe"):
            asr.transcribe(np.ones(8000, dtype=np.float32))
        asr.close()

    def test_worker_error_does_not_fallback_to_other_asr(self):
        asr, _ = self.make(stream({"type": "error", "id": "1", "error": "Vulkan failed"}))
        with self.assertRaisesRegex(RuntimeError, "Haas ASR recognition failed"):
            asr.transcribe(np.ones(8000, dtype=np.float32))
        asr.close()

    def test_bad_startup_closes_worker(self):
        proc = stream()
        proc.stdout = io.StringIO('{"type":"fatal","error":"bad model"}\n')
        with patch("auto_translate.asr.haas.shutil.which", return_value="speech.exe"), patch(
            "auto_translate.asr.haas.subprocess.Popen", return_value=proc
        ):
            with self.assertRaisesRegex(RuntimeError, "worker failed to load"):
                HaasASR(AsrConfig())
        proc.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
