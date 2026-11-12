"""Offline contract tests for the separate NeMo-Speech.cpp ASR process."""
import io
import json
import threading
import unittest
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import Mock, patch

import numpy as np

from auto_translate.asr.nemotron import NemotronASR, _multipart, _wav16
from auto_translate.asr.reconciler import MODEL


class NemotronPayloadTests(unittest.TestCase):
    def test_wav16_preserves_pcm_rate_and_mono(self):
        data = _wav16(np.array([0., 0.5, -0.5], dtype=np.float32))
        with wave.open(io.BytesIO(data), "rb") as audio:
            self.assertEqual(audio.getframerate(), 16000)
            self.assertEqual(audio.getnchannels(), 1)
            self.assertEqual(audio.getsampwidth(), 2)
            self.assertEqual(audio.getnframes(), 3)

    def test_rejects_corrupt_audio(self):
        for values in (np.array([np.nan]), np.array([[1., 2.]])):
            with self.assertRaises(ValueError):
                _wav16(values)

    def test_hardcode_japanese_language_and_wav(self):
        body, boundary = _multipart(_wav16(np.ones(1600, dtype=np.float32) * 0.1))
        self.assertIn(b'filename="utterance.wav"', body)
        self.assertIn(b'ja-JP', body)
        self.assertIn(b'Content-Type: audio/wav', body)
        self.assertTrue(body.endswith(("--" + boundary + "--\r\n").encode("ascii")))

    def test_reconciler_is_japanese_lfm_not_old_qwen(self):
        self.assertEqual(MODEL, "LiquidAI/LFM2.5-1.2B-JP-202606-GGUF:Q4_K_M")

    def test_transcribe_http_roundtrip(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                content = self.rfile.read(int(self.headers["Content-Length"]))
                requests.append((self.path, self.headers["Content-Type"], content))
                encoded = json.dumps({"text": "まだ行かない"}, ensure_ascii=False).encode("utf8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *args):
                pass

        with HTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.handle_request, daemon=True)
            thread.start()
            worker = object.__new__(NemotronASR)
            worker.url = f"http://127.0.0.1:{server.server_port}"
            worker.process = Mock()
            worker.process.poll.return_value = None
            worker.last_stop = "unknown"
            self.assertEqual(worker.transcribe(np.zeros(3200, dtype=np.float32), timeout=2),
                             "まだ行かない")
            thread.join(2)

        self.assertEqual(requests[0][0], "/v1/audio/transcriptions")
        self.assertIn("multipart/form-data; boundary=", requests[0][1])
        self.assertIn(b'ja-JP', requests[0][2])
        self.assertIn(b'RIFF', requests[0][2])
        self.assertEqual(worker.last_stop, "unknown")

    def test_server_exit_fails_without_fake_result(self):
        worker = object.__new__(NemotronASR)
        worker.process = Mock()
        worker.process.poll.return_value = 1
        with self.assertRaises(RuntimeError):
            worker.transcribe(np.ones(1600, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
