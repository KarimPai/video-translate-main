"""Protocol/lifecycle tests with a fake HTTP child, not ASR accuracy evidence."""
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from auto_translate.asr import create_asr
from auto_translate.asr.voxtral import (
    VoxtralASR,
    VoxtralError,
    VoxtralOOMError,
    _audio16,
)
from auto_translate.config import AsrConfig
from auto_translate.windows_subprocess import stopping

_FAKE_SERVER = r'''
import json, sys, time, wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
plan = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if plan.get("startup_error"):
    print(plan["startup_error"], flush=True)
    sys.exit(1)
requests = []
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def respond(self, status, value):
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass
    def do_GET(self):
        if self.path == "/health":
            self.respond(200, dict(status="ok", backend=plan.get("backend", "cuda")))
        else:
            self.respond(200, dict(data=[dict(id="voxtral")]))
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with wave.open(body["audio"], "rb") as audio:
            metadata = dict(rate=audio.getframerate(), width=audio.getsampwidth(),
                            channels=audio.getnchannels(), frames=audio.getnframes())
        index = len(requests)
        requests.append(dict(path=self.path, body=body, wav=metadata))
        Path(plan["requests"]).write_text(json.dumps(requests), encoding="utf-8")
        item = plan.get("responses", [dict(text="こんにちは")])[min(index, len(plan.get("responses", [0]))-1)]
        time.sleep(item.get("delay", 0))
        self.respond(item.get("status", 200), item.get("body", dict(text=item.get("text", "こんにちは"))))
HTTPServer((config["host"], config["port"]), Handler).serve_forever()
'''


class VoxtralAdapterTests(unittest.TestCase):
    def setUp(self):
        stopping.clear()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.executable = self.root / "audiocpp_server.exe"
        self.executable.write_bytes(b"fake runtime for lifecycle tests")
        self.model = self.root / "voxtral-mini-4b-realtime-2602-q4_k.gguf"
        self.model.write_bytes(b"GGUFfake model fixture")
        self.script = self.root / "server.py"
        self.script.write_text(_FAKE_SERVER, encoding="utf-8")
        self.requests_path = self.root / "requests.json"
        self.launches = []
        self.addCleanup(stopping.clear)

    def start(self, responses=None, *, response_timeout=2, **plan):
        plan["requests"] = str(self.requests_path)
        if responses is not None:
            plan["responses"] = responses
        plan_path = self.root / "plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        cfg = AsrConfig(voxtral_server_path=str(self.executable),
                        voxtral_model_path=str(self.model),
                        startup_timeout_s=3, response_timeout_s=response_timeout)

        def launch(command, **kwargs):
            self.config = json.loads(Path(command[2]).read_text(encoding="utf-8"))
            process = subprocess.Popen(
                [sys.executable, "-u", str(self.script), command[2], str(plan_path)], **kwargs)
            self.launches.append(process)
            return process

        with patch("auto_translate.asr.voxtral.owned_popen", side_effect=launch):
            adapter = create_asr(cfg)
        self.addCleanup(adapter.close)
        return adapter

    def wait_for_request(self):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if self.requests_path.exists():
                return
            time.sleep(.01)
        self.fail("fake HTTP server did not receive the utterance")

    def test_persistent_offline_server_transcribes_complete_japanese_utterances(self):
        worker = self.start([{"text": "ラプ様、こんにちは。"}, {"text": "え"}])
        process = worker.process
        self.assertIsInstance(worker, VoxtralASR)
        self.assertEqual(worker.transcribe(np.ones(48000, np.float32)), "ラプ様、こんにちは。")
        self.assertEqual(worker.transcribe(np.ones(3200, np.float32)), "え")
        self.assertIs(worker.process, process)
        self.assertEqual(len(self.launches), 1)
        self.assertFalse(self.config["lazy_load"])
        self.assertEqual(self.config["backend"], "cuda")
        entry = self.config["models"][0]
        self.assertEqual((entry["family"], entry["task"], entry["mode"]),
                         ("voxtral_realtime", "asr", "offline"))
        self.assertFalse(entry["default_request_options"]["do_sample"])
        self.assertEqual(entry["default_request_options"]["seed"], 1234)
        requests = json.loads(self.requests_path.read_text(encoding="utf-8"))
        self.assertEqual(requests[0]["path"], "/v1/audio/transcriptions")
        self.assertEqual(set(requests[0]["body"]), {"model", "audio"})
        self.assertEqual(requests[0]["wav"], {"rate": 16000, "width": 2, "channels": 1, "frames": 48000})
        self.assertFalse(Path(requests[0]["body"]["audio"]).exists())

    def test_normalization_filter_and_short_meaningful_reactions(self):
        texts = ["  ラプちゃん。\n", "草", "うん", "Hello and welcome everyone", "あい" * 8, "禁止語"]
        worker = self.start([{"text": text} for text in texts])
        worker.cfg.blocklist = ["禁止語"]
        actual = [worker.transcribe(np.ones(16000, np.float32)) for _ in texts]
        self.assertEqual(actual, ["ラプちゃん。", "草", "うん", "", "", ""])

    def test_raw_transcript_is_preserved_before_normalization_and_loop_trim(self):
        raw = "  ホ\u3099タンちゃんが来た。あいうえお" + "あいうえお" * 8 + "\n"
        worker = self.start([{"text": raw}, {"text": "Botan-chan!"}])
        self.assertEqual(worker.transcribe(np.ones(48000, np.float32)), "ボタンちゃんが来た。")
        self.assertEqual(worker.last_raw_transcript, raw)
        # The adapter keeps only narrow Latin candidates; original-audio and
        # metadata evidence belongs to run_asr's final language gate.
        self.assertEqual(worker.transcribe(np.ones(16000, np.float32)), "Botan-chan!")
        self.assertEqual(worker.last_raw_transcript, "Botan-chan!")
        worker.cancel_pending()
        self.assertEqual(worker.last_raw_transcript, "")

    def test_input_validation_clipping_and_empty(self):
        np.testing.assert_array_equal(_audio16(np.array([-2., -.5, 0, .5, 2.])),
                                      np.array([-32767, -16383, 0, 16383, 32767], dtype="<i2"))
        for audio in (np.ones((2, 2)), np.array([np.nan]), np.array([np.inf])):
            with self.subTest(audio=audio), self.assertRaisesRegex(ValueError, "finite mono"):
                _audio16(audio)
        worker = self.start()
        self.assertEqual(worker.transcribe(np.array([], np.float32)), "")
        self.assertFalse(self.requests_path.exists())

    def test_missing_or_corrupt_model_is_explicit(self):
        cfg = AsrConfig(voxtral_server_path=str(self.executable), voxtral_model_path=str(self.model))
        self.model.unlink()
        with self.assertRaisesRegex(VoxtralError, "checkpoint not found"):
            VoxtralASR(cfg)
        self.model.write_text("download interrupted", encoding="utf-8")
        with self.assertRaisesRegex(VoxtralError, "Invalid Voxtral GGUF"):
            VoxtralASR(cfg)
        cfg.voxtral_server_path = str(self.root / "missing.exe")
        with self.assertRaisesRegex(VoxtralError, "CUDA server not found"):
            VoxtralASR(cfg)

    def test_startup_failure_closes_owned_child_and_reports_runtime_components(self):
        with self.assertRaisesRegex(VoxtralError, "cudart/cuBLAS/cuFFT"):
            self.start(startup_error="could not load ggml-cuda.dll")
        self.assertIsNotNone(self.launches[0].poll())

    def test_cuda_out_of_memory_at_startup_is_explicit(self):
        with self.assertRaisesRegex(VoxtralOOMError, "Reduce PLaMo GPU offloading"):
            self.start(startup_error="cudaMalloc failed: out of memory")
        self.assertIsNotNone(self.launches[0].poll())

    def test_inference_oom_stops_server_without_quality_downgrade(self):
        worker = self.start([{"status": 500, "body": {"error": "CUDA_ERROR_OUT_OF_MEMORY"}}])
        process = worker.process
        with self.assertRaises(VoxtralOOMError):
            worker.transcribe(np.ones(16000, np.float32))
        self.assertIsNotNone(process.poll())
        self.assertTrue(worker.last_failed)
        self.assertEqual(len(self.launches), 1)

    def test_inference_timeout_kills_owned_server(self):
        worker = self.start([{"delay": 2}], response_timeout=.1)
        process = worker.process
        with self.assertRaisesRegex(TimeoutError, "owned server was stopped"):
            worker.transcribe(np.ones(16000, np.float32))
        self.assertIsNotNone(process.poll())

    def test_seek_discards_stale_reply_and_reuses_model_for_next_utterance(self):
        worker = self.start([{"text": "古い字幕", "delay": .2}, {"text": "次の字幕"}])
        received = []
        thread = threading.Thread(target=lambda: received.append(worker.transcribe(np.ones(16000))))
        thread.start()
        self.wait_for_request()
        worker.cancel_pending()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(received, [""])
        self.assertEqual(worker.transcribe(np.ones(16000)), "次の字幕")
        self.assertEqual(len(self.launches), 1)

    def test_seek_during_hung_request_still_retires_server_on_timeout(self):
        worker = self.start([{"delay": 2}], response_timeout=.2)
        process = worker.process
        failures = []

        def transcribe():
            try:
                worker.transcribe(np.ones(16000))
            except (RuntimeError, TimeoutError, OSError) as exc:
                failures.append(exc)

        thread = threading.Thread(target=transcribe)
        thread.start()
        self.wait_for_request()
        worker.cancel_pending()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], TimeoutError)
        self.assertIsNotNone(process.poll())

    def test_close_during_inference_discards_reply_and_preserves_external_child(self):
        worker = self.start([{"delay": 2}])
        external = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
        self.addCleanup(lambda: (external.terminate(), external.wait(timeout=3)))
        received = []
        thread = threading.Thread(target=lambda: received.append(worker.transcribe(np.ones(16000))))
        thread.start()
        self.wait_for_request()
        process = worker.process
        worker.close()
        worker.close()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(received, [""])
        self.assertIsNotNone(process.poll())
        self.assertIsNone(external.poll())

    def test_refuses_non_cuda_readiness(self):
        with self.assertRaisesRegex(VoxtralError, "Unexpected Voxtral CUDA"):
            self.start(backend="cpu")
        self.assertIsNotNone(self.launches[0].poll())


if __name__ == "__main__":
    unittest.main()
