import json
from pathlib import Path
import struct
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import httpx

from auto_translate.llama_server import LlamaServer, LlamaServerError, gguf_block_count, require_gguf
from auto_translate.windows_subprocess import stopping


class GgufTests(unittest.TestCase):
    def test_block_count_skips_strings_and_arrays(self):
        def string(value):
            data = value.encode()
            return struct.pack("<Q", len(data)) + data
        data = struct.pack("<4sIQQ", b"GGUF", 3, 0, 3)
        data += string("general.architecture") + struct.pack("<I", 8) + string("qwen2")
        data += string("tokenizer.tokens") + struct.pack("<IIQ", 9, 8, 2) + string("a") + string("b")
        data += string("qwen2.block_count") + struct.pack("<II", 4, 32)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.gguf"
            path.write_bytes(data)
            self.assertEqual(gguf_block_count(path), 32)
            self.assertEqual(require_gguf(str(path), "test"), path.resolve())
            path.write_bytes(b"bad")
            self.assertIsNone(gguf_block_count(path))
            with self.assertRaises(LlamaServerError):
                require_gguf(str(path), "test")

    def test_missing_gguf_is_actionable(self):
        with self.assertRaisesRegex(LlamaServerError, "Setup.bat"):
            require_gguf("missing-test-model.gguf", "test")


class ServerProtocolTests(unittest.TestCase):
    def server(self):
        obj = LlamaServer("test", Path("llama-server.exe"), [], startup_timeout_s=2)
        self.addCleanup(obj.close)
        obj.client.close()
        return obj

    def test_chat_uses_private_port_and_propagates_http_failure(self):
        obj = self.server()
        obj.port = 18888
        requests = []
        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={"choices": [{"message": {"content": "Hello"}}]})
        obj.client = httpx.Client(transport=httpx.MockTransport(respond), trust_env=False)
        self.assertEqual(obj.chat({"messages": []}, 1)["choices"][0]["message"]["content"], "Hello")
        self.assertEqual(str(requests[0].url), "http://127.0.0.1:18888/v1/chat/completions")
        self.assertEqual(json.loads(requests[0].content), {"messages": []})
        obj.client.close()
        obj.client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(503)), trust_env=False)
        with self.assertRaises(httpx.HTTPStatusError):
            obj.chat({}, 1)

    def test_malformed_payloads_are_rejected(self):
        for payload in ({"choices": []}, {"choices": [1]}, {"choices": {}},
                        {"choices": [{"message": {}}]}, {"choices": [{"message": {"content": None}}]}):
            with self.subTest(payload=payload):
                obj = self.server()
                obj.client = httpx.Client(transport=httpx.MockTransport(
                    lambda _, payload=payload: httpx.Response(200, json=payload)), trust_env=False)
                with self.assertRaises(LlamaServerError):
                    obj.chat({}, 1)

    def test_gui_stop_interrupts_a_blocking_warmup_request(self):
        obj = self.server()
        obj.client = Mock()
        obj.client.get.return_value.status_code = 200
        entered, released = threading.Event(), threading.Event()
        errors = []
        process = Mock(poll=Mock(return_value=None))
        process.terminate.side_effect = released.set
        def post(*args, **kwargs):
            entered.set()
            if not released.wait(2):
                raise AssertionError("owned request was not interrupted")
            raise httpx.ReadError("server terminated")
        obj.client.post.side_effect = post
        def request():
            try:
                obj.chat({}, 180)
            except Exception as exc:
                errors.append(exc)
        with tempfile.TemporaryDirectory() as folder:
            obj.log_path = Path(folder) / "server.log"
            with patch("auto_translate.llama_server.owned_popen", return_value=process):
                obj.start()
            worker = threading.Thread(target=request)
            try:
                worker.start()
                self.assertTrue(entered.wait(2))
                stopping.set()
                worker.join(2)
                self.assertFalse(worker.is_alive())
                process.terminate.assert_called_once()
                self.assertIsInstance(errors[0], httpx.ReadError)
            finally:
                stopping.clear()
                obj.close()
                worker.join(2)

    def test_close_only_terminates_owned_process_and_is_idempotent(self):
        obj = self.server()
        obj.client = Mock()
        process = Mock(poll=Mock(return_value=None))
        obj.process = process
        obj.close()
        obj.close()
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=3)
        obj.client.close.assert_called_once()
        self.assertFalse(obj.alive())
        with self.assertRaises(LlamaServerError):
            obj.start()

    def test_startup_adds_loopback_host_and_health_checks(self):
        obj = self.server()
        obj.client = Mock()
        obj.client.get.return_value.status_code = 200
        process = Mock(poll=Mock(return_value=None))
        with tempfile.TemporaryDirectory() as folder:
            obj.log_path = Path(folder) / "server.log"
            with patch("auto_translate.llama_server.owned_popen", return_value=process) as spawn, patch(
                    "auto_translate.llama_server._free_port", return_value=18888):
                obj.start()
            command = spawn.call_args.args[0]
            self.assertEqual(command[-4:], ["--host", "127.0.0.1", "--port", "18888"])
            obj.client.get.assert_called_once_with("http://127.0.0.1:18888/health", timeout=.5)
            obj.close()


if __name__ == "__main__":
    unittest.main()
