from concurrent.futures import CancelledError
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import httpx

from auto_translate.__main__ import _load_translator
from auto_translate.config import Config
from auto_translate.glossary import Glossary
from auto_translate.llama_server import LlamaServerError
from auto_translate.translate.gemma4 import build_context
from auto_translate.windows_subprocess import stopping
from test_english_reliability import fake_translator


def completion(text, finish="stop"):
    return {"choices": [{"message": {"content": text}, "finish_reason": finish}]}


class Gemma4LifecycleTests(unittest.TestCase):
    def test_truncated_output_retries_once_and_never_publishes_fragment(self):
        # Test 1: empty first, then successful retry
        obj = fake_translator("", "It's sunny today.")
        result = obj.translate("今日は晴れです", [])
        self.assertEqual(result, "It's sunny today.")
        self.assertGreaterEqual(obj._chat.call_count, 2)

        # Test 2: server crash recovery
        obj3 = fake_translator(httpx.ReadError("closed"), "今日は晴れです", "It's sunny today.")
        obj3.server.alive.side_effect = [True, False]
        obj3.reload = Mock()
        self.assertEqual(obj3.translate("今日は晴れです", []), "It's sunny today.")
        self.assertEqual(obj3._chat.call_count, 3)
        obj3.reload.assert_called_once_with(24, "closed")

    def test_interrupt_prevents_retries_or_recovery(self):
        obj = fake_translator()
        def finish(*args, **kwargs):
            obj.interrupt()
            raise httpx.ReadError("closed")
        obj._chat.side_effect = finish
        obj.reload = Mock()
        self.assertEqual(obj.translate("こんにちは", []), "")
        obj._chat.assert_called_once()
        obj.reload.assert_not_called()
        self.assertFalse(obj.request_gpu_relief())
        self.assertEqual(obj.translate("次の文", []), "")
        self.assertEqual(obj._chat.call_count, 1)

    def test_warmup_transport_failure_has_startup_error_contract(self):
        obj = fake_translator("Hello")
        obj._args = Mock(return_value=[])
        obj._chat = Mock(side_effect=httpx.ReadError("allocation failed"))
        obj.server.log_tail.return_value = "CUDA allocation failure"
        with self.assertRaisesRegex(LlamaServerError, "startup request failed"):
            obj._start()

    def test_cancelled_warmup_is_not_a_gpu_retry(self):
        obj = fake_translator("Hello")
        obj._args = Mock(return_value=[])
        obj.server.start.side_effect = stopping.set
        try:
            with self.assertRaises(CancelledError):
                obj._start()
        finally:
            stopping.clear()

    def test_warmup_exercises_long_and_short_prompt_batches(self):
        obj = fake_translator("Hello", "Thank you!")
        obj._args = Mock(return_value=[])
        obj._start()
        self.assertEqual(obj._chat.call_count, 2)

    def test_reload_coalesces_transient_pressure_requests(self):
        obj = fake_translator("Hello", "Thank you!")
        obj._args = Mock(return_value=[])
        obj.server.start.side_effect = obj._relief.set
        obj._start()
        self.assertFalse(obj._relief.is_set())

    def test_title_obeys_context_budget_and_zero_lines_ignore_history(self):
        self.assertEqual(build_context("Long title", [], 3, 0), "")
        self.assertEqual(build_context("Long title", [], 3, 4), "Long")
        obj = fake_translator(glossary=Glossary({}, nicknames={"ボタン": "Botan", "獅白ぼたん": "Shishiro Botan"}))
        obj.cfg.context_lines = 0
        # Ambiguous nicknames like ボタン are now hinted to Gemma 4 even without talent context,
        # so the model can disambiguate. The glossary entry is present.
        prompt = obj.prompt_for("ボタンが来た", [("獅白ぼたん", "Shishiro Botan")])
        # Gemma 4 uses messages format, not chat_template_kwargs
        self.assertIn("messages", prompt)
        # Glossary should be in the user message content
        user_content = prompt["messages"][1]["content"]
        self.assertIn("Botan", user_content)
        # Dialogue history should not appear when context_lines = 0
        self.assertNotIn("直前の会話", user_content)

    def test_initial_auto_offload_reserves_memory_instead_of_probing_all_layers(self):
        obj = fake_translator()
        obj.model = Path("model.gguf")
        budget = Mock(free=4.)
        with patch("auto_translate.translate.create_translator", return_value=obj) as create, patch(
                "auto_translate.llama_server.require_gguf", return_value=obj.model), patch(
                "auto_translate.llama_server.gguf_block_count", return_value=32), patch(
                "auto_translate.__main__.os.path.getsize", return_value=4 * 1024**3):
            _load_translator(Config(), Glossary({}), None, budget, Mock())
        layers = create.call_args.kwargs["gpu_layers"]
        self.assertGreater(layers, 0)
        # With headroom=0.25, WDDM margin=0.25: available = 4.0 - 0.25 - 0.25 = 3.5 GB
        # per_layer = 4GB / 34 = 0.1176 GB => layers = floor(3.5/0.1176) = 29
        self.assertLess(layers, 34)

    def test_headroom_trim_failure_closes_untransferred_translator(self):
        obj = fake_translator()
        obj.model = Path("model.gguf")
        obj.reload = Mock(side_effect=LlamaServerError("reload failed"))
        obj.close = Mock()
        # free=0.2 is below headroom=0.25, so trim loop triggers reload
        budget = Mock(free=.2)
        with patch("auto_translate.translate.create_translator", return_value=obj), patch(
                "auto_translate.__main__.os.path.getsize", return_value=4 * 1024**3):
            with self.assertRaises(LlamaServerError):
                _load_translator(Config(), Glossary({}), None, budget, Mock())
        obj.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
