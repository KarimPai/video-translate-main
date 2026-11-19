from concurrent.futures import CancelledError
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import httpx

from auto_translate.__main__ import _load_translator
from auto_translate.config import Config
from auto_translate.glossary import Glossary
from auto_translate.llama_server import LlamaServerError
from auto_translate.translate.plamo import build_context
from auto_translate.windows_subprocess import stopping
from test_english_reliability import fake_translator


def completion(text, finish="stop"):
    return {"choices": [{"message": {"content": text}, "finish_reason": finish}]}


class PlamoLifecycleTests(unittest.TestCase):
    def test_truncated_output_retries_once_and_never_publishes_fragment(self):
        for retry, expected in ((completion("It's sunny today."), "It's sunny today."),
                                (completion("It's a very", "length"), "")):
            with self.subTest(expected=expected):
                obj = fake_translator()
                del obj._chat
                obj.server.chat.side_effect = [completion("It's a", "length"), retry]
                self.assertEqual(obj.translate("今日は晴れです", []), expected)
                self.assertEqual(obj.server.chat.call_count, 2)
                self.assertTrue(obj.server.chat.call_args.args[0]["cache_prompt"])
        obj = fake_translator(httpx.ReadError("closed"), "今日は晴れです", "It's sunny today.")
        obj.server.alive.side_effect = [True, False]
        obj.reload = Mock()
        self.assertEqual(obj.translate("今日は晴れです", []), "It's sunny today.")
        self.assertEqual(obj._chat.call_count, 3)
        obj.reload.assert_called_once_with(24, "closed")

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
        obj = fake_translator(httpx.ReadError("allocation failed"))
        obj._args = Mock(return_value=[])
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
        prompts = [call.args[0]["chat_template_kwargs"]["context"] for call in obj._chat.call_args_list]
        self.assertEqual(len(prompts), 2)
        self.assertGreater(len(prompts[0]), len(prompts[1]) * 3)

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
        # Ambiguous nicknames like ボタン are now hinted to PLaMo even without talent context,
        # so the model can disambiguate. The glossary entry is present.
        prompt = obj.prompt_for("ボタンが来た", [("獅白ぼたん", "Shishiro Botan")])
        self.assertIn("glossary", prompt["chat_template_kwargs"])
        self.assertIn({"source": "ボタン", "target": "Botan"}, prompt["chat_template_kwargs"]["glossary"])
        # Dialogue history should not appear in context when context_lines = 0.
        self.assertNotIn("直前の会話", prompt["chat_template_kwargs"].get("context", ""))

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
        self.assertLess(layers, 24)

    def test_headroom_trim_failure_closes_untransferred_translator(self):
        obj = fake_translator()
        obj.model = Path("model.gguf")
        obj.reload = Mock(side_effect=LlamaServerError("reload failed"))
        obj.close = Mock()
        budget = Mock(free=.5)
        with patch("auto_translate.translate.create_translator", return_value=obj), patch(
                "auto_translate.__main__.os.path.getsize", return_value=4 * 1024**3):
            with self.assertRaises(LlamaServerError):
                _load_translator(Config(), Glossary({}), None, budget, Mock())
        obj.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
