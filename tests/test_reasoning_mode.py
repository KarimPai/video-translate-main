"""Tests for Gemma 4 reasoning mode configuration and content safety."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Ensure project root is on sys.path
_root = (Path(__file__).resolve().parent.parent)
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from auto_translate.config import TranslateConfig


class ReasoningConfigTests(unittest.TestCase):
    """Verify reasoning configuration fields and validation."""

    def test_reasoning_disabled_by_default(self):
        cfg = TranslateConfig()
        self.assertFalse(cfg.reasoning_enabled)
        self.assertEqual(cfg.reasoning_budget_tokens, 96)

    def test_reasoning_enabled_accepts_valid_values(self):
        cfg = TranslateConfig(reasoning_enabled=True, reasoning_budget_tokens=96)
        self.assertTrue(cfg.reasoning_enabled)
        self.assertEqual(cfg.reasoning_budget_tokens, 96)

    def test_reasoning_budget_must_be_positive_when_enabled(self):
        with self.assertRaisesRegex(ValueError, "reasoning_budget_tokens must be positive"):
            TranslateConfig(reasoning_enabled=True, reasoning_budget_tokens=0)

    def test_reasoning_budget_accepts_zero_when_disabled(self):
        # Zero budget is fine when reasoning is off
        cfg = TranslateConfig(reasoning_enabled=False, reasoning_budget_tokens=0)
        self.assertFalse(cfg.reasoning_enabled)
        self.assertEqual(cfg.reasoning_budget_tokens, 0)

    def test_reasoning_budget_range_validation(self):
        with self.assertRaisesRegex(ValueError, "reasoning_budget_tokens must be an integer in 0..1024"):
            TranslateConfig(reasoning_budget_tokens=1025)
        with self.assertRaisesRegex(ValueError, "reasoning_budget_tokens must be an integer in 0..1024"):
            TranslateConfig(reasoning_budget_tokens=-1)

    def test_reasoning_total_tokens_must_fit_in_context(self):
        # max_tokens + reasoning_budget_tokens must leave room for input
        with self.assertRaisesRegex(ValueError, "must leave room"):
            TranslateConfig(
                reasoning_enabled=True,
                max_tokens=1024,
                reasoning_budget_tokens=1024,
                context_size=2048,
            )

    def test_reasoning_enabled_type_validation(self):
        # The boolean check in __post_init__ catches non-bool types
        with self.assertRaises(ValueError):
            TranslateConfig(reasoning_enabled=1)

    def test_reasoning_with_384_max_tokens(self):
        """Verify the recommended reasoning config: max_tokens=384, budget=96."""
        cfg = TranslateConfig(
            reasoning_enabled=True,
            max_tokens=384,
            reasoning_budget_tokens=96,
        )
        self.assertTrue(cfg.reasoning_enabled)
        self.assertEqual(cfg.max_tokens, 384)
        self.assertEqual(cfg.reasoning_budget_tokens, 96)
        # Total output budget: 384 + 96 = 480 tokens
        self.assertEqual(cfg.max_tokens + cfg.reasoning_budget_tokens, 480)

    def test_non_reasoning_config_unchanged(self):
        """Ensure existing non-reasoning config still works."""
        cfg = TranslateConfig(
            max_tokens=128,
            gpu_layers=42,
            context_size=4096,
        )
        self.assertFalse(cfg.reasoning_enabled)
        self.assertEqual(cfg.max_tokens, 128)


class ReasoningArgsTests(unittest.TestCase):
    """Verify _args() produces correct llama-server flags for reasoning mode."""

    def _make_translator(self, reasoning_enabled=False, **kwargs):
        """Create a minimal Gemma4Translator mock for testing _args()."""
        from auto_translate.translate.gemma4 import Gemma4Translator
        from auto_translate.config import TranslateConfig
        from auto_translate.glossary import Glossary
        from auto_translate.stream_context import StreamContext

        cfg = TranslateConfig(
            reasoning_enabled=reasoning_enabled,
            **kwargs,
        )
        glossary = Glossary({})
        stream = StreamContext()

        # Create a partial object to test _args() without starting the server
        translator = object.__new__(Gemma4Translator)
        translator.cfg = cfg
        translator.glossary = glossary
        translator.stream = stream
        translator.gpu_layers = 42
        translator.layer_count = 42
        translator.model = Path("models/gemma4/test.gguf")
        translator.executable = Path(".runtime/llama-server.exe")
        translator._closed = True  # prevent any server operations
        return translator

    def test_args_reasoning_off_by_default(self):
        translator = self._make_translator()
        args = translator._args()
        # Find --reasoning flag and its value
        reasoning_idx = args.index("--reasoning")
        self.assertEqual(args[reasoning_idx + 1], "off")

    def test_args_reasoning_on_when_enabled(self):
        translator = self._make_translator(reasoning_enabled=True)
        args = translator._args()
        reasoning_idx = args.index("--reasoning")
        self.assertEqual(args[reasoning_idx + 1], "on")
        # Reasoning-specific flags should be present
        self.assertIn("--reasoning-budget", args)
        budget_idx = args.index("--reasoning-budget")
        self.assertEqual(args[budget_idx + 1], "96")

    def test_args_reasoning_budget_custom(self):
        translator = self._make_translator(
            reasoning_enabled=True,
            reasoning_budget_tokens=64,
        )
        args = translator._args()
        budget_idx = args.index("--reasoning-budget")
        self.assertEqual(args[budget_idx + 1], "64")

    def test_args_no_reasoning_flags_when_off(self):
        translator = self._make_translator(reasoning_enabled=False)
        args = translator._args()
        self.assertNotIn("--reasoning-budget", args)


class ReasoningContentSafetyTests(unittest.TestCase):
    """Ensure reasoning_content is NEVER used as subtitle output."""

    def test_chat_uses_content_only(self):
        """Verify _chat extracts only 'content', never 'reasoning_content'."""
        from auto_translate.translate.gemma4 import Gemma4Translator
        from auto_translate.config import TranslateConfig
        from auto_translate.glossary import Glossary
        from auto_translate.stream_context import StreamContext

        cfg = TranslateConfig(reasoning_enabled=True, max_tokens=128)
        glossary = Glossary({})
        stream = StreamContext()

        translator = object.__new__(Gemma4Translator)
        translator.cfg = cfg
        translator.glossary = glossary
        translator.stream = stream
        translator._closed = False
        translator._lock = MagicMock()
        translator._lock.__enter__ = MagicMock(return_value=None)
        translator._lock.__exit__ = MagicMock(return_value=None)

        # Mock the server to return reasoning_content but empty content
        mock_server = MagicMock()
        mock_server.chat.return_value = {
            "choices": [{
                "message": {
                    "content": "",
                    "reasoning_content": "This is internal reasoning that should NOT become a subtitle",
                },
                "finish_reason": "stop",
            }],
            "usage": {"reasoning_tokens": 50},
        }
        translator.server = mock_server

        # _chat should return empty string, NOT the reasoning content
        result = translator._chat({"messages": []}, timeout=1.0)
        self.assertEqual(result, "")
        self.assertNotIn("reasoning", result)

    def test_chat_prefers_content_over_reasoning(self):
        """When both content and reasoning_content exist, only content is returned."""
        from auto_translate.translate.gemma4 import Gemma4Translator
        from auto_translate.config import TranslateConfig
        from auto_translate.glossary import Glossary

        cfg = TranslateConfig(reasoning_enabled=True, max_tokens=128)
        glossary = Glossary({})

        translator = object.__new__(Gemma4Translator)
        translator.cfg = cfg
        translator.glossary = glossary
        translator._closed = False
        translator._lock = MagicMock()
        translator._lock.__enter__ = MagicMock(return_value=None)
        translator._lock.__exit__ = MagicMock(return_value=None)

        mock_server = MagicMock()
        mock_server.chat.return_value = {
            "choices": [{
                "message": {
                    "content": "Hello, how are you?",
                    "reasoning_content": "The user said こんにちは, which means hello",
                },
                "finish_reason": "stop",
            }],
            "usage": {"reasoning_tokens": 10},
        }
        translator.server = mock_server

        result = translator._chat({"messages": []}, timeout=1.0)
        self.assertEqual(result, "Hello, how are you?")
        self.assertNotIn("reasoning", result)
        self.assertNotIn("こんにちは", result)

    def test_non_reasoning_mode_ignores_reasoning_field(self):
        """When reasoning is off, reasoning_content in response is ignored."""
        from auto_translate.translate.gemma4 import Gemma4Translator
        from auto_translate.config import TranslateConfig
        from auto_translate.glossary import Glossary

        cfg = TranslateConfig(reasoning_enabled=False, max_tokens=128)
        glossary = Glossary({})

        translator = object.__new__(Gemma4Translator)
        translator.cfg = cfg
        translator.glossary = glossary
        translator._closed = False
        translator._lock = MagicMock()
        translator._lock.__enter__ = MagicMock(return_value=None)
        translator._lock.__exit__ = MagicMock(return_value=None)

        mock_server = MagicMock()
        mock_server.chat.return_value = {
            "choices": [{
                "message": {
                    "content": "Good morning!",
                    "reasoning_content": "This should be ignored",
                },
                "finish_reason": "stop",
            }],
        }
        translator.server = mock_server

        result = translator._chat({"messages": []}, timeout=1.0)
        self.assertEqual(result, "Good morning!")

    def test_max_tokens_includes_reasoning_budget(self):
        """When reasoning is on, max_tokens sent to API includes the budget."""
        from auto_translate.translate.gemma4 import Gemma4Translator
        from auto_translate.config import TranslateConfig
        from auto_translate.glossary import Glossary

        cfg = TranslateConfig(
            reasoning_enabled=True,
            max_tokens=384,
            reasoning_budget_tokens=96,
        )
        glossary = Glossary({})

        translator = object.__new__(Gemma4Translator)
        translator.cfg = cfg
        translator.glossary = glossary
        translator._closed = False
        translator._lock = MagicMock()
        translator._lock.__enter__ = MagicMock(return_value=None)
        translator._lock.__exit__ = MagicMock(return_value=None)

        mock_server = MagicMock()
        mock_server.chat.return_value = {
            "choices": [{
                "message": {"content": "Test"},
                "finish_reason": "stop",
            }],
        }
        translator.server = mock_server

        translator._chat({"messages": []}, timeout=1.0)

        # Verify the body sent to the server has max_tokens = 384 + 96 = 480
        call_args = mock_server.chat.call_args
        body = call_args[0][0]
        self.assertEqual(body["max_tokens"], 480)


class LlamaServerReasoningTests(unittest.TestCase):
    """Verify llama_server.chat() handles reasoning mode responses."""

    def test_chat_accepts_empty_content_with_reasoning(self):
        """When reasoning mode is active, empty content with reasoning is accepted."""
        from auto_translate.llama_server import LlamaServer, LlamaServerError

        server = object.__new__(LlamaServer)
        server.name = "test"
        server.port = 8080
        mock_client = MagicMock()
        mock_client.post.return_value.json.return_value = {
            "choices": [{
                "message": {
                    "content": "",
                    "reasoning_content": "Internal reasoning",
                },
                "finish_reason": "stop",
            }],
        }
        mock_client.post.return_value.raise_for_status = MagicMock()
        server.client = mock_client

        result = server.chat({"messages": []}, timeout=1.0)
        self.assertIn("choices", result)

    def test_chat_rejects_completely_empty_message(self):
        """A message with no content AND no reasoning is rejected."""
        from auto_translate.llama_server import LlamaServer, LlamaServerError

        server = object.__new__(LlamaServer)
        server.name = "test"
        server.port = 8080
        mock_client = MagicMock()
        mock_client.post.return_value.json.return_value = {
            "choices": [{
                "message": {},
                "finish_reason": "stop",
            }],
        }
        mock_client.post.return_value.raise_for_status = MagicMock()
        server.client = mock_client

        with self.assertRaises(LlamaServerError):
            server.chat({"messages": []}, timeout=1.0)


if __name__ == "__main__":
    unittest.main()
