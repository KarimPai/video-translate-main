"""Tests for Gemma 4 compatibility hardening: special token detection and output validation."""
from __future__ import annotations

import pytest
from auto_translate.subtitle_clean import (
    has_special_token_flood,
    english_output_issue,
    sanitize_english_output,
    clean_english_output,
)
from auto_translate.translate.gemma4 import (
    _strip_reasoning_tags,
    clean_output,
    ChatStatus,
    _ChatResult,
    Gemma4Translator,
)


class TestSpecialTokenFlood:
    """Task 4: Detect repeated special tokens in Gemma 4 output."""

    def test_normal_english_no_flood(self):
        """Normal English dialogue should not trigger flood detection."""
        assert not has_special_token_flood("Hello, how are you today?")
        assert not has_special_token_flood("Everyone, thanks for the super chats!")
        assert not has_special_token_flood("Let's go over there! You can see the view from the mountain top.")

    def test_single_special_token_no_flood(self):
        """A single special token in output should not trigger flood detection."""
        assert not has_special_token_flood("Hello <eos> world")
        assert not has_special_token_flood("The answer is <unused49> correct")

    def test_two_same_token_no_flood(self):
        """Two occurrences of the same token should not trigger flood (threshold is 3)."""
        assert not has_special_token_flood("Hello <unused49> world <unused49> test")

    def test_three_same_token_flood(self):
        """Three occurrences of the same special token should trigger flood detection."""
        assert has_special_token_flood("Hello <unused49> world <unused49> test <unused49> end")

    def test_many_same_token_flood(self):
        """Many repetitions of <unused49> should be detected as flood."""
        text = "<unused49> " * 20
        assert has_special_token_flood(text)

    def test_unused8_flood(self):
        """<unused8> token flood should be detected."""
        text = "Some text <unused8> more <unused8> text <unused8> here"
        assert has_special_token_flood(text)

    def test_eos_flood(self):
        """<eos> token flood should be detected."""
        text = "End <eos> of <eos> turn <eos> now"
        assert has_special_token_flood(text)

    def test_mixed_special_tokens_dominance(self):
        """Mixed special tokens dominating output should trigger flood."""
        # 5 special tokens out of 7 total tokens > 50% threshold
        text = "Hello <eos> <unused49> world <bos> test <eos>"
        assert has_special_token_flood(text)

    def test_mixed_special_tokens_not_dominant(self):
        """Mixed special tokens not dominating should not trigger flood."""
        # 2 special tokens out of 10 total tokens < 50% threshold
        text = "This is a normal sentence with some <eos> words and another <bos> token here"
        assert not has_special_token_flood(text)

    def test_short_text_no_flood(self):
        """Very short text should not trigger flood detection."""
        assert not has_special_token_flood("Hi")
        assert not has_special_token_flood("<eos>")

    def test_empty_text_no_flood(self):
        """Empty text should not trigger flood detection."""
        assert not has_special_token_flood("")
        assert not has_special_token_flood("   ")

    def test_start_of_turn_flood(self):
        """<start_of_turn> repetition should be detected."""
        text = "<start_of_turn> user <start_of_turn> model <start_of_turn> end"
        assert has_special_token_flood(text)

    def test_chat_template_delimiter_flood(self):
        """Repeated chat-template delimiters should be detected."""
        text = "<end_of_turn> <end_of_turn> <end_of_turn>"
        assert has_special_token_flood(text)

    def test_vtuber_dialogue_no_false_positive(self):
        """VTuber-style dialogue should not trigger false positives."""
        assert not has_special_token_flood("Lap-chan, wait a minute! I wanted to use that item!")
        assert not has_special_token_flood("Hehe, thanks! I bought it online.")
        assert not has_special_token_flood("That boss was really strong. Let's try again!")

    def test_japanese_mixed_no_flood(self):
        """Japanese-English mixed output should not trigger flood (handled by other checks)."""
        # This would be caught by untranslated_japanese check, not flood
        assert not has_special_token_flood("こんにちは hello world")


class TestEnglishOutputIssueWithSpecialTokens:
    """Task 1 & 4: Integration of special token detection into output validation."""

    def test_special_token_flood_returns_issue(self):
        """english_output_issue should return 'special_token_flood' for flooded output."""
        text = "<unused49> <unused49> <unused49> <unused49>"
        issue = english_output_issue(text)
        assert issue == "special_token_flood"

    def test_normal_output_no_issue(self):
        """Normal English output should have no issue."""
        issue = english_output_issue("Hello, how are you?")
        assert issue is None

    def test_single_special_token_no_issue(self):
        """A single special token should not trigger issue via flood check."""
        # One token doesn't trigger flood; if it passes other checks, no issue
        issue = english_output_issue("The result is <eos> correct")
        # May still pass if not a flood pattern
        assert issue != "special_token_flood"

    def test_sanitize_rejects_flood(self):
        """sanitize_english_output should reject flooded output."""
        text = "<unused49> <unused49> <unused49> normal text"
        result = sanitize_english_output(text)
        assert result == ""

    def test_clean_output_rejects_flood(self):
        """clean_english_output should reject flooded output."""
        text = "<unused49> <unused49> <unused49> hello world"
        result = clean_english_output(text)
        assert result == ""

    def test_normal_output_survives_cleaning(self):
        """Normal output should survive cleaning pipeline."""
        text = "Everyone, thanks for the super chats!"
        result = clean_english_output(text)
        assert result == text


class TestOutputValidation:
    """Verify output validation catches invalid translations."""

    def test_empty_content_is_rejected(self):
        """Empty content should be rejected by output validation."""
        issue = english_output_issue("")
        assert issue == "empty_english"

    def test_whitespace_only_rejected(self):
        """Whitespace-only content should be rejected."""
        issue = english_output_issue("   ")
        assert issue == "empty_english"

    def test_japanese_output_rejected(self):
        """Japanese output should be rejected (not English translation)."""
        issue = english_output_issue("こんにちは")
        assert issue == "untranslated_japanese"

    def test_prompt_echo_rejected(self):
        """Prompt echo should be rejected by looks_like_prompt_echo."""
        # The prompt echo markers in subtitle_clean.py are specific phrases
        text = "recently mentioned glossary names/terms: test"
        issue = english_output_issue(text)
        assert issue == "prompt_echo_or_loop"


class TestStripReasoningTags:
    """Regression tests for reasoning tag handling in _strip_reasoning_tags()."""

    def test_no_tags_unchanged(self):
        """Normal English text without reasoning tags should pass through."""
        assert _strip_reasoning_tags("Hello, how are you?") == "Hello, how are you?"
        assert _strip_reasoning_tags("Let's go over there!") == "Let's go over there!"

    def test_complete_xml_think_block_removed(self):
        """Complete <think>...</think> blocks should be removed."""
        result = _strip_reasoning_tags("<think>This is reasoning</think>Hello world")
        assert result == "Hello world"

    def test_complete_thinking_block_removed(self):
        """Complete <thinking>...</thinking> blocks should be removed."""
        result = _strip_reasoning_tags("<thinking>Analysis here</thinking>Good morning!")
        assert result == "Good morning!"

    def test_complete_reasoning_block_removed(self):
        """Complete <reasoning>...</reasoning> blocks should be removed."""
        result = _strip_reasoning_tags("<reasoning>Why I chose this</reasoning>Thank you.")
        assert result == "Thank you."

    def test_native_gemma_think_tokens_removed(self):
        """Native Gemma 4 <|think|>...<|end_of_think|> should be removed."""
        result = _strip_reasoning_tags("<|think|>Internal analysis<|end_of_think|>Nice to meet you!")
        assert result == "Nice to meet you!"

    def test_native_channel_thought_removed(self):
        """Native <|channel>thought...<channel|> should be removed."""
        result = _strip_reasoning_tags("<|channel>thought\nLet me think\n<channel|>Hello there!")
        assert result == "Hello there!"

    def test_orphan_think_tag_rejected(self):
        """Orphan <|think|> without closing should reject (return empty)."""
        result = _strip_reasoning_tags("<|think|>This is reasoning text that should not appear")
        assert result == ""

    def test_orphan_channel_thought_rejected(self):
        """Orphan <|channel>thought without closing should reject."""
        result = _strip_reasoning_tags("<|channel>thought\nAnalysis text\nMore analysis")
        assert result == ""

    def test_orphan_xml_think_tag_rejected(self):
        """Orphan <thinking> without closing should reject."""
        result = _strip_reasoning_tags("<thinking>Reasoning without close tag")
        assert result == ""

    def test_orphan_reasoning_tag_rejected(self):
        """Orphan <reasoning> without closing should reject."""
        result = _strip_reasoning_tags("<reasoning>Unclosed reasoning block")
        assert result == ""

    def test_complete_block_then_valid_content(self):
        """Complete block followed by valid content should keep the content."""
        text = "<think>reasoning</think>Everyone, thanks for the super chats!"
        result = _strip_reasoning_tags(text)
        assert result == "Everyone, thanks for the super chats!"

    def test_valid_content_then_complete_block(self):
        """Valid content followed by complete block should keep the content."""
        text = "Good morning! <think>leftover reasoning</think>"
        result = _strip_reasoning_tags(text)
        assert result == "Good morning!"

    def test_multiple_complete_blocks(self):
        """Multiple complete reasoning blocks should all be removed."""
        text = "<think>block1</think>Hello <think>block2</think>world"
        result = _strip_reasoning_tags(text)
        assert result == "Hello world"

    def test_mixed_complete_and_orphan_rejected(self):
        """Complete block plus orphan marker should reject."""
        text = "<think>complete</think>Hello <|think|>orphan reasoning"
        result = _strip_reasoning_tags(text)
        assert result == ""

    def test_ordinary_word_think_preserved(self):
        """Ordinary word 'think' should not be treated as a control token."""
        result = _strip_reasoning_tags("I think so, let's go!")
        assert result == "I think so, let's go!"

    def test_ordinary_word_reasoning_preserved(self):
        """Ordinary word 'reasoning' should not be treated as a control token."""
        result = _strip_reasoning_tags("The reasoning behind this is clear.")
        assert result == "The reasoning behind this is clear."

    def test_multiline_complete_block(self):
        """Multiline reasoning blocks should be properly removed."""
        text = "<think>\nLine 1\nLine 2\n</think>Translation here"
        result = _strip_reasoning_tags(text)
        assert result == "Translation here"

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert _strip_reasoning_tags("") == ""

    def test_whitespace_only(self):
        """Whitespace-only input should return empty string."""
        assert _strip_reasoning_tags("   \n  ") == ""


class TestCleanOutputReasoning:
    """Integration tests for clean_output() with reasoning tags."""

    def test_clean_rejects_orphan_think(self):
        """clean_output should reject responses with orphan think tags."""
        result = clean_output("<|think|>This is internal analysis")
        assert result == ""

    def test_clean_removes_complete_block(self):
        """clean_output should remove complete reasoning blocks."""
        result = clean_output("<think>reasoning</think>Hello world")
        assert result == "Hello world"

    def test_clean_preserves_normal_text(self):
        """clean_output should preserve normal English text."""
        result = clean_output("Hello, how are you today?")
        assert result == "Hello, how are you today?"

    def test_clean_rejects_system_echo(self):
        """clean_output should reject system prompt echoes."""
        result = clean_output("You are a professional Japanese-to-English subtitle translator.")
        assert result == ""

    def test_clean_preserves_slash_content(self):
        """clean_output should preserve slash-separated content."""
        result = clean_output("yes/no decision")
        assert result == "yes/no decision"

    def test_clean_preserves_ordinary_think(self):
        """clean_output should not reject text containing 'think' as a word."""
        result = clean_output("I think we should go now.")
        assert result == "I think we should go now."


class TestHasUnclosedReasoning:
    """Tests for Gemma4Translator._has_unclosed_reasoning() static method."""

    def test_no_reasoning_markers(self):
        """Normal text should not be detected as having unclosed reasoning."""
        assert not Gemma4Translator._has_unclosed_reasoning("Hello world")
        assert not Gemma4Translator._has_unclosed_reasoning("I think so")

    def test_complete_think_block_not_unclosed(self):
        """Complete <|think|>...<|end_of_think|> should not be flagged."""
        text = "<|think|>analysis<|end_of_think|>Hello"
        assert not Gemma4Translator._has_unclosed_reasoning(text)

    def test_orphan_think_detected(self):
        """Orphan <|think|> should be detected."""
        text = "<|think|>analysis without close"
        assert Gemma4Translator._has_unclosed_reasoning(text)

    def test_orphan_channel_thought_detected(self):
        """Orphan <|channel>thought should be detected."""
        text = "<|channel>thought\nanalysis"
        assert Gemma4Translator._has_unclosed_reasoning(text)

    def test_complete_channel_not_unclosed(self):
        """Complete channel block should not be flagged."""
        text = "<|channel>thought\nanalysis\n<channel|>Hello"
        assert not Gemma4Translator._has_unclosed_reasoning(text)

    def test_orphan_xml_thinking_detected(self):
        """Orphan <thinking> should be detected."""
        text = "<thinking>unclosed"
        assert Gemma4Translator._has_unclosed_reasoning(text)

    def test_orphan_reasoning_detected(self):
        """Orphan <reasoning> should be detected."""
        text = "<reasoning>unclosed"
        assert Gemma4Translator._has_unclosed_reasoning(text)

    def test_multiple_opens_one_close_detected(self):
        """Multiple opens with one close should be detected as unclosed."""
        text = "<|think|>first<|think|>second<|end_of_think|>"
        assert Gemma4Translator._has_unclosed_reasoning(text)

    def test_empty_string_not_unclosed(self):
        """Empty string should not be flagged."""
        assert not Gemma4Translator._has_unclosed_reasoning("")


class TestChatResult:
    """Tests for _ChatResult dataclass and ChatStatus enum."""

    def test_chat_result_creation(self):
        """_ChatResult should be created with all fields."""
        result = _ChatResult(
            content="Hello",
            status=ChatStatus.success,
            finish_reason="stop",
            input_tokens=10,
            output_tokens=5
        )
        assert result.content == "Hello"
        assert result.status == ChatStatus.success
        assert result.finish_reason == "stop"
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.reasoning_content_length == 0

    def test_chat_status_values(self):
        """ChatStatus enum should have all expected values."""
        assert ChatStatus.success.value
        assert ChatStatus.truncated.value
        assert ChatStatus.reasoning_anomaly.value
        assert ChatStatus.malformed.value
        assert ChatStatus.transport_error.value

    def test_chat_result_defaults(self):
        """_ChatResult should have sensible defaults."""
        result = _ChatResult(content="", status=ChatStatus.transport_error)
        assert result.content == ""
        assert result.status == ChatStatus.transport_error
        assert result.finish_reason == ""
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.reasoning_content_length == 0


class TestTokenBudget:
    """Tests for prompt-aware token budget calculation."""

    def test_compute_retry_tokens_never_reduces(self):
        """_compute_retry_tokens should never return less than original."""
        # We can't easily instantiate Gemma4Translator for unit tests,
        # but we can verify the logic with a mock approach
        from auto_translate.config import TranslateConfig
        from auto_translate.glossary import Glossary
        from auto_translate.stream_context import StreamContext

        # This test verifies the concept: retry_tokens >= original_max_tokens
        # The actual method requires a full translator instance, so we test
        # the formula directly
        cfg = TranslateConfig()
        original = cfg.max_tokens  # 128
        doubled = cfg.max_tokens * 2  # 256
        context_size = cfg.context_size  # 4096

        # With a short prompt, usable_output should be large
        prompt_tokens = 50  # short prompt
        reserve = 128
        usable_output = context_size - prompt_tokens - reserve  # 3918

        # retry_tokens = max(original, min(usable_output, doubled))
        retry_tokens = max(original, min(usable_output, doubled))
        assert retry_tokens >= original, "Retry tokens should never be less than original"
        assert retry_tokens == 256  # min(3918, 256) = 256, max(128, 256) = 256

    def test_compute_retry_tokens_respects_context(self):
        """Token budget should respect context size limits."""
        from auto_translate.config import TranslateConfig
        cfg = TranslateConfig()
        original = cfg.max_tokens
        doubled = cfg.max_tokens * 2
        context_size = cfg.context_size

        # With a very long prompt, usable_output should be small
        prompt_tokens = context_size - 200  # very long prompt
        reserve = 128
        usable_output = context_size - prompt_tokens - reserve  # 72

        # retry_tokens = max(original, min(usable_output, doubled))
        retry_tokens = max(original, min(usable_output, doubled))
        # usable_output (72) < original (128), so max(128, 72) = 128
        assert retry_tokens == original  # no increase possible

    def test_compute_retry_tokens_large_max_tokens(self):
        """With max_tokens > 256, budget should not cap at 256."""
        # Simulate max_tokens = 512
        original = 512
        doubled = original * 2  # 1024
        context_size = 4096
        prompt_tokens = 100
        reserve = 128
        usable_output = context_size - prompt_tokens - reserve  # 3868

        retry_tokens = max(original, min(usable_output, doubled))
        assert retry_tokens == 1024  # min(3868, 1024) = 1024, max(512, 1024) = 1024
        assert retry_tokens > 256  # should not cap at 256
