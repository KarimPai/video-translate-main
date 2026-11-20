"""Tests for Gemma 4 compatibility hardening: special token detection and output validation."""
from __future__ import annotations

import pytest
from auto_translate.subtitle_clean import (
    has_special_token_flood,
    english_output_issue,
    sanitize_english_output,
    clean_english_output,
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
