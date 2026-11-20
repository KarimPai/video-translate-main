"""Regression tests for slash-separated alternatives preservation.

These tests verify that:
1. The system prompt explicitly forbids alternatives and explanations
2. Slash-separated content is preserved, never deleted
3. Explanatory text is handled correctly
4. `english_output_issue()` does not reject valid slash content
"""
import pytest
from auto_translate.translate.gemma4 import (
    _SYSTEM_PROMPT,
    resolve_slash_alternatives,
    clean_output,
    build_prompt,
)
from auto_translate.subtitle_clean import (
    has_slash_alternatives,
    english_output_issue,
)


class TestSystemPrompt:
    """Verify the system prompt contains the necessary instructions."""

    def test_forbids_slash_alternatives(self):
        """System prompt must explicitly forbid slash-separated alternatives."""
        assert "slash-separated alternatives" in _SYSTEM_PROMPT.lower() or "slash" in _SYSTEM_PROMPT.lower()

    def test_forbids_explanatory_text(self):
        """System prompt must explicitly forbid explanatory notes."""
        assert "explanatory" in _SYSTEM_PROMPT.lower() or "translation notes" in _SYSTEM_PROMPT.lower()

    def test_forbids_definitions(self):
        """System prompt must explicitly forbid definitions."""
        assert "definitions" in _SYSTEM_PROMPT.lower() or "multiple meanings" in _SYSTEM_PROMPT.lower()

    def test_requires_single_translation(self):
        """System prompt must require a single translation."""
        assert "only the english translation" in _SYSTEM_PROMPT.lower() or "one translation" in _SYSTEM_PROMPT.lower()


class TestSlashAlternativesDetection:
    """Test has_slash_alternatives detection (diagnostic only)."""

    def test_simple_alternative(self):
        """Detect simple slash-separated patterns."""
        assert has_slash_alternatives("Hello/Hi") is True

    def test_alternative_with_spaces(self):
        """Detect patterns with spaces around slash."""
        assert has_slash_alternatives("Hello / Hi") is True

    def test_multi_word_alternative(self):
        """Detect multi-word patterns."""
        assert has_slash_alternatives("good morning / hello") is True

    def test_no_alternative(self):
        """Normal text without alternatives should not be flagged."""
        assert has_slash_alternatives("Hello, how are you?") is False

    def test_empty_string(self):
        """Empty string should not be flagged."""
        assert has_slash_alternatives("") is False

    def test_none_input(self):
        """None input should not be flagged."""
        assert has_slash_alternatives(None) is False

    def test_url_preserved(self):
        """URLs with slashes should not be flagged as alternatives."""
        assert has_slash_alternatives("Check out https://example.com") is False

    def test_multiple_alternatives(self):
        """Multiple alternatives in one string should be detected."""
        assert has_slash_alternatives("Hello/Hi, good/great job") is True


class TestResolveSlashAlternatives:
    """Test resolve_slash_alternatives function — now preserves all slashes."""

    def test_simple_alternative_preserved(self):
        """Simple alternative should be preserved, not resolved."""
        result = resolve_slash_alternatives("Hello/Hi")
        assert result == "Hello/Hi"

    def test_alternative_with_spaces_preserved(self):
        """Alternative with spaces should be preserved."""
        result = resolve_slash_alternatives("Hello / Hi")
        assert result == "Hello / Hi"

    def test_multi_word_alternative_preserved(self):
        """Multi-word alternative should be preserved."""
        result = resolve_slash_alternatives("good morning / hello")
        assert result == "good morning / hello"

    def test_alternative_in_sentence_preserved(self):
        """Alternative within a sentence should be preserved."""
        result = resolve_slash_alternatives("Say Hello/Hi to them")
        assert result == "Say Hello/Hi to them"

    def test_multiple_alternatives_preserved(self):
        """Multiple alternatives should all be preserved."""
        result = resolve_slash_alternatives("Hello/Hi, good/great job")
        assert result == "Hello/Hi, good/great job"

    def test_no_alternative_unchanged(self):
        """Text without alternatives should be unchanged."""
        result = resolve_slash_alternatives("Hello, how are you?")
        assert result == "Hello, how are you?"

    def test_empty_string(self):
        """Empty string should return empty string."""
        result = resolve_slash_alternatives("")
        assert result == ""

    def test_nested_alternatives_preserved(self):
        """Nested or chained alternatives should be preserved."""
        result = resolve_slash_alternatives("a/b/c")
        assert result == "a/b/c"

    def test_yes_no_preserved(self):
        """'yes/no' must be preserved — never corrupted to 'yes'."""
        result = resolve_slash_alternatives("yes/no")
        assert result == "yes/no"

    def test_on_off_preserved(self):
        """'on/off' must be preserved — never corrupted to 'on'."""
        result = resolve_slash_alternatives("on/off")
        assert result == "on/off"

    def test_and_or_preserved(self):
        """'and/or' must be preserved."""
        result = resolve_slash_alternatives("and/or")
        assert result == "and/or"

    def test_input_output_preserved(self):
        """'input/output' must be preserved."""
        result = resolve_slash_alternatives("input/output")
        assert result == "input/output"

    def test_24_7_preserved(self):
        """'24/7' must be preserved."""
        result = resolve_slash_alternatives("24/7")
        assert result == "24/7"

    def test_url_preserved(self):
        """URLs must be preserved."""
        result = resolve_slash_alternatives("Check out https://example.com")
        assert result == "Check out https://example.com"

    def test_date_preserved(self):
        """Dates must be preserved."""
        result = resolve_slash_alternatives("The date is 12/25/2024")
        assert result == "The date is 12/25/2024"

    def test_fraction_preserved(self):
        """Fractions must be preserved."""
        result = resolve_slash_alternatives("I ate 1/2 of the pizza")
        assert result == "I ate 1/2 of the pizza"


class TestCleanOutputIntegration:
    """Test that clean_output properly preserves slash content."""

    def test_alternative_in_clean_output_preserved(self):
        """clean_output should preserve slash alternatives."""
        result = clean_output("Hello/Hi")
        assert result == "Hello/Hi"

    def test_alternative_with_markdown(self):
        """clean_output should handle markdown but preserve slashes."""
        result = clean_output("**Hello/Hi**")
        assert result == "Hello/Hi"

    def test_normal_translation(self):
        """Normal translation should pass through unchanged."""
        result = clean_output("Hello, how are you?")
        assert result == "Hello, how are you?"

    def test_yes_no_in_clean_output(self):
        """'yes/no' must survive clean_output unchanged."""
        result = clean_output("Is it yes/no?")
        assert result == "Is it yes/no?"

    def test_24_7_in_clean_output(self):
        """'24/7' must survive clean_output unchanged."""
        result = clean_output("We work 24/7")
        assert result == "We work 24/7"


class TestEnglishOutputIssue:
    """Test that english_output_issue does NOT reject slash content."""

    def test_slash_content_no_issue(self):
        """Slash-separated content should NOT be detected as an issue."""
        issue = english_output_issue("Hello/Hi")
        assert issue is None

    def test_yes_no_no_issue(self):
        """'yes/no' should not be flagged as an issue."""
        issue = english_output_issue("Is it yes/no?")
        assert issue is None

    def test_24_7_no_issue(self):
        """'24/7' should not be flagged as an issue."""
        issue = english_output_issue("We work 24/7")
        assert issue is None

    def test_input_output_no_issue(self):
        """'input/output' should not be flagged as an issue."""
        issue = english_output_issue("Check the input/output")
        assert issue is None

    def test_normal_text_no_issue(self):
        """Normal text should not have an issue."""
        issue = english_output_issue("Hello, how are you?")
        assert issue is None

    def test_empty_text_issue(self):
        """Empty text should still be detected as empty."""
        issue = english_output_issue("")
        assert issue == "empty_english"


class TestBuildPrompt:
    """Test that build_prompt includes the updated system prompt."""

    def test_prompt_has_system_message(self):
        """build_prompt should include the system message."""
        prompt = build_prompt("こんにちは", [], "")
        assert "messages" in prompt
        assert len(prompt["messages"]) >= 2
        assert prompt["messages"][0]["role"] == "system"

    def test_system_prompt_content(self):
        """System prompt should contain the new instructions."""
        prompt = build_prompt("こんにちは", [], "")
        system_content = prompt["messages"][0]["content"]
        assert "slash-separated alternatives" in system_content.lower() or "slash" in system_content.lower()


class TestRepresentativeProblemCases:
    """Test cases representing the problematic patterns — all now preserved."""

    def test_greeting_preserved(self):
        """'Hello/Hi' is preserved, not resolved to 'Hello'."""
        assert has_slash_alternatives("Hello/Hi") is True
        assert resolve_slash_alternatives("Hello/Hi") == "Hello/Hi"

    def test_adjective_preserved(self):
        """'good/great' is preserved, not resolved to 'good'."""
        assert has_slash_alternatives("That was good/great") is True
        assert resolve_slash_alternatives("That was good/great") == "That was good/great"

    def test_verb_preserved(self):
        """'run/running' is preserved, not resolved to 'run'."""
        assert has_slash_alternatives("Start run/running now") is True
        assert resolve_slash_alternatives("Start run/running now") == "Start run/running now"

    def test_phrase_preserved(self):
        """Phrase alternatives are preserved."""
        assert has_slash_alternatives("Good morning / Hello") is True
        assert resolve_slash_alternatives("Good morning / Hello") == "Good morning / Hello"

    def test_explanatory_text_not_alternative(self):
        """Explanatory text in parentheses is not a slash alternative."""
        text = "Hello (this is a greeting)"
        assert has_slash_alternatives(text) is False

    def test_complex_alternative_preserved(self):
        """Complex sentence with alternatives is preserved."""
        text = "I'm going to the store/market to buy some food"
        assert has_slash_alternatives(text) is True
        result = resolve_slash_alternatives(text)
        assert result == text  # fully preserved

    def test_slash_not_rejected_by_output_issue(self):
        """Slash content must not be rejected by english_output_issue."""
        issue = english_output_issue("Hello/Hi")
        assert issue is None, "Slash content should not be rejected"

    def test_yes_no_not_rejected(self):
        """'yes/no' must not be rejected by english_output_issue."""
        issue = english_output_issue("Is it yes/no?")
        assert issue is None, "'yes/no' should not be rejected"
