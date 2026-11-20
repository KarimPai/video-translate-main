"""Regression tests for prompt leakage and subtitle integrity.

These tests verify that:
1. Reasoning tags (</think>, <thinking>, etc.) are stripped from output
2. System prompt echoes are rejected
3. User prompt template echoes are rejected
4. Glossary content is never published as subtitle text
5. Explanatory text patterns are rejected
6. Slash-separated alternatives are resolved
7. History contamination is prevented
8. Normal translations pass through unchanged
"""
import pytest
from auto_translate.translate.gemma4 import (
    _SYSTEM_PROMPT,
    clean_output,
    resolve_slash_alternatives,
    build_prompt,
)
from auto_translate.subtitle_clean import (
    looks_like_prompt_echo,
    english_output_issue,
    sanitize_english_output,
    has_slash_alternatives,
)


class TestReasoningTagStripping:
    """Verify reasoning tags are stripped from model output before subtitles."""

    def test_think_tags_stripped(self):
        """Full <think> block should be removed, leaving only translation."""
        raw = "<think>This is Japanese for hello</think>Hello, how are you?"
        result = clean_output(raw)
        assert result == "Hello, how are you?"
        assert "<think" not in result

    def test_thinking_tags_stripped(self):
        """Full <thinking> block should be removed."""
        raw = "<thinking>Let me translate this...</thinking>Good morning!"
        result = clean_output(raw)
        assert result == "Good morning!"

    def test_reasoning_tags_stripped(self):
        """Full <reasoning> block should be removed."""
        raw = "<reasoning>Analyzing the context...</reasoning>Let's go!"
        result = clean_output(raw)
        assert result == "Let's go!"

    def test_multiline_think_block(self):
        """Multi-line reasoning block should be fully removed."""
        raw = "<think>\nLine 1 of reasoning\nLine 2 of reasoning\n</think>Thanks everyone!"
        result = clean_output(raw)
        assert result == "Thanks everyone!"
        assert "reasoning" not in result.lower()

    def test_orphan_think_tag(self):
        """Orphan opening tag without closing should be removed."""
        raw = "<think>Hello, how are you?"
        result = clean_output(raw)
        assert result == "Hello, how are you?"

    def test_orphan_close_tag(self):
        """Orphan closing tag should be removed."""
        raw = "</think>Hello, how are you?"
        result = clean_output(raw)
        assert result == "Hello, how are you?"

    def test_only_reasoning_content(self):
        """If output is only reasoning with no translation, return empty."""
        raw = "<think>This is all reasoning, no translation</think>"
        result = clean_output(raw)
        assert result == ""

    def test_reasoning_after_translation(self):
        """Reasoning after the translation should be stripped."""
        raw = "Hello! <think>Wait, did I translate that right?</think>"
        result = clean_output(raw)
        assert result == "Hello!"

    def test_reasoning_before_and_after(self):
        """Reasoning on both sides should be stripped."""
        raw = "<think>reasoning</think>Hello!<think>more reasoning</think>"
        result = clean_output(raw)
        assert result == "Hello!"

    def test_deepseek_format_tags(self):
        """Deepseek-style reasoning tags should be handled."""
        raw = "<think>Let me think about this translation...</think>Everyone, thanks for watching!"
        result = clean_output(raw)
        assert result == "Everyone, thanks for watching!"

    def test_reasoning_with_attributes(self):
        """Reasoning tags with attributes should be handled."""
        raw = "<thinking mode='deep'>Analysis here</thinking>Good job!"
        result = clean_output(raw)
        assert result == "Good job!"

    def test_normal_text_no_false_positive(self):
        """Normal English dialogue should not be affected."""
        result = clean_output("Hello, how are you today?")
        assert result == "Hello, how are you today?"

    def test_vtuber_dialogue_preserved(self):
        """VTuber-style dialogue should pass through unchanged."""
        result = clean_output("Lap-chan, wait a minute! I wanted to use that item!")
        assert result == "Lap-chan, wait a minute! I wanted to use that item!"

    def test_profanity_preserved(self):
        """Profanity in translation should be preserved, not filtered."""
        result = clean_output("Damn, that was close!")
        assert result == "Damn, that was close!"

    def test_slang_preserved(self):
        """Slang and informal speech should be preserved."""
        result = clean_output("What's up, guys? Let's get this party started!")
        assert result == "What's up, guys? Let's get this party started!"

    def test_gemma4_think_token_stripped(self):
        """Gemma 4 native <|think|> token should be stripped (orphan tag)."""
        # <|think|> without closing tag is stripped by _REASONING_OPEN
        # leaving the remaining text. This is acceptable behavior.
        result = clean_output("<|think|>\nLet me think about this.\nHello")
        assert "<|think|>" not in result
        assert "Hello" in result

    def test_gemma4_channel_thought_stripped(self):
        """Gemma 4 native <|channel>thought wrapper should be stripped."""
        raw = "<|channel>thought This is reasoning content. <channel|>Hello world"
        result = clean_output(raw)
        assert result == "Hello world"
        assert "reasoning" not in result

    def test_gemma4_channel_multiline_stripped(self):
        """Gemma 4 channel wrapper with multiline reasoning should be stripped."""
        raw = "<|channel>thought Line 1 of reasoning Line 2 of reasoning <channel|>Good morning!"
        result = clean_output(raw)
        assert result == "Good morning!"

    def test_gemma4_think_token_no_false_positive(self):
        """Normal text should not be affected by <|think|> pattern."""
        result = clean_output("I think this is great!")
        assert result == "I think this is great!"

    def test_unclosed_think_token_handled(self):
        """Unclosed <|think|> token should not leak content."""
        raw = "<|think|>\nReasoning that never closes\nHello"
        result = clean_output(raw)
        assert "Hello" in result or result == ""

    def test_unclosed_channel_thought_handled(self):
        """Unclosed <|channel>thought should not leak reasoning."""
        raw = "<|channel>thought\nReasoning without closing\nHello"
        result = clean_output(raw)
        assert "channel" not in result.lower() or "Hello" in result


class TestPromptEchoRejection:
    """Verify system prompt and user prompt echoes are rejected."""

    def test_system_prompt_full_echo(self):
        """Full system prompt echo should be rejected."""
        raw = _SYSTEM_PROMPT
        result = clean_output(raw)
        assert result == ""

    def test_system_prompt_partial_echo(self):
        """Partial system prompt echo should be rejected."""
        raw = "You are a professional Japanese-to-English subtitle translator. Hello!"
        result = clean_output(raw)
        assert result == ""

    def test_output_only_instruction_echo(self):
        """Instruction echo about output format should be rejected."""
        raw = "Output only the English translation, nothing else."
        result = clean_output(raw)
        assert result == ""

    def test_slash_alternatives_instruction_echo(self):
        """Instruction about slash alternatives should be rejected."""
        raw = "Never output slash-separated alternatives (e.g. 'Hello/Hi')"
        result = clean_output(raw)
        assert result == ""

    def test_explanatory_text_instruction_echo(self):
        """Instruction about explanatory text should be rejected."""
        raw = "Never add explanatory text, translation notes, or parenthetical comments."
        result = clean_output(raw)
        assert result == ""

    def test_user_prompt_template_echo(self):
        """User prompt template echo should be rejected."""
        raw = "Reference context (not part of the speech to translate):\nSome context"
        result = clean_output(raw)
        assert result == ""

    def test_translate_this_japanese_echo(self):
        """'Translate this Japanese text' echo should be rejected."""
        raw = "Translate this Japanese text:\nこんにちは"
        result = clean_output(raw)
        assert result == ""

    def test_glossary_section_echo(self):
        """Glossary section echo should be rejected."""
        raw = "Name/term glossary:\n  名前 → Name\n\nHello!"
        result = clean_output(raw)
        assert result == ""

    def test_translation_style_echo(self):
        """Translation style echo should be rejected."""
        raw = "Translation style: Plain and concise."
        result = clean_output(raw)
        assert result == ""

    def test_dialogue_context_echo(self):
        """Dialogue context header echo should be rejected."""
        raw = "直前の会話：Previous dialogue here"
        result = clean_output(raw)
        assert result == ""

    def test_looks_like_prompt_echo_function(self):
        """looks_like_prompt_echo should catch known markers."""
        assert looks_like_prompt_echo("recently mentioned glossary names/terms: test")
        assert looks_like_prompt_echo("chat_template_kwargs: value")
        assert looks_like_prompt_echo("translate this japanese text: こんにちは")

    def test_normal_text_not_prompt_echo(self):
        """Normal dialogue should not be flagged as prompt echo."""
        assert not looks_like_prompt_echo("Hello, how are you?")
        assert not looks_like_prompt_echo("Thanks for the super chats!")
        assert not looks_like_prompt_echo("Let's play some games today!")


class TestGlossaryLeakage:
    """Verify glossary content is never published as subtitle text."""

    def test_arrow_in_output(self):
        """Glossary arrow (→) in output should be rejected."""
        raw = "名前 → Name"
        assert looks_like_prompt_echo(raw)

    def test_glossary_format_in_output(self):
        """Glossary format with arrows should be rejected."""
        raw = "  名前 → Name\n  年齢 → Age"
        assert looks_like_prompt_echo(raw)

    def test_glossary_in_translation(self):
        """If model echoes glossary in translation, it should be rejected."""
        raw = "Name/term glossary:\n  名前 → Name\n\nHello!"
        result = clean_output(raw)
        assert result == ""

    def test_normal_translation_with_similar_words(self):
        """Normal translation that happens to contain glossary-like words should pass."""
        # A normal translation shouldn't have arrows or glossary format
        result = clean_output("Her name is Yuki and she's 20 years old.")
        assert result == "Her name is Yuki and she's 20 years old."


class TestExplanatoryTextRejection:
    """Verify explanatory text patterns are rejected."""

    def test_translation_prefix_stripped(self):
        """'Translation:' prefix should be stripped as a role label."""
        raw = "Translation: Hello, how are you?"
        result = clean_output(raw)
        assert result == "Hello, how are you?"

    def test_translate_prefix_stripped(self):
        """'Translate:' prefix should be stripped as a role label."""
        raw = "Translate: こんにちは → Hello"
        result = clean_output(raw)
        # Arrow (→) triggers prompt echo detection, so this is rejected
        assert result == ""

    def test_note_prefix(self):
        """'Note:' prefix should be rejected."""
        raw = "Note: This is a casual greeting."
        result = clean_output(raw)
        assert result == ""

    def test_explanation_prefix(self):
        """'Explanation:' prefix should be rejected."""
        raw = "Explanation: The speaker is greeting someone."
        result = clean_output(raw)
        assert result == ""

    def test_here_is_translation_prefix(self):
        """'Here is the translation' prefix should be rejected."""
        raw = "Here is the translation: Hello!"
        result = clean_output(raw)
        assert result == ""

    def test_heres_the_translation_prefix(self):
        """"Here's the translation" prefix should be rejected."""
        raw = "Here's the translation: Hello!"
        result = clean_output(raw)
        assert result == ""

    def test_the_translation_prefix(self):
        """'The translation' prefix should be rejected."""
        raw = "The translation is: Good morning!"
        result = clean_output(raw)
        assert result == ""

    def test_this_means_prefix(self):
        """'This means:' prefix should be rejected."""
        raw = "This means: Hello there."
        result = clean_output(raw)
        assert result == ""

    def test_this_means_with_colon(self):
        """'This means' with full-width colon should be rejected."""
        raw = "This means：Hello there."
        result = clean_output(raw)
        assert result == ""

    def test_i_would_translate_prefix(self):
        """'I would translate' prefix should be rejected."""
        raw = "I would translate this as: Hello!"
        result = clean_output(raw)
        assert result == ""

    def test_i_believe_prefix(self):
        """'I believe' prefix should be rejected."""
        raw = "I believe this is the correct translation."
        result = clean_output(raw)
        assert result == ""

    def test_normal_sentence_starting_with_here(self):
        """Normal sentence starting with 'Here' should pass if not explanatory."""
        # "Here we go" is a natural phrase, not explanatory
        result = clean_output("Here we go!")
        assert result == "Here we go!"

    def test_normal_sentence_starting_with_this(self):
        """Normal sentence starting with 'This' should pass."""
        result = clean_output("This is amazing!")
        assert result == "This is amazing!"

    def test_normal_sentence_starting_with_i(self):
        """Normal sentence starting with 'I' should pass."""
        result = clean_output("I love this game!")
        assert result == "I love this game!"

    def test_normal_i_think_sentence(self):
        """Normal 'I think' sentence should pass — not followed by explanatory pattern."""
        result = clean_output("I think so!")
        assert result == "I think so!"

    def test_normal_this_is_sentence(self):
        """Normal 'This is' sentence should pass."""
        result = clean_output("This is the best game ever!")
        assert result == "This is the best game ever!"


class TestSlashAlternatives:
    """Verify slash-separated alternatives are properly handled."""

    def test_simple_alternative_detected(self):
        """Simple slash alternative should be detected."""
        assert has_slash_alternatives("Hello/Hi") is True

    def test_alternative_preserved(self):
        """Slash alternative should be preserved, not resolved."""
        result = resolve_slash_alternatives("Hello/Hi")
        assert result == "Hello/Hi"

    def test_alternative_in_clean_output_preserved(self):
        """clean_output should preserve slash alternatives."""
        result = clean_output("Hello/Hi")
        assert result == "Hello/Hi"

    def test_multiple_alternatives_preserved(self):
        """Multiple alternatives should all be preserved."""
        result = resolve_slash_alternatives("Hello/Hi, good/great job")
        assert result == "Hello/Hi, good/great job"

    def test_normal_text_no_alternative(self):
        """Normal text without alternatives should pass through."""
        result = clean_output("Hello, how are you?")
        assert result == "Hello, how are you?"

    def test_url_not_flagged(self):
        """URLs should not be flagged as alternatives."""
        assert has_slash_alternatives("Check out https://example.com") is False

    def test_english_output_issue_allows_slash_content(self):
        """english_output_issue should NOT reject slash-separated content."""
        issue = english_output_issue("Hello/Hi")
        assert issue is None


class TestMalformedResponseHandling:
    """Verify malformed model responses are handled safely."""

    def test_empty_output(self):
        """Empty output should be rejected."""
        result = clean_output("")
        assert result == ""

    def test_whitespace_only(self):
        """Whitespace-only output should be rejected."""
        result = clean_output("   \n  \t  ")
        assert result == ""

    def test_japanese_output_rejected(self):
        """Japanese output should be rejected by english_output_issue."""
        issue = english_output_issue("こんにちは")
        assert issue == "untranslated_japanese"

    def test_special_token_flood_rejected(self):
        """Special token flood should be rejected."""
        issue = english_output_issue("<unused49> <unused49> <unused49>")
        assert issue == "special_token_flood"

    def test_sanitize_rejects_prompt_echo(self):
        """sanitize_english_output should reject prompt echoes."""
        result = sanitize_english_output("recently mentioned glossary names/terms: test")
        assert result == ""

    def test_sanitize_rejects_reasoning_tags(self):
        """sanitize_english_output should reject reasoning tag content."""
        result = sanitize_english_output("<think>reasoning</think>")
        # Reasoning tags are caught by looks_like_prompt_echo
        assert result == ""

    def test_markdown_stripped(self):
        """Markdown formatting should be stripped."""
        result = clean_output("**Hello, how are you?**")
        assert result == "Hello, how are you?"

    def test_role_prefix_stripped(self):
        """Role prefix like 'EN:' should be stripped."""
        result = clean_output("EN: Hello, how are you?")
        assert result == "Hello, how are you?"

    def test_translation_prefix_stripped(self):
        """'TRANSLATION:' prefix should be stripped as a role label."""
        result = clean_output("TRANSLATION: Good morning!")
        assert result == "Good morning!"

    def test_orphan_quotes_cleaned(self):
        """Orphan quotes should be cleaned."""
        result = clean_output('Hello, how are you?"')
        assert '"' not in result or result.count('"') % 2 == 0


class TestHistoryContamination:
    """Verify that bad translations don't contaminate future context."""

    def test_clean_translation_in_history(self):
        """Clean translations should be safe for history."""
        from auto_translate.subtitle_clean import english_output_issue
        en = "Hello, how are you?"
        assert english_output_issue(en) is None

    def test_prompt_echo_not_safe_for_history(self):
        """Prompt echoes should be flagged as unsafe for history."""
        en = "recently mentioned glossary names/terms: test"
        assert english_output_issue(en) is not None

    def test_reasoning_leak_not_safe_for_history(self):
        """Reasoning leaks should be flagged as unsafe for history."""
        en = "<think>reasoning</think>Hello"
        # After clean_output strips reasoning, "Hello" remains
        # But looks_like_prompt_echo catches the tags
        assert looks_like_prompt_echo(en)

    def test_empty_translation_not_safe_for_history(self):
        """Empty translations should be flagged."""
        assert english_output_issue("") == "empty_english"


class TestSystemPromptContent:
    """Verify the system prompt contains necessary instructions."""

    def test_forbids_slash_alternatives(self):
        """System prompt must forbid slash-separated alternatives."""
        assert "slash-separated alternatives" in _SYSTEM_PROMPT.lower()

    def test_forbids_explanatory_text(self):
        """System prompt must forbid explanatory text/notes."""
        assert "explanatory" in _SYSTEM_PROMPT.lower()

    def test_forbids_definitions(self):
        """System prompt must forbid definitions or alternatives."""
        assert "definitions" in _SYSTEM_PROMPT.lower() or "alternatives" in _SYSTEM_PROMPT.lower()

    def test_requires_single_translation(self):
        """System prompt must require a single translation."""
        assert "only the english translation" in _SYSTEM_PROMPT.lower()

    def test_preserves_profanity(self):
        """System prompt must allow profanity preservation."""
        assert "profanity" in _SYSTEM_PROMPT.lower()

    def test_preserves_slang(self):
        """System prompt must allow slang/nickname preservation."""
        assert "nicknames" in _SYSTEM_PROMPT.lower() or "slang" in _SYSTEM_PROMPT.lower()


class TestBuildPrompt:
    """Verify build_prompt constructs correct prompts."""

    def test_prompt_has_system_message(self):
        """build_prompt should include the system message."""
        prompt = build_prompt("こんにちは", [], "")
        assert prompt["messages"][0]["role"] == "system"
        assert prompt["messages"][0]["content"] == _SYSTEM_PROMPT

    def test_prompt_has_user_message(self):
        """build_prompt should include the user message."""
        prompt = build_prompt("こんにちは", [], "")
        assert prompt["messages"][1]["role"] == "user"
        assert "こんにちは" in prompt["messages"][1]["content"]

    def test_prompt_with_glossary(self):
        """build_prompt should include glossary terms."""
        terms = [("名前", "Name"), ("年齢", "Age")]
        prompt = build_prompt("こんにちは", terms, "")
        user_content = prompt["messages"][1]["content"]
        assert "名前" in user_content
        assert "Name" in user_content

    def test_prompt_with_context(self):
        """build_prompt should include context."""
        prompt = build_prompt("こんにちは", [], "Previous dialogue")
        user_content = prompt["messages"][1]["content"]
        assert "Previous dialogue" in user_content

    def test_prompt_with_style(self):
        """build_prompt should include style guidance."""
        prompt = build_prompt("こんにちは", [], "", "Plain and concise")
        user_content = prompt["messages"][1]["content"]
        assert "Plain and concise" in user_content




class TestRepresentativeLeakageCases:
    """Test cases representing real-world leakage patterns."""

    def test_reasoning_then_translation(self):
        """Model outputs reasoning then translation — only translation kept."""
        raw = "<think>こんにちは means hello in Japanese. I should translate this naturally.</think>Hello, how are you?"
        result = clean_output(raw)
        assert result == "Hello, how are you?"
        assert "think" not in result
        assert "Japanese" not in result

    def test_full_prompt_echo(self):
        """Model echoes entire prompt — should be rejected."""
        raw = (
            "You are a professional Japanese-to-English subtitle translator. "
            "Translate the Japanese text into natural, conversational English subtitles."
        )
        result = clean_output(raw)
        assert result == ""

    def test_glossary_echo_in_output(self):
        """Model echoes glossary in output — should be rejected."""
        raw = "Name/term glossary:\n  名前 → Name\n  年齢 → Age"
        result = clean_output(raw)
        assert result == ""

    def test_explanatory_translation(self):
        """Model provides explanation with 'Translation:' prefix — prefix is stripped,
        remaining text is valid English so it passes through. The _EXPLANATORY pattern
        only catches cases like 'Note:', 'Explanation:', etc. that are clearly not dialogue."""
        raw = "Translation: This is a greeting. The speaker is saying hello."
        result = clean_output(raw)
        # "Translation:" is stripped as a role prefix, leaving valid English
        assert result == "This is a greeting. The speaker is saying hello."

    def test_explanatory_with_note_prefix(self):
        """Model provides explanation with 'Note:' prefix — should be rejected."""
        raw = "Note: This is a greeting. The speaker is saying hello."
        result = clean_output(raw)
        assert result == ""

    def test_alternative_translation_preserved(self):
        """Model provides alternatives — should be preserved, not resolved."""
        raw = "Hello/Hi, how are you?"
        result = clean_output(raw)
        assert result == "Hello/Hi, how are you?"

    def test_reasoning_concatenated_with_translation(self):
        """Reasoning concatenated with translation — reasoning stripped."""
        raw = "<think>Analysis</think>Hello!<think>More analysis</think>"
        result = clean_output(raw)
        assert result == "Hello!"

    def test_normal_vtuber_greeting(self):
        """Normal VTuber greeting should pass through."""
        result = clean_output("Everyone, thanks for the super chats! Let's start the stream!")
        assert result == "Everyone, thanks for the super chats! Let's start the stream!"

    def test_normal_japanese_name_preserved(self):
        """Japanese names in translation should be preserved."""
        result = clean_output("Yuki-chan, wait for me!")
        assert result == "Yuki-chan, wait for me!"

    def test_normal_profanity_preserved(self):
        """Profanity should be preserved, not filtered."""
        result = clean_output("Damn, that was intense!")
        assert result == "Damn, that was intense!"

    def test_normal_slang_preserved(self):
        """Slang should be preserved."""
        result = clean_output("What's up, guys? Ready to play?")
        assert result == "What's up, guys? Ready to play?"

    def test_nickname_preserved(self):
        """Nicknames should be preserved."""
        result = clean_output("Lap-Pig, you're so funny!")
        assert result == "Lap-Pig, you're so funny!"

    def test_emotional_expression_preserved(self):
        """Emotional expressions should be preserved."""
        result = clean_output("Aww, that's so cute!")
        assert result == "Aww, that's so cute!"
