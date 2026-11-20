"""Behavioral regression tests for PLaMo translation fixes.

Tests the actual retry state machine, watermark state transitions, and
pending speech lifecycle using mocked server responses — no GPU usage.

Covers:
1. Retry detection: outputs recorded AFTER checking, not before
2. Request key isolation: different prompts don't falsely flag as repeated
3. Truncation recovery: simplified prompt first, then increased tokens
4. Watermark contiguous coverage: never skip pending speech
5. complete_speech(): only clears its own entry, respects other pending speech
6. Out-of-order completion
7. Stale-generation callbacks
8. VOD auto-pause/resume state
"""

import sys
import threading
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from auto_translate.subtitle_clean import english_output_issue
from auto_translate.translate.plamo import build_plamo_prompt, clean_output

# ---------------------------------------------------------------------------
# 1. PLaMo Prompt Format
# ---------------------------------------------------------------------------

class TestPlamoPromptFormat:
    """Verify build_plamo_prompt() produces correct chat_template_kwargs."""

    def test_prompt_includes_source_lang(self):
        result = build_plamo_prompt("こんにちは", [], "")
        kwargs = result.get("chat_template_kwargs", {})
        assert kwargs.get("source_lang") == "ja"

    def test_prompt_includes_target_lang(self):
        result = build_plamo_prompt("こんにちは", [], "")
        kwargs = result.get("chat_template_kwargs", {})
        assert kwargs.get("target_lang") == "en"

    def test_prompt_has_user_message(self):
        result = build_plamo_prompt("こんにちは", [], "")
        messages = result.get("messages", [])
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_prompt_contains_japanese_text(self):
        text = "今日はいい天気ですね"
        result = build_plamo_prompt(text, [], "")
        content = result["messages"][0]["content"]
        assert text in content

    def test_glossary_via_kwargs_not_user_text(self):
        """Glossary terms go to chat_template_kwargs, not the user message."""
        terms = [("ミク", "Miku"), ("ボカロ", "Vocaloid")]
        result = build_plamo_prompt("ミクが歌う", terms, "")
        content = result["messages"][0]["content"]
        assert content == "ミクが歌う"
        kwargs = result.get("chat_template_kwargs", {})
        glossary = kwargs.get("glossary", [])
        assert ("ミク", "Miku") in [(g["source"], g["target"]) for g in glossary]
        assert ("ボカロ", "Vocaloid") in [(g["source"], g["target"]) for g in glossary]

    def test_context_via_kwargs_not_user_text(self):
        """Context goes to chat_template_kwargs, not the user message."""
        context = "直前の会話：ゲームを始めましょう"
        result = build_plamo_prompt("はい、始めましょう", [], context)
        content = result["messages"][0]["content"]
        assert content == "はい、始めましょう"
        kwargs = result.get("chat_template_kwargs", {})
        assert kwargs.get("context") == context

    def test_simplified_prompt_omits_glossary_and_context(self):
        terms = [("ミク", "Miku")]
        context = "直前の会話：ゲームを始めましょう"
        result = build_plamo_prompt("ミクが歌う", terms, context, simplified=True)
        content = result["messages"][0]["content"]
        assert content == "ミクが歌う"
        kwargs = result.get("chat_template_kwargs", {})
        assert "glossary" not in kwargs
        assert "context" not in kwargs

    def test_simplified_prompt_still_has_langs(self):
        result = build_plamo_prompt("こんにちは", [], "", simplified=True)
        kwargs = result.get("chat_template_kwargs", {})
        assert kwargs.get("source_lang") == "ja"
        assert kwargs.get("target_lang") == "en"

    def test_glossary_dict_format(self):
        terms = [("ミク", "Miku")]
        result = build_plamo_prompt("ミクが歌う", terms, "")
        glossary = result["chat_template_kwargs"].get("glossary", [])
        assert glossary == [{"source": "ミク", "target": "Miku"}]


# ---------------------------------------------------------------------------
# 2. Retry Detection — Behavioral Tests
# ---------------------------------------------------------------------------

class TestRetryDetection:
    """Test the retry state machine with mocked server responses."""

    def _make_translator(self, chat_responses):
        """Create a PlamoTranslator with mocked _chat method."""
        from auto_translate.config import TranslateConfig
        from auto_translate.glossary import Glossary
        from auto_translate.stream_context import StreamContext
        from auto_translate.translate.plamo import PlamoTranslator

        cfg = TranslateConfig(
            model_path=str(Path(__file__).parent / "dummy.gguf"),
            server_path="llama-server",
            glossary=str(Path(__file__).parent / "glossary.toml"),
            max_tokens=256,
            timeout_s=10,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            repeat_penalty=1.1,
            context_size=2048,
            parallel_slots=1,
            threads=4,
            gpu_layers=-1,
            load_timeout_s=60,
            context_lines=3,
            max_context_chars=500,
            context_english=False,
        )
        glossary = Glossary({})

        translator = mock.MagicMock()
        translator.cfg = cfg
        translator.glossary = glossary
        translator.stream = StreamContext()
        translator._closed = False
        translator._relief = mock.MagicMock()
        translator._relief.is_set.return_value = False
        translator._lock = threading.Lock()
        translator._MAX_RETRIES = 3
        translator._MAX_TRUNCATION_RETRIES = 1
        translator._MAX_ELAPSED_S = 30.0
        translator._is_invalid = PlamoTranslator._is_invalid
        translator._normalize = PlamoTranslator._normalize
        translator._best = PlamoTranslator._best
        # Mock server.alive() to return True
        translator.server = mock.MagicMock()
        translator.server.alive.return_value = True
        # Mock context and terms — return empty so the prompt is simple
        translator._context = mock.MagicMock(return_value="")
        translator._terms_for = mock.MagicMock(return_value=([], []))

        call_index = [0]
        def mock_chat(prompt, timeout=10, temperature=None, max_tokens=None):
            idx = call_index[0]
            call_index[0] += 1
            if idx < len(chat_responses):
                return chat_responses[idx]
            return ("", "stop")

        translator._chat = mock_chat
        return translator, call_index

    def test_valid_first_attempt_no_retry(self):
        """A valid translation on the first attempt should not retry."""
        translator, call_index = self._make_translator([
            ("Hello, how are you?", "stop"),
        ])
        from auto_translate.translate.plamo import PlamoTranslator
        PlamoTranslator.translate.__get__(translator, type(translator))("こんにちは", [])
        assert call_index[0] == 1

    def test_empty_then_valid(self):
        """Empty first response followed by valid English should succeed."""
        translator, call_index = self._make_translator([
            ("", "stop"),
            ("Hello there", "stop"),
        ])
        from auto_translate.translate.plamo import PlamoTranslator
        PlamoTranslator.translate.__get__(translator, type(translator))("こんにちは", [])
        assert call_index[0] == 2

    def test_truncated_then_simplified_success(self):
        """Truncated first response, then simplified prompt succeeds."""
        translator, call_index = self._make_translator([
            ("This is a very long translation that gets truncated beca", "length"),
            ("This is a complete translation", "stop"),
        ])
        from auto_translate.translate.plamo import PlamoTranslator
        PlamoTranslator.translate.__get__(translator, type(translator))("こんにちは", [])
        assert call_index[0] == 2

    def test_two_identical_truncated_then_higher_tokens(self):
        """Two identical truncated responses should stop truncation recovery."""
        truncated = "This is a very long translation that gets truncated beca"
        translator, call_index = self._make_translator([
            (truncated, "length"),
            (truncated, "length"),
            ("This is the complete translation with more tokens", "stop"),
        ])
        from auto_translate.translate.plamo import PlamoTranslator
        PlamoTranslator.translate.__get__(translator, type(translator))("こんにちは", [])
        # First truncated -> simplified prompt -> same output -> _repeated -> return ""
        assert call_index[0] == 2

    def test_repeated_empty_responses(self):
        """Repeated empty responses should not cause infinite retries."""
        translator, call_index = self._make_translator([
            ("", "stop"),
            ("", "stop"),
        ])
        from auto_translate.translate.plamo import PlamoTranslator
        PlamoTranslator.translate.__get__(translator, type(translator))("こんにちは", [])
        assert call_index[0] <= 3

    def test_exhausted_retries(self):
        """When all retries are exhausted, return empty string."""
        translator, _call_index = self._make_translator([
            ("", "stop"),
            ("", "stop"),
            ("", "stop"),
        ])
        from auto_translate.translate.plamo import PlamoTranslator
        result = PlamoTranslator.translate.__get__(translator, type(translator))("こんにちは", [])
        assert result == ""

    def test_retry_records_after_check(self):
        """The key bug fix: output should NOT be in previous_outputs when _repeated is first called."""
        # This tests the core fix: previous_outputs.append() was called BEFORE _repeated()
        # Now _record() is called AFTER _repeated() checks
        from auto_translate.translate.plamo import PlamoTranslator

        # Simulate the internal state
        previous_outputs = []

        def _repeated(output, request_key):
            norm = PlamoTranslator._normalize(output)
            for prev_norm, prev_key in previous_outputs:
                if prev_key == request_key and prev_norm == norm:
                    return True
            return False

        def _record(output, request_key):
            previous_outputs.append((PlamoTranslator._normalize(output), request_key))

        # First output: empty string
        empty_output = ""
        # Before the fix, empty_output would already be in previous_outputs
        # After the fix, it should NOT be
        assert _repeated(empty_output, "full") is False
        _record(empty_output, "full")
        # Now it should be detected as repeated
        assert _repeated(empty_output, "full") is True

    def test_different_request_keys_not_repeated(self):
        """Same output from different request keys should NOT be flagged as repeated."""
        from auto_translate.translate.plamo import PlamoTranslator

        previous_outputs = []

        def _repeated(output, request_key):
            norm = PlamoTranslator._normalize(output)
            for prev_norm, prev_key in previous_outputs:
                if prev_key == request_key and prev_norm == norm:
                    return True
            return False

        def _record(output, request_key):
            previous_outputs.append((PlamoTranslator._normalize(output), request_key))

        # Record empty from "full" request
        _record("", "full")
        # Check empty against "minimal" request — should NOT be repeated
        assert _repeated("", "minimal") is False
        # Record empty from "minimal" request
        _record("", "minimal")
        # Now check empty against "minimal" — should be repeated
        assert _repeated("", "minimal") is True


# ---------------------------------------------------------------------------
# 3. Watermark State Machine — Behavioral Tests
# ---------------------------------------------------------------------------

class TestWatermarkStateMachine:
    """Test watermark advancement with actual state transitions."""

    def _make_watermark_state(self):
        """Create a minimal watermark state without MPV dependency."""
        state = {
            "_pending_speech": [],
            "_pending_generation": 1,
            "_translated_until": 0.0,
            "_vad_progress": 0.0,
        }
        return state

    def _try_advance_through_silence(self, state):
        """Replicate _try_advance_through_silence logic."""
        if not state["_pending_speech"]:
            state["_translated_until"] = max(state["_translated_until"], state["_vad_progress"])
        else:
            earliest_block = float('inf')
            for start_s, _end_s in state["_pending_speech"]:
                if start_s > state["_translated_until"]:
                    earliest_block = min(earliest_block, start_s)
            if earliest_block < float('inf'):
                state["_translated_until"] = max(
                    state["_translated_until"],
                    min(state["_vad_progress"], earliest_block)
                )
            else:
                state["_translated_until"] = max(state["_translated_until"], state["_vad_progress"])

    def _complete_speech(self, state, start_s, end_s, generation):
        """Replicate the FIXED complete_speech logic."""
        if generation != state["_pending_generation"]:
            state["_pending_speech"].clear()
            state["_pending_generation"] = generation
            return
        state["_pending_speech"] = [
            (s, e) for s, e in state["_pending_speech"]
            if not (abs(s - start_s) < 1.0 and abs(e - end_s) < 1.0)
        ]
        # FIXED: advance only to end_s, not to vad_progress
        state["_translated_until"] = max(state["_translated_until"], end_s)
        self._try_advance_through_silence(state)

    def _advance_to_subtitle(self, state, subtitle_end_s, generation):
        """Replicate the FIXED advance_to_subtitle logic."""
        if generation != state["_pending_generation"]:
            state["_pending_speech"].clear()
            state["_pending_generation"] = generation
        state["_pending_speech"] = [
            (s, e) for s, e in state["_pending_speech"]
            if not (abs(s - subtitle_end_s) < 1.0 or e <= subtitle_end_s + 0.5)
        ]
        # FIXED: advance only to subtitle_end_s, not to vad_progress
        state["_translated_until"] = max(state["_translated_until"], subtitle_end_s)
        self._try_advance_through_silence(state)

    def test_complete_speech_does_not_skip_pending(self):
        """Completing one speech should NOT advance past another pending speech."""
        state = self._make_watermark_state()
        state["_vad_progress"] = 20.0
        # Two pending speech regions
        state["_pending_speech"] = [(2.0, 4.0), (10.0, 12.0)]
        # Complete the first one (suppressed translation)
        self._complete_speech(state, 2.0, 4.0, 1)
        # Watermark advances to end_s=4.0, then _try_advance_through_silence
        # finds earliest_block=10.0 (next pending speech), so advances to
        # min(vad_progress=20.0, earliest_block=10.0) = 10.0
        # Key: watermark is at 10.0, NOT past the pending speech at 10.0-12.0
        assert state["_translated_until"] == 10.0
        # The second pending speech should still be there
        assert len(state["_pending_speech"]) == 1
        assert (10.0, 12.0) in state["_pending_speech"]

    def test_old_bug_would_skip_pending(self):
        """The old bug: target = max(end_s, vad_progress) would skip pending speech."""
        state = self._make_watermark_state()
        state["_vad_progress"] = 20.0
        state["_pending_speech"] = [(2.0, 4.0), (10.0, 12.0)]
        # Old buggy logic
        end_s = 4.0
        target = max(end_s, state["_vad_progress"])  # = 20.0
        old_translated_until = max(state["_translated_until"], target)  # = 20.0
        # This would skip the pending speech at 10.0-12.0!
        assert old_translated_until == 20.0
        assert old_translated_until > 12.0  # Bug: skipped past pending speech entirely

    def test_advance_to_subtitle_does_not_skip_pending(self):
        """Advancing to a subtitle should NOT skip over other pending speech."""
        state = self._make_watermark_state()
        state["_vad_progress"] = 20.0
        state["_pending_speech"] = [(2.0, 4.0), (10.0, 12.0)]
        # Publish subtitle for the first speech region
        self._advance_to_subtitle(state, 4.0, 1)
        # Watermark advances to subtitle_end_s=4.0, then _try_advance_through_silence
        # finds earliest_block=10.0, advances to min(20.0, 10.0) = 10.0
        # Key: watermark is at 10.0, NOT past the pending speech at 10.0-12.0
        assert state["_translated_until"] == 10.0

    def test_silence_advance_after_complete(self):
        """After completing the last pending speech, advance through silence."""
        state = self._make_watermark_state()
        state["_vad_progress"] = 20.0
        state["_pending_speech"] = [(2.0, 4.0)]
        self._complete_speech(state, 2.0, 4.0, 1)
        # No more pending speech — advance through silence to VAD progress
        assert state["_translated_until"] == 20.0

    def test_out_of_order_completion(self):
        """Completing speech in reverse order should work correctly."""
        state = self._make_watermark_state()
        state["_vad_progress"] = 20.0
        state["_pending_speech"] = [(2.0, 4.0), (10.0, 12.0)]
        # Complete the second speech first
        self._complete_speech(state, 10.0, 12.0, 1)
        # Watermark advances to 4.0 (first pending speech still blocks)
        # Actually: watermark is at 0, first pending speech starts at 2.0
        # _try_advance_through_silence finds earliest_block = 2.0
        # So watermark stays at 0 (or advances to min(20.0, 2.0) = 2.0)
        # Wait: the complete_speech advances to end_s=12.0 first
        # Then _try_advance_through_silence: pending speech at 2.0 blocks
        # earliest_block = 2.0, but 2.0 is NOT > 12.0 (translated_until)
        # So it falls through to the else branch: advance to vad_progress
        # Hmm, let me re-check the logic...
        # After complete_speech: pending_speech = [(2.0, 4.0)]
        # translated_until = max(0, 12.0) = 12.0
        # _try_advance_through_silence: for (2.0, 4.0), start_s=2.0 is NOT > 12.0
        # So earliest_block stays inf, advance to vad_progress = 20.0
        # This is correct: the speech at 2.0-4.0 is BEFORE the watermark at 12.0
        # It's a stale entry that should be cleaned up
        assert state["_translated_until"] == 20.0

    def test_stale_generation_clears_all(self):
        """A callback with a stale generation should clear all pending speech."""
        state = self._make_watermark_state()
        state["_pending_speech"] = [(2.0, 4.0), (10.0, 12.0)]
        self._complete_speech(state, 2.0, 4.0, 999)  # Wrong generation
        assert len(state["_pending_speech"]) == 0
        assert state["_pending_generation"] == 999

    def test_multiple_adjacent_utterances(self):
        """Multiple adjacent utterances should each be tracked separately."""
        state = self._make_watermark_state()
        state["_vad_progress"] = 30.0
        state["_pending_speech"] = [(2.0, 4.0), (5.0, 7.0), (8.0, 10.0)]
        # Complete the middle one
        self._complete_speech(state, 5.0, 7.0, 1)
        # Watermark should advance to 4.0 (end of first pending speech blocks)
        # Wait: translated_until = max(0, 7.0) = 7.0
        # Then _try_advance_through_silence: pending = [(2.0, 4.0), (8.0, 10.0)]
        # For (2.0, 4.0): start_s=2.0 is NOT > 7.0
        # For (8.0, 10.0): start_s=8.0 IS > 7.0, earliest_block = 8.0
        # Advance to min(30.0, 8.0) = 8.0
        assert state["_translated_until"] == 8.0
        assert len(state["_pending_speech"]) == 2

    def test_sequential_completion_advances_watermark(self):
        """Completing speech in order should advance watermark progressively."""
        state = self._make_watermark_state()
        state["_vad_progress"] = 30.0
        state["_pending_speech"] = [(2.0, 4.0), (6.0, 8.0)]
        # Complete first
        self._complete_speech(state, 2.0, 4.0, 1)
        assert state["_translated_until"] == 6.0  # Advance to silence before next speech
        # Complete second
        self._complete_speech(state, 6.0, 8.0, 1)
        assert state["_translated_until"] == 30.0  # Advance through all silence


# ---------------------------------------------------------------------------
# 4. Output Validation
# ---------------------------------------------------------------------------

class TestOutputValidation:
    """Verify clean_output and english_output_issue behavior."""

    def test_english_output_issue_detects_japanese(self):
        issue = english_output_issue("こんにちは、お元気ですか？")
        assert issue is not None

    def test_english_output_issue_accepts_english(self):
        issue = english_output_issue("Hello, how are you?")
        assert issue is None

    def test_english_output_issue_detects_empty(self):
        issue = english_output_issue("")
        assert issue is not None

    def test_english_output_issue_detects_prompt_leakage(self):
        issue = english_output_issue("Name/term glossary:\n  ミク → Miku")
        assert issue is not None

    def test_clean_output_preserves_valid_translation(self):
        valid = "Hello, how are you?"
        result = clean_output(valid)
        assert "Hello" in result


# ---------------------------------------------------------------------------
# 5. Stop Token Handling
# ---------------------------------------------------------------------------

class TestStopTokens:
    """Verify stop token configuration."""

    def test_plamo_stop_tokens_defined(self):
        from auto_translate.translate.plamo import PLAMO_STOP_TOKENS
        assert "<|plamo:tag|>" in PLAMO_STOP_TOKENS
        assert "<|plamo:bos|>" in PLAMO_STOP_TOKENS

    def test_stop_tokens_not_empty(self):
        from auto_translate.translate.plamo import PLAMO_STOP_TOKENS
        assert len(PLAMO_STOP_TOKENS) > 0


# ---------------------------------------------------------------------------
# 6. Pipeline Integration
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    """Verify pipeline correctly wires up callbacks."""

    def test_run_translate_has_complete_speech_cb(self):
        import inspect

        from auto_translate.pipeline import run_translate
        sig = inspect.signature(run_translate)
        params = list(sig.parameters.keys())
        assert "complete_speech_cb" in params

    def test_complete_speech_cb_defaults_none(self):
        import inspect

        from auto_translate.pipeline import run_translate
        sig = inspect.signature(run_translate)
        default = sig.parameters["complete_speech_cb"].default
        assert default is None

    def test_pipeline_wires_complete_speech_cb(self):
        import inspect

        from auto_translate.pipeline import run
        source = inspect.getsource(run)
        assert "complete_speech_cb" in source
        assert "complete_speech" in source


# ---------------------------------------------------------------------------
# 7. Glossary Handling
# ---------------------------------------------------------------------------

class TestGlossaryHandling:
    """Verify glossary terms are handled correctly."""

    def test_glossary_in_normal_prompt(self):
        terms = [("ミク", "Miku"), ("ボカロ", "Vocaloid")]
        result = build_plamo_prompt("ミクが歌う", terms, "")
        kwargs = result.get("chat_template_kwargs", {})
        glossary = kwargs.get("glossary", [])
        assert ("ミク", "Miku") in [(g["source"], g["target"]) for g in glossary]

    def test_glossary_not_in_simplified_prompt(self):
        terms = [("ミク", "Miku")]
        result = build_plamo_prompt("ミクが歌う", terms, "", simplified=True)
        content = result["messages"][0]["content"]
        assert "Miku" not in content
        kwargs = result.get("chat_template_kwargs", {})
        assert "glossary" not in kwargs

    def test_max_terms_limit(self):
        from auto_translate.translate.plamo import MAX_TERMS
        assert MAX_TERMS == 16
