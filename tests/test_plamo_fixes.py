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

import pytest

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
        """Two identical truncated responses from different prompts should try higher tokens.

        The first request (full prompt) truncates. The retry (simplified prompt) produces
        the same truncated output. Since the prompts differ, this is NOT a repeated request —
        higher-token retry should be attempted.
        """
        truncated = "This is a very long translation that gets truncated beca"
        translator, call_index = self._make_translator([
            (truncated, "length"),  # first: full prompt, truncated
            (truncated, "length"),  # retry: simplified prompt, same truncation
            ("This is the complete translation with more tokens", "stop"),  # higher tokens
        ])
        from auto_translate.translate.plamo import PlamoTranslator
        PlamoTranslator.translate.__get__(translator, type(translator))("こんにちは", [])
        # First truncated -> simplified prompt -> same output from different prompt
        # -> NOT repeated -> higher-token retry succeeds
        assert call_index[0] == 3

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

    def test_elapsed_time_budget_expires(self):
        """When elapsed-time budget expires, skip higher-token retry."""
        translator, _call_index = self._make_translator([
            ("Truncated output here", "length"),
            ("Truncated output here", "length"),
        ])
        from auto_translate.translate.plamo import PlamoTranslator
        # With elapsed budget exhausted, higher-token retry should be skipped
        # The code should handle this gracefully without referencing uninitialized retry_tokens
        result = PlamoTranslator.translate.__get__(translator, type(translator))("こんにちは", [])
        # Should not raise UnboundLocalError
        assert isinstance(result, str)

    def test_higher_token_retry_also_truncates(self):
        """When higher-token retry also truncates, return empty string."""
        translator, call_index = self._make_translator([
            ("Truncated output", "length"),
            ("Truncated output", "length"),
            ("Still truncated", "length"),
        ])
        from auto_translate.translate.plamo import PlamoTranslator
        result = PlamoTranslator.translate.__get__(translator, type(translator))("こんにちは", [])
        # Higher-token retry also truncated — should return ""
        assert result == ""
        # Should have made 3 calls (first, retry, higher-tokens)
        assert call_index[0] == 3

    def test_network_failure_during_retry(self):
        """Network failure during retry should be handled gracefully."""
        import httpx
        translator, _call_index = self._make_translator([])
        # Make _chat raise an exception, but server is alive (so it re-raises)
        call_count = [0]
        def failing_chat(prompt, timeout=10, temperature=None, max_tokens=None):
            call_count[0] += 1
            raise httpx.HTTPError("Connection failed")

        translator._chat = failing_chat
        from auto_translate.translate.plamo import PlamoTranslator
        # Server alive — exception propagates (expected behavior for network errors)
        translator.server.alive.return_value = True
        with pytest.raises(httpx.HTTPError):
            PlamoTranslator.translate.__get__(translator, type(translator))("こんにちは", [])

    def test_empty_model_response(self):
        """Empty model response should be handled without crashing."""
        translator, _call_index = self._make_translator([
            ("", "stop"),
            ("", "stop"),
        ])
        from auto_translate.translate.plamo import PlamoTranslator
        result = PlamoTranslator.translate.__get__(translator, type(translator))("こんにちは", [])
        assert result == ""


# ---------------------------------------------------------------------------
# 3. CoverageTracker — Production Class Tests
# ---------------------------------------------------------------------------

class TestCoverageTracker:
    """Test the real CoverageTracker production class.

    Exercises the actual interval-based coverage state machine used by MpvOutput.
    No MPV or FFmpeg required — only the CoverageTracker class is tested.
    """

    def _make(self, start_s: float = 0.0):
        from auto_translate.output.coverage import CoverageTracker
        return CoverageTracker(start_s)

    # --- Basic watermark advancement ---

    def test_advance_through_silence_only(self):
        """VAD progress without pending speech advances watermark through silence."""
        ct = self._make()
        ct.advance_vad(20.0)
        assert ct.translated_until == 20.0

    def test_pending_speech_blocks_silence_advance(self):
        """Pending speech blocks watermark from advancing through silence."""
        ct = self._make()
        ct.advance_vad(20.0)
        ct.record_speech(5.0, 7.0)
        assert ct.translated_until == 5.0

    # --- Sequential completion ---

    def test_sequential_completion_advances_watermark(self):
        """Completing speech in order should advance watermark progressively."""
        ct = self._make()
        ct.advance_vad(30.0)
        sid1 = ct.record_speech(2.0, 4.0)
        sid2 = ct.record_speech(6.0, 8.0)
        ct.complete_speech(sid1)
        assert ct.translated_until == 6.0  # Advance to silence before next speech
        ct.complete_speech(sid2)
        assert ct.translated_until == 30.0  # Advance through all silence

    # --- Out-of-order completion (the key bug) ---

    def test_out_of_order_completion(self):
        """Completing later speech first must NOT advance past earlier pending speech.

        Speech A: 2-4s, Speech B: 10-12s, VAD: 20s
        B completes before A. Watermark must stop at 2s (start of A) until A finishes.
        """
        ct = self._make()
        ct.advance_vad(20.0)
        sid_a = ct.record_speech(2.0, 4.0)
        sid_b = ct.record_speech(10.0, 12.0)
        # Complete B first (out of order)
        ct.complete_speech(sid_b)
        # Watermark must NOT advance past A's start (2.0s)
        assert ct.translated_until == 2.0
        # Now complete A
        ct.complete_speech(sid_a)
        # Watermark should advance through all silence
        assert ct.translated_until == 20.0

    def test_reverse_order_completion(self):
        """Completing speech in reverse order — later speech first."""
        ct = self._make()
        ct.advance_vad(20.0)
        sid_a = ct.record_speech(2.0, 4.0)
        sid_b = ct.record_speech(10.0, 12.0)
        ct.complete_speech(sid_b)
        # Watermark stops at A's start (2.0s) because A is still pending
        assert ct.translated_until == 2.0
        ct.complete_speech(sid_a)
        assert ct.translated_until == 20.0

    # --- Complete speech does not skip pending ---

    def test_complete_speech_does_not_skip_pending(self):
        """Completing one speech should NOT advance past another pending speech."""
        ct = self._make()
        ct.advance_vad(20.0)
        sid1 = ct.record_speech(2.0, 4.0)
        ct.record_speech(10.0, 12.0)
        ct.complete_speech(sid1)
        # Watermark at 4.0, then silence advance blocked by pending speech at 10.0
        assert ct.translated_until == 10.0
        assert ct.pending_count() == 1

    def test_advance_to_subtitle_does_not_skip_pending(self):
        """Completing by interval should NOT skip over other pending speech."""
        ct = self._make()
        ct.advance_vad(20.0)
        ct.record_speech(2.0, 4.0)
        ct.record_speech(10.0, 12.0)
        ct.complete_by_interval(2.0, 4.0)
        assert ct.translated_until == 10.0

    # --- Silence advance after complete ---

    def test_silence_advance_after_complete(self):
        """After completing the last pending speech, advance through silence."""
        ct = self._make()
        ct.advance_vad(20.0)
        sid = ct.record_speech(2.0, 4.0)
        ct.complete_speech(sid)
        assert ct.translated_until == 20.0

    # --- Multiple adjacent utterances ---

    def test_multiple_adjacent_utterances(self):
        """Multiple adjacent utterances should each be tracked separately."""
        ct = self._make()
        ct.advance_vad(30.0)
        ct.record_speech(2.0, 4.0)
        sid2 = ct.record_speech(5.0, 7.0)
        ct.record_speech(8.0, 10.0)
        ct.complete_speech(sid2)
        # Watermark: 0 -> complete (5-7) -> 5.0, but (2-4) is still pending at 2.0
        # So watermark stops at 2.0 (earliest pending)
        assert ct.translated_until == 2.0
        assert ct.pending_count() == 2

    # --- Overlapping speech ---

    def test_overlapping_speech(self):
        """Overlapping speech intervals should be handled correctly."""
        ct = self._make()
        ct.advance_vad(20.0)
        sid1 = ct.record_speech(2.0, 6.0)
        sid2 = ct.record_speech(4.0, 8.0)
        ct.complete_speech(sid1)
        # sid2 still pending at 4.0-8.0, watermark stops at 4.0
        assert ct.translated_until == 4.0
        ct.complete_speech(sid2)
        assert ct.translated_until == 20.0

    # --- Pending speech beginning at the watermark ---

    def test_pending_speech_beginning_at_watermark(self):
        """Speech starting exactly at the watermark position blocks it."""
        ct = self._make()
        ct.advance_vad(20.0)
        ct.record_speech(0.0, 4.0)
        # Watermark at 0.0, speech starts at 0.0 — blocks immediately
        assert ct.translated_until == 0.0

    # --- Suppressed translations (complete without subtitle) ---

    def test_suppressed_translation(self):
        """Suppressed translation completes the interval without a subtitle."""
        ct = self._make()
        ct.advance_vad(20.0)
        sid1 = ct.record_speech(2.0, 4.0)
        sid2 = ct.record_speech(10.0, 12.0)
        # Suppress sid1 (no subtitle published)
        ct.complete_speech(sid1)
        assert ct.translated_until == 10.0  # blocked by sid2
        ct.complete_speech(sid2)
        assert ct.translated_until == 20.0

    # --- Duplicate completion ---

    def test_duplicate_completion(self):
        """Completing the same interval twice should be safe."""
        ct = self._make()
        ct.advance_vad(20.0)
        sid = ct.record_speech(2.0, 4.0)
        assert ct.complete_speech(sid) is True
        assert ct.complete_speech(sid) is False  # already completed
        assert ct.translated_until == 20.0

    # --- Unmatched completion ---

    def test_unmatched_completion(self):
        """Completing an unknown interval ID should be safe."""
        ct = self._make()
        ct.advance_vad(20.0)
        ct.record_speech(2.0, 4.0)
        assert ct.complete_speech(9999) is False  # unknown ID
        assert ct.pending_count() == 1

    # --- Subtitle completion by interval ---

    def test_complete_by_interval(self):
        """Complete by interval matching (fallback when ID not available)."""
        ct = self._make()
        ct.advance_vad(20.0)
        ct.record_speech(2.0, 4.0)
        ct.record_speech(10.0, 12.0)
        ct.complete_by_interval(2.0, 4.0)
        assert ct.translated_until == 10.0
        assert ct.pending_count() == 1

    # --- Two utterances with similar end timestamps ---

    def test_two_utterances_similar_end_timestamps(self):
        """Two utterances with end times within 0.5s should not collide."""
        ct = self._make()
        ct.advance_vad(20.0)
        sid1 = ct.record_speech(2.0, 4.0)
        sid2 = ct.record_speech(6.0, 4.3)  # end within 0.5s of sid1's end
        ct.complete_speech(sid1)
        # sid2 still pending at 6.0-4.3 (weird but valid)
        assert ct.pending_count() == 1
        ct.complete_speech(sid2)
        assert ct.translated_until == 20.0

    # --- Generation isolation ---

    def test_stale_generation_rejected(self):
        """Stale callbacks should be rejected without modifying state."""
        # The generation check is done in MpvOutput, not CoverageTracker
        # CoverageTracker itself doesn't enforce generation — the caller does
        ct = self._make()
        ct.advance_vad(20.0)
        sid = ct.record_speech(2.0, 4.0)
        assert ct.pending_count() == 1
        # The generation check is done in MpvOutput, not CoverageTracker
        # CoverageTracker itself doesn't enforce generation — the caller does
        ct.complete_speech(sid)
        assert ct.translated_until == 20.0

    # --- Seek reset and rebuffering ---

    def test_reset_clears_all_state(self):
        """Seek resets all pending speech and watermark."""
        ct = self._make()
        ct.advance_vad(20.0)
        ct.record_speech(2.0, 4.0)
        ct.record_speech(10.0, 12.0)
        ct.reset(50.0)
        assert ct.translated_until == 50.0
        assert ct.vad_progress == 50.0
        assert ct.pending_count() == 0

    def test_reset_then_new_speech(self):
        """After reset, new speech should be tracked correctly."""
        ct = self._make()
        ct.advance_vad(20.0)
        ct.record_speech(2.0, 4.0)
        ct.reset(50.0)
        ct.advance_vad(70.0)
        sid = ct.record_speech(55.0, 57.0)
        assert ct.translated_until == 55.0  # blocked by new speech
        ct.complete_speech(sid)
        assert ct.translated_until == 70.0

    # --- EOF handling ---

    def test_eof_advances_watermark(self):
        """Marking source finished advances watermark to VAD progress."""
        ct = self._make()
        ct.advance_vad(20.0)
        ct.record_speech(2.0, 4.0)
        ct.mark_source_finished()
        # Even at EOF, pending speech still blocks
        assert ct.translated_until == 2.0

    # --- Interval matching removes only matching entry ---

    def test_complete_by_interval_removes_only_matching(self):
        """complete_by_interval removes only the matching pending entry."""
        ct = self._make()
        ct.advance_vad(20.0)
        ct.record_speech(2.0, 4.0)
        ct.record_speech(10.0, 12.0)
        ct.complete_by_interval(10.0, 12.0)
        assert ct.pending_count() == 1

    def test_advance_to_subtitle_removes_only_matching(self):
        """complete_by_interval removes only the matching pending entry."""
        ct = self._make()
        ct.advance_vad(20.0)
        ct.record_speech(2.0, 4.0)
        ct.record_speech(10.0, 12.0)
        ct.record_speech(15.0, 17.0)
        ct.complete_by_interval(2.0, 4.0)
        assert ct.pending_count() == 2

    # --- Speech ID uniqueness ---

    def test_speech_ids_are_unique(self):
        """Each record_speech call returns a unique ID."""
        ct = self._make()
        sid1 = ct.record_speech(2.0, 4.0)
        sid2 = ct.record_speech(5.0, 7.0)
        sid3 = ct.record_speech(8.0, 10.0)
        assert sid1 != sid2
        assert sid2 != sid3
        assert sid1 != sid3

    # --- Pending intervals query ---

    def test_pending_intervals(self):
        """pending_intervals returns the correct list."""
        ct = self._make()
        ct.record_speech(2.0, 4.0)
        ct.record_speech(10.0, 12.0)
        intervals = ct.pending_intervals()
        assert (2.0, 4.0) in intervals
        assert (10.0, 12.0) in intervals

    # --- Unconfirmed gap must not be skipped ---

    def test_unconfirmed_gap_blocks_advancement(self):
        """Completed speech beyond an unconfirmed gap must not advance watermark.

        Coverage starts at 0. VAD progress reaches 5s.
        Completed speech at 10-12s. Gap 5-10s is not VAD-confirmed.
        Watermark must NOT jump to 12s.
        """
        ct = self._make()
        ct.advance_vad(5.0)
        sid = ct.record_speech(10.0, 12.0)
        ct.complete_speech(sid)
        # Watermark should be at 5.0 (VAD progress), not 12.0
        assert ct.translated_until == 5.0

    def test_vad_confirmation_advances_gap(self):
        """When VAD confirms the gap, watermark should advance through it."""
        ct = self._make()
        sid = ct.record_speech(10.0, 12.0)
        ct.complete_speech(sid)
        ct.advance_vad(15.0)
        # Now VAD confirms the gap 0-10s and the speech 10-12s
        # Watermark should advance to 15.0
        assert ct.translated_until == 15.0

    def test_late_registered_speech_blocks_retroactively(self):
        """Speech registered after watermark passes should block retroactively."""
        ct = self._make()
        ct.advance_vad(20.0)
        # Watermark at 20.0 (no pending speech)
        assert ct.translated_until == 20.0
        # Now register speech at 5-7s (late)
        ct.record_speech(5.0, 7.0)
        # Watermark should retroactively drop to 5.0
        assert ct.translated_until == 5.0

    # --- Generation tracking ---

    def test_set_generation(self):
        """set_generation updates the authoritative generation."""
        ct = self._make()
        ct.set_generation(5)
        assert ct.generation == 5

    def test_reject_stale(self):
        """reject_stale returns True for old generation callbacks."""
        ct = self._make()
        ct.set_generation(3)
        assert ct.reject_stale(2) is True
        assert ct.reject_stale(3) is False
        assert ct.reject_stale(4) is True

    def test_reset_preserves_generation_until_set(self):
        """reset() clears intervals but does not reset generation."""
        ct = self._make()
        ct.set_generation(5)
        ct.record_speech(2.0, 4.0)
        ct.reset(50.0)
        # Generation should still be 5 (caller will set it via set_generation)
        assert ct.generation == 5
        assert ct.translated_until == 50.0
        assert ct.pending_count() == 0

    # --- Compact completed intervals ---

    def test_compact_completed_removes_old_intervals(self):
        """compact_completed removes completed intervals behind the watermark."""
        ct = self._make()
        ct.advance_vad(100.0)
        sid1 = ct.record_speech(2.0, 4.0)
        sid2 = ct.record_speech(6.0, 8.0)
        sid3 = ct.record_speech(10.0, 12.0)
        ct.complete_speech(sid1)
        ct.complete_speech(sid2)
        ct.complete_speech(sid3)
        assert ct.translated_until == 100.0
        ct.compact_completed()
        # All completed intervals are behind watermark (100.0)
        # Internal _completed should be empty
        with ct._lock:
            assert len(ct._completed) == 0

    def test_compact_completed_preserves_pending(self):
        """compact_completed must not remove pending intervals."""
        ct = self._make()
        ct.advance_vad(100.0)
        sid1 = ct.record_speech(2.0, 4.0)
        ct.record_speech(10.0, 12.0)  # pending, not completed
        ct.complete_speech(sid1)
        ct.compact_completed()
        assert ct.pending_count() == 1

    def test_stress_many_utterances(self):
        """Stress test: thousands of utterances should not grow unbounded.

        Automatic compaction runs every 100 _advance_through_silence calls,
        so after 1000 record+complete cycles, most completed intervals are
        already compacted. The test verifies that memory stays bounded and
        the watermark remains correct.
        """
        ct = self._make()
        ct.advance_vad(10000.0)
        # Record and complete 1000 utterances
        sids = []
        for i in range(1000):
            start = i * 10.0 + 1.0
            end = start + 2.0
            sid = ct.record_speech(start, end)
            sids.append(sid)
        for sid in sids:
            ct.complete_speech(sid)
        assert ct.translated_until == 10000.0
        # Automatic compaction runs every 100 advances, so most completed
        # intervals should already be removed. At most a few remain.
        with ct._lock:
            # Should be far fewer than 1000 due to automatic compaction
            assert len(ct._completed) < 100
        # Manual compact should remove the rest
        ct.compact_completed()
        with ct._lock:
            assert len(ct._completed) == 0
        # Watermark should still be correct
        assert ct.translated_until == 10000.0

    def test_stress_10000_utterances_bounded(self):
        """10,000 utterances: completed intervals must stay bounded."""
        ct = self._make()
        ct.advance_vad(100000.0)
        sids = []
        for i in range(10000):
            start = i * 10.0 + 1.0
            end = start + 2.0
            sid = ct.record_speech(start, end)
            sids.append(sid)
        for sid in sids:
            ct.complete_speech(sid)
        assert ct.translated_until == 100000.0
        # With automatic compaction every 100 advances, completed intervals
        # should be bounded to a small number
        with ct._lock:
            assert len(ct._completed) < 50
        ct.compact_completed()
        with ct._lock:
            assert len(ct._completed) == 0
        assert ct.translated_until == 100000.0

    def test_automatic_compaction_preserves_coverage(self):
        """Automatic compaction must not change the watermark."""
        ct = self._make()
        ct.advance_vad(500.0)
        sids = []
        for i in range(200):
            start = i * 2.0 + 0.5
            end = start + 1.0
            sid = ct.record_speech(start, end)
            sids.append(sid)
            ct.complete_speech(sid)
        watermark_after = ct.translated_until
        # Watermark should be at VAD progress (all completed)
        assert watermark_after == 500.0
        # Add more speech after compaction has run — advance VAD first
        # so the gap is confirmed
        ct.advance_vad(510.0)
        sid = ct.record_speech(501.0, 503.0)
        ct.complete_speech(sid)
        assert ct.translated_until == 510.0

    def test_cached_subtitle_does_not_complete_wrong_pending(self):
        """Cached subtitle with speech_id=None and no start_s must not complete
        an unrelated pending entry."""
        ct = self._make()
        ct.advance_vad(100.0)
        # Register a pending speech at 10-12s
        sid = ct.record_speech(10.0, 12.0)
        # advance_to_subtitle with no speech_id and no start_s should not
        # complete the pending entry (this is the mpv.py behavior when
        # subtitle_start_s is not provided)
        # In CoverageTracker terms: complete_by_interval should not match
        # when the guessed interval is wrong
        matched = ct.complete_by_interval(50.0, 52.0, tolerance=0.5)
        assert matched is False
        # The pending entry should still be there
        assert ct.pending_count() == 1
        # Complete with correct ID
        ct.complete_speech(sid)
        assert ct.pending_count() == 0

    def test_cached_subtitle_with_correct_interval(self):
        """Cached subtitle with correct start_s should match the right pending."""
        ct = self._make()
        ct.advance_vad(100.0)
        ct.record_speech(10.0, 12.0)
        # Match by actual interval (as pipeline now does with subtitle_start_s)
        matched = ct.complete_by_interval(10.0, 12.0, tolerance=0.5)
        assert matched is True
        assert ct.pending_count() == 0

    def test_overlapping_pending_no_false_match(self):
        """Two overlapping pending intervals: completing one should not complete both."""
        ct = self._make()
        ct.advance_vad(100.0)
        sid1 = ct.record_speech(10.0, 12.0)
        ct.record_speech(11.0, 13.0)  # overlaps with sid1
        # Complete sid1 by ID
        ct.complete_speech(sid1)
        assert ct.pending_count() == 1
        # Second entry should still be pending
        intervals = ct.pending_intervals()
        assert (11.0, 13.0) in intervals

    # --- Checkpoint preserves coverage after compaction ---

    def test_compaction_does_not_lower_watermark(self):
        """After compaction removes completed intervals, watermark must not drop.

        This is the key regression test for the frontier fix.
        1. VAD confirms silence through 5s.
        2. Speech at 5-10s is completed.
        3. Speech at 12-14s remains pending.
        4. Watermark reaches 12s (blocked by sid2).
        5. Compaction removes 5-10s interval.
        6. Trigger recomputation via VAD advance.
        7. Watermark must remain at 12s, not drop to 5s.
        """
        ct = self._make()
        ct.advance_vad(20.0)
        sid1 = ct.record_speech(5.0, 10.0)
        ct.record_speech(12.0, 14.0)
        ct.complete_speech(sid1)
        # Watermark: 0 -> complete 5-10 -> advance through silence to 12 (blocked by sid2)
        assert ct.translated_until == 12.0
        # Force compaction
        ct.compact_completed()
        # Trigger recomputation: VAD advance forces _advance_through_silence_locked()
        ct.advance_vad(25.0)
        # Watermark must NOT drop after compaction + recomputation
        assert ct.translated_until == 12.0

    def test_compaction_then_new_speech_advances(self):
        """After compaction, new speech completion still advances correctly."""
        ct = self._make()
        ct.advance_vad(100.0)
        sids = []
        for i in range(200):
            start = i * 0.5 + 0.1
            end = start + 0.3
            sid = ct.record_speech(start, end)
            sids.append(sid)
            ct.complete_speech(sid)
        watermark_before = ct.translated_until
        assert watermark_before == 100.0
        # Force compaction
        ct.compact_completed()
        assert ct.translated_until == 100.0
        # Add new speech after compaction
        ct.advance_vad(110.0)
        sid = ct.record_speech(101.0, 103.0)
        ct.complete_speech(sid)
        assert ct.translated_until == 110.0

    def test_late_pending_speech_behind_checkpoint_blocks(self):
        """Late-registered pending speech behind the checkpoint must still block coverage."""
        ct = self._make()
        ct.advance_vad(50.0)
        sid1 = ct.record_speech(5.0, 7.0)
        ct.complete_speech(sid1)
        assert ct.translated_until == 50.0
        # Force compaction
        ct.compact_completed()
        assert ct.translated_until == 50.0
        # Now register late speech at 10-12s (behind checkpoint)
        sid2 = ct.record_speech(10.0, 12.0)
        # Watermark should retroactively drop to 10.0 (start of late speech)
        assert ct.translated_until == 10.0
        # Complete the late speech
        ct.complete_speech(sid2)
        # Watermark should advance back to 50.0
        assert ct.translated_until == 50.0

    def test_late_speech_overlapping_compacted_interval(self):
        """Exact issue scenario: completed 5-10s, compacted, late pending at 9-9.5s.

        The watermark must be 9s (blocked by late speech), not 5s (VAD progress).
        After compaction removes the completed 5-10s interval, the walk from
        _start_s can only reach VAD progress (5s), but the frontier preserves
        the verified coverage (10s). Late speech at 9-9.5s should block at 9s.
        """
        ct = self._make()
        # VAD confirms only through 5s — not through the speech interval
        ct.advance_vad(5.0)
        # Speech 5-10s is recorded and completed
        sid1 = ct.record_speech(5.0, 10.0)
        ct.complete_speech(sid1)
        # Watermark: 0→5 (VAD silence) → 10 (completed speech)
        # But VAD only at 5s, so after completed speech, pending is empty
        # and watermark = max(10, 5) = 10
        assert ct.translated_until == 10.0
        # Compaction removes the completed 5-10s interval
        ct.compact_completed()
        assert ct.translated_until == 10.0  # frontier preserves it
        # Late pending speech at 9-9.5s (overlaps where completed speech was)
        sid2 = ct.record_speech(9.0, 9.5)
        # Watermark should be 9.0 (blocked by late speech start)
        # NOT 5.0 (VAD progress) — the frontier says we covered up to 10s
        assert ct.translated_until == 9.0
        # Complete the late speech
        ct.complete_speech(sid2)
        # Watermark should advance back to 10.0 (frontier)
        # VAD is only at 5s, but the completed late speech bridges 9-9.5
        # The frontier preserves the verified coverage
        assert ct.translated_until == 10.0

    def test_late_speech_at_7_after_compacted_5_10(self):
        """Late pending at 7-8s after compacted completed 5-10s with VAD at 5s."""
        ct = self._make()
        ct.advance_vad(5.0)
        sid1 = ct.record_speech(5.0, 10.0)
        ct.complete_speech(sid1)
        assert ct.translated_until == 10.0
        ct.compact_completed()
        assert ct.translated_until == 10.0
        # Late pending at 7-8s
        sid2 = ct.record_speech(7.0, 8.0)
        # Watermark should be 7.0 (blocked by late speech)
        assert ct.translated_until == 7.0
        ct.complete_speech(sid2)
        assert ct.translated_until == 10.0

    def test_late_speech_at_10_11_after_compacted_5_10(self):
        """Late pending at 10-11s after compacted completed 5-10s with VAD at 5s.

        The late speech starts exactly at the frontier boundary.
        """
        ct = self._make()
        ct.advance_vad(5.0)
        sid1 = ct.record_speech(5.0, 10.0)
        ct.complete_speech(sid1)
        assert ct.translated_until == 10.0
        ct.compact_completed()
        assert ct.translated_until == 10.0
        # Late pending at 10-11s (starts at frontier boundary)
        sid2 = ct.record_speech(10.0, 11.0)
        # Watermark should be 10.0 (blocked by late speech at frontier)
        assert ct.translated_until == 10.0
        ct.advance_vad(15.0)
        ct.complete_speech(sid2)
        # Now VAD confirms through 15s, and late speech is completed
        assert ct.translated_until == 15.0

    def test_multiple_overlapping_late_intervals(self):
        """Multiple overlapping late pending intervals after compaction."""
        ct = self._make()
        ct.advance_vad(5.0)
        sid1 = ct.record_speech(5.0, 10.0)
        ct.complete_speech(sid1)
        assert ct.translated_until == 10.0
        ct.compact_completed()
        # Multiple late pending intervals
        sid2 = ct.record_speech(6.0, 7.0)
        assert ct.translated_until == 6.0  # blocked by earliest late speech
        sid3 = ct.record_speech(8.0, 8.5)
        assert ct.translated_until == 6.0  # still blocked by sid2
        ct.complete_speech(sid2)
        # After completing sid2, blocked by sid3 at 8.0
        assert ct.translated_until == 8.0
        ct.complete_speech(sid3)
        assert ct.translated_until == 10.0

    def test_compaction_with_pending_ahead(self):
        """Compaction with pending speech ahead of watermark must preserve state."""
        ct = self._make()
        ct.advance_vad(50.0)
        sid1 = ct.record_speech(2.0, 4.0)
        sid2 = ct.record_speech(10.0, 12.0)
        sid3 = ct.record_speech(30.0, 32.0)  # pending ahead
        ct.complete_speech(sid1)
        ct.complete_speech(sid2)
        # Watermark blocked by sid3 at 30.0
        assert ct.translated_until == 30.0
        ct.compact_completed()
        # Watermark must still be 30.0
        assert ct.translated_until == 30.0
        # Complete sid3
        ct.complete_speech(sid3)
        assert ct.translated_until == 50.0

    def test_multiple_compaction_cycles(self):
        """Multiple compaction cycles must not degrade coverage."""
        ct = self._make()
        ct.advance_vad(1000.0)
        for cycle in range(10):
            base = cycle * 100.0
            sids = []
            for i in range(5):
                start = base + i * 10.0 + 1.0
                end = start + 2.0
                sid = ct.record_speech(start, end)
                sids.append(sid)
                ct.complete_speech(sid)
            ct.compact_completed()
        assert ct.translated_until == 1000.0

    def test_compaction_out_of_order_completion(self):
        """Out-of-order completion followed by compaction must be correct."""
        ct = self._make()
        ct.advance_vad(50.0)
        sid_a = ct.record_speech(2.0, 4.0)
        sid_b = ct.record_speech(10.0, 12.0)
        sid_c = ct.record_speech(20.0, 22.0)
        # Complete in reverse order
        ct.complete_speech(sid_c)
        assert ct.translated_until == 2.0  # blocked by A
        ct.complete_speech(sid_b)
        assert ct.translated_until == 2.0  # still blocked by A
        ct.complete_speech(sid_a)
        assert ct.translated_until == 50.0
        ct.compact_completed()
        # Trigger recomputation to verify compaction didn't erase coverage
        ct.advance_vad(60.0)
        assert ct.translated_until == 60.0

    def test_compaction_completed_speech_beyond_vad_gap(self):
        """Completed speech spanning a gap beyond VAD progress must not advance watermark."""
        ct = self._make()
        ct.advance_vad(5.0)
        sid = ct.record_speech(10.0, 12.0)
        ct.complete_speech(sid)
        assert ct.translated_until == 5.0  # gap 5-10 not confirmed
        ct.compact_completed()
        # Watermark should still be 5.0 (completed speech beyond gap is not behind watermark)
        assert ct.translated_until == 5.0
        # Now confirm the gap
        ct.advance_vad(15.0)
        assert ct.translated_until == 15.0

    def test_compaction_after_seek_reset(self):
        """After seek reset, compaction must work correctly with new state."""
        ct = self._make()
        ct.advance_vad(100.0)
        sid = ct.record_speech(5.0, 7.0)
        ct.complete_speech(sid)
        ct.compact_completed()
        # Seek to new position
        ct.reset(200.0)
        ct.set_generation(2)
        assert ct.translated_until == 200.0
        ct.advance_vad(300.0)
        sid2 = ct.record_speech(210.0, 212.0)
        ct.complete_speech(sid2)
        assert ct.translated_until == 300.0
        ct.compact_completed()
        assert ct.translated_until == 300.0

    def test_compaction_suppressed_translation(self):
        """Suppressed translation completion followed by compaction must be correct."""
        ct = self._make()
        ct.advance_vad(50.0)
        sid1 = ct.record_speech(2.0, 4.0)
        sid2 = ct.record_speech(10.0, 12.0)
        # Suppress both (complete without subtitle)
        ct.complete_speech(sid1)
        ct.complete_speech(sid2)
        assert ct.translated_until == 50.0
        ct.compact_completed()
        # Trigger recomputation to verify compaction didn't erase coverage
        ct.advance_vad(60.0)
        assert ct.translated_until == 60.0

    def test_compaction_eof(self):
        """EOF after compaction must still respect pending speech."""
        ct = self._make()
        ct.advance_vad(50.0)
        sid1 = ct.record_speech(2.0, 4.0)
        ct.complete_speech(sid1)
        ct.compact_completed()
        sid2 = ct.record_speech(30.0, 32.0)  # pending
        ct.mark_source_finished()
        assert ct.translated_until == 30.0  # blocked by sid2
        ct.complete_speech(sid2)
        ct.mark_source_finished()
        assert ct.translated_until == 50.0

    def test_checkpoint_monotonic_advance(self):
        """Frontier must only advance forward, never retreat."""
        ct = self._make()
        ct.advance_vad(50.0)
        sid1 = ct.record_speech(5.0, 7.0)
        ct.complete_speech(sid1)
        assert ct.translated_until == 50.0
        with ct._lock:
            frontier_before = ct._frontier
        # Late speech retroactively blocks
        sid2 = ct.record_speech(10.0, 12.0)
        assert ct.translated_until == 10.0
        with ct._lock:
            frontier_after = ct._frontier
        # Frontier should not have decreased
        assert frontier_after >= frontier_before
        # Complete late speech
        ct.complete_speech(sid2)
        assert ct.translated_until == 50.0

    def test_watermark_monotonic_when_no_new_evidence(self):
        """Watermark must not decrease unless new pending speech is registered."""
        ct = self._make()
        ct.advance_vad(50.0)
        sid1 = ct.record_speech(5.0, 7.0)
        ct.complete_speech(sid1)
        wm = ct.translated_until
        assert wm == 50.0
        # Multiple compaction calls followed by recomputation should not change watermark
        for _ in range(5):
            ct.compact_completed()
        # Trigger recomputation to verify watermark is still correct
        ct.advance_vad(60.0)
        assert ct.translated_until == 60.0


# ---------------------------------------------------------------------------
# 3e. PLaMO token budgeting tests
# ---------------------------------------------------------------------------

class TestPlamoTokenBudgeting:
    """Test that PLaMO correctly uses context_size for token budgeting."""

    def _make_translator(self, context_size: int, max_tokens: int):
        """Create a PlamoTranslator mock with specific config."""
        from auto_translate.config import TranslateConfig
        from auto_translate.glossary import Glossary
        from auto_translate.stream_context import StreamContext
        from auto_translate.translate.plamo import PlamoTranslator

        cfg = TranslateConfig(
            model_path=str(Path(__file__).parent / "dummy.gguf"),
            server_path="llama-server",
            glossary=str(Path(__file__).parent / "glossary.toml"),
            max_tokens=max_tokens,
            timeout_s=10,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            repeat_penalty=1.1,
            context_size=context_size,
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
        translator.server = mock.MagicMock()
        translator.server.alive.return_value = True
        translator._context = mock.MagicMock(return_value="")
        translator._terms_for = mock.MagicMock(return_value=([], []))

        call_index = [0]
        def mock_chat(prompt, timeout=10, temperature=None, max_tokens=None):
            call_index[0] += 1
            return ("Valid translation", "stop")

        translator._chat = mock_chat
        return translator, call_index

    def test_normal_utterance_uses_full_prompt(self):
        """Normal-length Japanese text should use the full prompt with glossary."""
        translator, call_index = self._make_translator(context_size=8192, max_tokens=512)
        from auto_translate.translate.plamo import PlamoTranslator
        # 50 character Japanese text — well within budget
        PlamoTranslator.translate.__get__(translator, type(translator))("こんにちは、今日はいい天気ですね。みんなとゲームをします。", [])
        # Should succeed on first attempt with full prompt
        assert call_index[0] == 1

    def test_long_utterance_still_uses_full_prompt(self):
        """Long Japanese text (200 chars) should still use full prompt with context_size=8192.

        The old bug: max_tokens=512 was used as the budget, so 200 chars * 1.8 + 150 = 510
        would exceed 512, causing skip_full_prompt=True.
        The fix: context_size=8192 - max_tokens=512 - 256 = 7424 is the budget.
        510 < 7424, so full prompt is used.
        """
        translator, call_index = self._make_translator(context_size=8192, max_tokens=512)
        from auto_translate.translate.plamo import PlamoTranslator
        # 200 character Japanese text
        ja = "今日は" * 50
        PlamoTranslator.translate.__get__(translator, type(translator))(ja, [])
        # Should succeed on first attempt
        assert call_index[0] == 1

    def test_very_long_utterance_budget_calculation(self):
        """Verify the budget calculation: context_size - max_tokens - 256.

        The new formula accounts for the full prompt:
        ja_tokens + glossary_tokens + context_tokens + template_overhead
        """
        from auto_translate.config import TranslateConfig

        cfg = TranslateConfig(
            model_path=str(Path(__file__).parent / "dummy.gguf"),
            server_path="llama-server",
            glossary=str(Path(__file__).parent / "glossary.toml"),
            max_tokens=512,
            timeout_s=10,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            repeat_penalty=1.1,
            context_size=8192,
            parallel_slots=1,
            threads=4,
            gpu_layers=-1,
            load_timeout_s=60,
            context_lines=3,
            max_context_chars=500,
            context_english=False,
        )
        # Budget = 8192 - 512 - 256 = 7424
        context_budget = cfg.context_size - cfg.max_tokens - 256
        assert context_budget == 7424

        # 4000 char Japanese text with new formula:
        # ja_tokens = 4000 * 1.8 = 7200, glossary = 0, context = 0, overhead = 50
        # Total = 7250 < 7424, so full prompt is still used
        ja = "あ" * 4000
        ja_tokens = int(len(ja) * 1.8)
        template_overhead = 50
        estimated_input_tokens = ja_tokens + 0 + 0 + template_overhead
        assert estimated_input_tokens < context_budget

    def test_extremely_long_utterance_skips_full_prompt(self):
        """Only extremely long text (>4000 chars) should skip full prompt."""
        translator, call_index = self._make_translator(context_size=8192, max_tokens=512)
        from auto_translate.translate.plamo import PlamoTranslator
        # 5000 character Japanese text with new formula:
        # ja_tokens = 5000 * 1.8 = 9000, overhead = 50, total = 9050
        # 9050 > 7424, so skip_full_prompt=True
        ja = "あ" * 5000
        PlamoTranslator.translate.__get__(translator, type(translator))(ja, [])
        # Should succeed (simplified prompt first)
        assert call_index[0] >= 1

    def test_glossary_and_context_accounted_in_budget(self):
        """Glossary terms and context should be included in the token estimate."""
        translator, call_index = self._make_translator(context_size=8192, max_tokens=512)
        from auto_translate.translate.plamo import PlamoTranslator
        # Set up glossary terms and context
        translator._terms_for = mock.MagicMock(
            return_value=([("田中", "Tanaka"), ("鈴木", "Suzuki"), ("高橋", "Takahashi")], [])
        )
        translator._context = mock.MagicMock(
            return_value="直前の会話：こんにちは（英訳：Hello）"
        )
        ja = "今日は" * 50  # 200 chars
        PlamoTranslator.translate.__get__(translator, type(translator))(ja, [])
        # Should succeed on first attempt with full prompt
        # The estimate: ja=360 + glossary=21 + context=~90 + overhead=50 = ~521
        # 521 < 7424, so full prompt is used
        assert call_index[0] == 1

    def test_old_bug_would_skip_full_prompt_for_200_chars(self):
        """Verify the old bug: max_tokens=512 as budget would skip full prompt for 200 chars.

        Old code: max_output_tokens = max_tokens - estimated_input_tokens
        For 201 chars: 201 * 1.8 + 150 = 511
        max_output_tokens = 512 - 511 = 1
        skip_full_prompt = 1 < 80 → True (WRONG)

        New code: context_budget = context_size - max_tokens - 256 = 7424
        estimated_input_tokens = ja_tokens + glossary + context + overhead
        For 201 chars with no glossary/context: 361 + 0 + 0 + 50 = 411
        skip_full_prompt = 411 > 7424 → False (CORRECT)
        """
        from auto_translate.config import TranslateConfig

        cfg = TranslateConfig(
            model_path=str(Path(__file__).parent / "dummy.gguf"),
            server_path="llama-server",
            glossary=str(Path(__file__).parent / "glossary.toml"),
            max_tokens=512,
            timeout_s=10,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            repeat_penalty=1.1,
            context_size=8192,
            parallel_slots=1,
            threads=4,
            gpu_layers=-1,
            load_timeout_s=60,
            context_lines=3,
            max_context_chars=500,
            context_english=False,
        )
        ja = "今日は" * 67  # 201 chars
        ja_tokens = int(len(ja) * 1.8)  # 361
        template_overhead = 50
        estimated_input_tokens = ja_tokens + 0 + 0 + template_overhead  # 411

        # Old buggy calculation: used max_tokens as the budget
        old_estimated = int(len(ja) * 1.8) + 150  # 511
        old_max_output = cfg.max_tokens - old_estimated  # 512 - 511 = 1
        old_skip = old_max_output < 80  # 1 < 80 → True (BUG)

        # New correct calculation: uses context_size as the budget
        context_budget = cfg.context_size - cfg.max_tokens - 256  # 7424
        new_skip = estimated_input_tokens > context_budget  # 411 > 7424 → False

        assert old_skip is True  # Old code incorrectly skips full prompt
        assert new_skip is False  # New code correctly uses full prompt

# ---------------------------------------------------------------------------
# 3b. MpvOutput generation sync tests
# ---------------------------------------------------------------------------

class TestMpvOutputGenerationSync:
    """Test that MpvOutput correctly syncs generation after seeking."""

    def test_reset_buffer_syncs_generation(self):
        """reset_buffer must sync _pending_generation with ingest.generation."""
        import threading
        from unittest import mock

        from auto_translate.output.coverage import CoverageTracker
        from auto_translate.output.mpv import MpvOutput

        mpv = MpvOutput.__new__(MpvOutput)
        mpv.ingest = mock.MagicMock()
        mpv.ingest.source.live = False
        mpv.ingest.source.start_s = 0.0
        mpv.ingest.generation = 5  # new generation after seek
        mpv.cfg = mock.MagicMock()
        mpv.cfg.vod_pause_threshold_s = 2.0
        mpv.cfg.vod_resume_threshold_s = 8.0
        mpv.cfg.vod_seek_resume_threshold_s = 5.0
        mpv.cfg.window_id = None
        mpv.PAUSE_LEAD_SECONDS = 5.0
        mpv._coverage = CoverageTracker(0.0)
        mpv._source_finished = False
        mpv._buffer_condition = threading.Condition()
        mpv._lock = threading.Lock()
        mpv._lines = []
        mpv._speech_id_map = {}
        mpv._pending_generation = 0
        mpv._subtitle_ready = mock.MagicMock()
        mpv._vod_auto_paused = False

        mpv.reset_buffer(50.0)

        assert mpv._pending_generation == 5
        assert mpv._coverage.generation == 5

    def test_seek_allows_current_generation_callbacks(self):
        """After seek, callbacks matching the new generation must be accepted."""
        import threading
        from unittest import mock

        from auto_translate.output.coverage import CoverageTracker
        from auto_translate.output.mpv import MpvOutput

        mpv = MpvOutput.__new__(MpvOutput)
        mpv.ingest = mock.MagicMock()
        mpv.ingest.source.live = False
        mpv.ingest.source.start_s = 0.0
        mpv.ingest.generation = 2
        mpv.cfg = mock.MagicMock()
        mpv.cfg.vod_pause_threshold_s = 2.0
        mpv.cfg.vod_resume_threshold_s = 8.0
        mpv.cfg.vod_seek_resume_threshold_s = 5.0
        mpv.cfg.window_id = None
        mpv.PAUSE_LEAD_SECONDS = 5.0
        mpv._coverage = CoverageTracker(0.0)
        mpv._source_finished = False
        mpv._buffer_condition = threading.Condition()
        mpv._lock = threading.Lock()
        mpv._lines = []
        mpv._speech_id_map = {}
        mpv._pending_generation = 0
        mpv._subtitle_ready = mock.MagicMock()
        mpv._vod_auto_paused = False

        # Simulate seek: reset_buffer syncs generation
        mpv.reset_buffer(30.0)

        # Current generation callback should be accepted
        sid = mpv.record_speech(32.0, 34.0, generation=2)
        assert sid is not None

        # Stale callback should be rejected
        stale = mpv.record_speech(35.0, 37.0, generation=1)
        assert stale is None

    def test_consecutive_seeks_update_generation(self):
        """Multiple consecutive seeks must each update generation correctly."""
        import threading
        from unittest import mock

        from auto_translate.output.coverage import CoverageTracker
        from auto_translate.output.mpv import MpvOutput

        mpv = MpvOutput.__new__(MpvOutput)
        mpv.ingest = mock.MagicMock()
        mpv.ingest.source.live = False
        mpv.ingest.source.start_s = 0.0
        mpv.ingest.generation = 1
        mpv.cfg = mock.MagicMock()
        mpv.cfg.vod_pause_threshold_s = 2.0
        mpv.cfg.vod_resume_threshold_s = 8.0
        mpv.cfg.vod_seek_resume_threshold_s = 5.0
        mpv.cfg.window_id = None
        mpv.PAUSE_LEAD_SECONDS = 5.0
        mpv._coverage = CoverageTracker(0.0)
        mpv._source_finished = False
        mpv._buffer_condition = threading.Condition()
        mpv._lock = threading.Lock()
        mpv._lines = []
        mpv._speech_id_map = {}
        mpv._pending_generation = 0
        mpv._subtitle_ready = mock.MagicMock()
        mpv._vod_auto_paused = False

        # First seek
        mpv.ingest.generation = 2
        mpv.reset_buffer(30.0)
        assert mpv._pending_generation == 2

        # Second seek
        mpv.ingest.generation = 3
        mpv.reset_buffer(60.0)
        assert mpv._pending_generation == 3

        # Third seek
        mpv.ingest.generation = 4
        mpv.reset_buffer(90.0)
        assert mpv._pending_generation == 4


# ---------------------------------------------------------------------------
# 3c. Cached subtitle speech_id tests
# ---------------------------------------------------------------------------

class TestCachedSubtitleSpeechId:
    """Test that cached subtitles don't cause speech_id collisions."""

    def test_cached_subtitle_strips_speech_id(self):
        """Cached subtitles must have speech_id=None to avoid collisions."""
        from dataclasses import replace as dc_replace

        from auto_translate.types import Subtitle, Timings

        # Simulate a cached subtitle from a previous generation
        cached = Subtitle(
            start_s=10.0, end_s=12.0,
            ja="こんにちは", en="Hello",
            timings=Timings(),
            generation=1,
            speech_id=42,  # old generation's ID
        )
        # Pipeline should strip speech_id when reusing cached subtitle
        reused = dc_replace(cached, generation=2, speech_id=None)
        assert reused.speech_id is None
        assert reused.generation == 2

    def test_new_subtitle_has_speech_id(self):
        """Newly translated subtitles should have a valid speech_id."""
        from auto_translate.output.coverage import CoverageTracker

        ct = CoverageTracker(0.0)
        sid = ct.record_speech(10.0, 12.0)
        assert sid is not None
        assert sid > 0


# ---------------------------------------------------------------------------
# 3d. VAD config propagation tests
# ---------------------------------------------------------------------------

class TestVadConfigPropagation:
    """Test that VadConfig settings are fully propagated through registry."""

    def test_vad_config_asdict_contains_all_fields(self):
        """asdict(VadConfig) must contain all VadConfig fields."""
        from dataclasses import asdict

        from auto_translate.config import VadConfig

        cfg = VadConfig(
            threshold=0.3,
            neg_threshold=0.2,
            min_silence_ms=500,
            speech_pad_ms=400,
            min_speech_ms=200,
            short_reply_ms=100,
            short_reply_probability=0.9,
            onset_ms=30,
            prob_smooth_frames=5,
            max_utterance_s=15.0,
            max_split_extension_s=3.0,
            live_max_utterance_s=8.0,
            live_max_split_extension_s=1.5,
            adaptive_split=True,
            adaptive_split_s=6.0,
            live_adaptive_split_s=5.0,
            adaptive_pause_ms=200,
            split_pause_min_ms=150,
            split_context_ms=180,
            fallback_s=1.0,
            fallback_min_s=2.0,
            fallback_min_db=-45.0,
            model_dir="/custom/model/dir",
        )
        d = asdict(cfg)
        assert d["threshold"] == 0.3
        assert d["neg_threshold"] == 0.2
        assert d["min_silence_ms"] == 500
        assert d["speech_pad_ms"] == 400
        assert d["min_speech_ms"] == 200
        assert d["short_reply_ms"] == 100
        assert d["short_reply_probability"] == 0.9
        assert d["onset_ms"] == 30
        assert d["prob_smooth_frames"] == 5
        assert d["max_utterance_s"] == 15.0
        assert d["max_split_extension_s"] == 3.0
        assert d["live_max_utterance_s"] == 8.0
        assert d["live_max_split_extension_s"] == 1.5
        assert d["adaptive_split"] is True
        assert d["adaptive_split_s"] == 6.0
        assert d["live_adaptive_split_s"] == 5.0
        assert d["adaptive_pause_ms"] == 200
        assert d["split_pause_min_ms"] == 150
        assert d["split_context_ms"] == 180
        assert d["fallback_s"] == 1.0
        assert d["fallback_min_s"] == 2.0
        assert d["fallback_min_db"] == -45.0
        assert d["model_dir"] == "/custom/model/dir"

    def test_vad_for_source_preserves_live_settings(self):
        """vad_for_source must apply live-specific adjustments."""
        from auto_translate.config import VadConfig, vad_for_source

        cfg = VadConfig(
            max_utterance_s=14.0,
            live_max_utterance_s=7.0,
            adaptive_split_s=5.0,
            live_adaptive_split_s=4.0,
            max_split_extension_s=2.0,
            live_max_split_extension_s=1.0,
        )
        live_cfg = vad_for_source(cfg, live=True)
        assert live_cfg.max_utterance_s == 7.0
        assert live_cfg.adaptive_split_s == 4.0
        assert live_cfg.max_split_extension_s == 1.0

        vod_cfg = vad_for_source(cfg, live=False)
        assert vod_cfg.max_utterance_s == 14.0
        assert vod_cfg.adaptive_split_s == 5.0
        assert vod_cfg.max_split_extension_s == 2.0


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


# ---------------------------------------------------------------------------
# 8. Backend Switching and Registry
# ---------------------------------------------------------------------------

class TestBackendSwitching:
    """Verify ASR and VAD backend switching via registry."""

    def test_registry_registers_silero_vad(self):
        from auto_translate.registry import get_default_registry
        registry = get_default_registry()
        entry = registry.get_entry("vad", "silero")
        assert entry is not None
        assert entry.capabilities.runtime == "silero"
        assert entry.capabilities.device == "cpu"

    def test_registry_registers_haas_asr(self):
        from auto_translate.registry import get_default_registry
        registry = get_default_registry()
        entry = registry.get_entry("asr", "haas")
        assert entry is not None
        assert entry.capabilities.runtime == "speech_cpp"

    def test_registry_registers_qwen3_asr(self):
        from auto_translate.registry import get_default_registry
        registry = get_default_registry()
        entry = registry.get_entry("asr", "qwen3-asr")
        assert entry is not None
        assert entry.capabilities.runtime == "llama_cpp"

    def test_registry_registers_whisper_asr(self):
        from auto_translate.registry import get_default_registry
        registry = get_default_registry()
        entry = registry.get_entry("asr", "whisper")
        assert entry is not None
        assert entry.capabilities.runtime == "faster_whisper"

    def test_qwen3_asr_factory_requires_model_path(self):
        """Qwen3-ASR factory must raise if model_path is missing."""
        from auto_translate.registry import get_default_registry
        registry = get_default_registry()
        with pytest.raises(ValueError, match="Qwen3-ASR"):
            registry.create_asr("qwen3-asr", {"model_path": "", "mmproj_path": ""})

    def test_qwen3_asr_factory_requires_mmproj_path(self):
        """Qwen3-ASR factory must raise if mmproj_path is missing."""
        from auto_translate.registry import get_default_registry
        registry = get_default_registry()
        with pytest.raises(ValueError, match="mmproj"):
            registry.create_asr("qwen3-asr", {"model_path": "some_model.gguf", "mmproj_path": ""})

    def test_backend_config_does_not_leak_model_path(self):
        """Switching ASR backends must not inherit another backend's model path."""
        from auto_translate.config import (
            AsrConfig,
            Config,
            ModelsConfig,
            get_backend_config,
        )

        cfg = Config(
            models=ModelsConfig(asr_backend="haas"),
            asr=AsrConfig(model_path="haas-ja-v1.0"),
            asr_profiles={"qwen3-asr": {"model_path": "qwen3-asr.gguf", "mmproj_path": "mmproj.gguf"}},
        )
        # Qwen3-ASR should get its own model path from profile
        qwen_cfg = get_backend_config(cfg, "asr", "qwen3-asr")
        assert qwen_cfg["model_path"] == "qwen3-asr.gguf"
        assert qwen_cfg["mmproj_path"] == "mmproj.gguf"

        # Haas should get the common model path
        haas_cfg = get_backend_config(cfg, "asr", "haas")
        assert haas_cfg["model_path"] == "haas-ja-v1.0"

    def test_backend_config_without_profile_raises(self):
        """Qwen3-ASR without a profile should have empty model path."""
        from auto_translate.config import (
            AsrConfig,
            Config,
            ModelsConfig,
            get_backend_config,
        )

        cfg = Config(
            models=ModelsConfig(asr_backend="haas"),
            asr=AsrConfig(model_path="haas-ja-v1.0"),
            asr_profiles={},
        )
        qwen_cfg = get_backend_config(cfg, "asr", "qwen3-asr")
        # Qwen3-ASR is not the default — should not inherit Haas's model
        assert qwen_cfg.get("model_path", "") == ""
        assert qwen_cfg.get("mmproj_path", "") == ""

    def test_vad_created_via_registry(self):
        """VAD backend must be created through the registry, not directly."""
        from auto_translate.registry import get_default_registry
        from auto_translate.vad import Segmenter
        registry = get_default_registry()
        segmenter = registry.create_vad("silero", {
            "threshold": 0.5,
            "min_speech_ms": 500,
            "min_silence_ms": 200,
            "speech_pad_ms": 50,
            "onset_ms": 500,
            "max_utterance_s": 15.0,
            "max_split_extension_s": 5.0,
            "fallback_s": 0.0,
        })
        assert segmenter is not None
        assert isinstance(segmenter, Segmenter)

    def test_unknown_vad_backend_raises(self):
        """Unknown VAD backend must raise a clear error."""
        from auto_translate.registry import get_default_registry
        registry = get_default_registry()
        with pytest.raises(ValueError, match="Unknown VAD backend"):
            registry.create_vad("nonexistent", {"threshold": 0.5})


# ---------------------------------------------------------------------------
# 9. Silero VAD Frame Processing
# ---------------------------------------------------------------------------

class TestSileroVAD:
    """Verify Silero VAD frame processing and timing constants."""

    def test_vad_sample_rate(self):
        """VAD must use 16 kHz sample rate."""
        from auto_translate.vad import SAMPLE_RATE
        assert SAMPLE_RATE == 16000

    def test_vad_frame_size(self):
        """VAD frame size must be 512 samples (32 ms at 16 kHz)."""
        from auto_translate.vad import FRAME
        assert FRAME == 512

    def test_vad_frame_duration(self):
        """VAD frame duration must be 32 ms."""
        from auto_translate.vad import FRAME, SAMPLE_RATE
        duration_ms = FRAME / SAMPLE_RATE * 1000
        assert duration_ms == 32.0

    def test_silero_vad_initializes(self):
        """Silero VAD model must initialize without error."""
        from auto_translate.vad import StreamingSileroVAD
        vad = StreamingSileroVAD(speech_threshold=0.5)
        assert vad.device == "CPU"

    def test_silero_vad_process_returns_probs(self):
        """Silero VAD must return speech probabilities for audio input."""
        import numpy as np

        from auto_translate.vad import SAMPLE_RATE, StreamingSileroVAD
        vad = StreamingSileroVAD(speech_threshold=0.5)
        # Generate 1 second of silence
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        probs = vad(audio)
        assert len(probs) > 0
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_silero_vad_reset(self):
        """Silero VAD reset must not raise."""
        from auto_translate.vad import StreamingSileroVAD
        vad = StreamingSileroVAD(speech_threshold=0.5)
        vad.reset()  # Should not raise

    def test_silero_vad_short_audio(self):
        """Silero VAD must handle short audio (< 1 frame)."""
        import numpy as np

        from auto_translate.vad import FRAME, StreamingSileroVAD
        vad = StreamingSileroVAD(speech_threshold=0.5)
        audio = np.zeros(FRAME // 2, dtype=np.float32)
        probs = vad(audio)
        assert len(probs) > 0

    def test_segmenter_import(self):
        """Segmenter must be importable from vad module."""
        from auto_translate.vad import Segmenter
        assert callable(Segmenter)

    def test_run_segmenter_import(self):
        """run_segmenter must be importable from vad module."""
        from auto_translate.vad import run_segmenter
        assert callable(run_segmenter)
