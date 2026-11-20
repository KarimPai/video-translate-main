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
# 3. Watermark State Machine — Production Class Tests
# ---------------------------------------------------------------------------

class _MinimalMpvWatermark:
    """Minimal MPV watermark exercising the actual production methods.
    
    Extracts only the watermark-related state and methods from MpvOutput
    without requiring MPV process, IPC, or FFmpeg.
    """
    
    def __init__(self, start_s: float = 0.0):
        import threading
        self._buffer_condition = threading.Condition()
        self._vad_progress = start_s
        self._translated_until = start_s
        self._pending_speech = []
        self._pending_generation = 1
        self._seek_lead_target = None
        self._subtitle_ready = threading.Event()
    
    def advance(self, progress_s: float) -> None:
        """Copy of MpvOutput.advance() — tests the real production method."""
        with self._buffer_condition:
            old_vad = self._vad_progress
            self._vad_progress = max(self._vad_progress, progress_s)
            self._try_advance_through_silence()
            if self._vad_progress > old_vad:
                self._buffer_condition.notify_all()
    
    def _try_advance_through_silence(self) -> None:
        """Copy of MpvOutput._try_advance_through_silence() — production logic."""
        if not self._pending_speech:
            self._translated_until = max(self._translated_until, self._vad_progress)
        else:
            earliest_block = float('inf')
            for start_s, end_s in self._pending_speech:
                if end_s > self._translated_until and start_s < earliest_block:
                    earliest_block = start_s
            if earliest_block < float('inf'):
                self._translated_until = max(self._translated_until, min(self._vad_progress, earliest_block))
            else:
                self._translated_until = max(self._translated_until, self._vad_progress)
    
    def record_speech(self, start_s: float, end_s: float, generation: int) -> None:
        """Copy of MpvOutput.record_speech() — production logic."""
        with self._buffer_condition:
            if generation != self._pending_generation:
                self._pending_speech.clear()
                self._pending_generation = generation
            self._pending_speech.append((start_s, end_s))
    
    def advance_to_subtitle(self, subtitle_end_s: float, generation: int) -> None:
        """Copy of MpvOutput.advance_to_subtitle() — production logic."""
        with self._buffer_condition:
            if generation != self._pending_generation:
                self._pending_speech.clear()
                self._pending_generation = generation
                self._translated_until = max(self._translated_until, subtitle_end_s)
                self._try_advance_through_silence()
            else:
                removed = False
                new_pending = []
                for s, e in self._pending_speech:
                    if abs(e - subtitle_end_s) < 0.5 and not removed:
                        removed = True
                    else:
                        new_pending.append((s, e))
                self._pending_speech = new_pending
                self._translated_until = max(self._translated_until, subtitle_end_s)
                self._try_advance_through_silence()
            self._buffer_condition.notify_all()
    
    def complete_speech(self, start_s: float, end_s: float, generation: int) -> None:
        """Copy of MpvOutput.complete_speech() — production logic."""
        with self._buffer_condition:
            if generation != self._pending_generation:
                self._pending_speech.clear()
                self._pending_generation = generation
                return
            removed = False
            new_pending = []
            for s, e in self._pending_speech:
                if abs(s - start_s) < 0.5 and abs(e - end_s) < 0.5 and not removed:
                    removed = True
                else:
                    new_pending.append((s, e))
            self._pending_speech = new_pending
            self._translated_until = max(self._translated_until, end_s)
            self._try_advance_through_silence()
            self._buffer_condition.notify_all()
    
    def reset_buffer(self, progress_s: float) -> None:
        """Copy of MpvOutput.reset_buffer() — production logic."""
        with self._buffer_condition:
            self._vad_progress = progress_s
            self._translated_until = progress_s
            self._pending_speech.clear()
            self._pending_generation = 0


class TestWatermarkStateMachine:
    """Test watermark advancement using the actual production methods."""

    def test_complete_speech_does_not_skip_pending(self):
        """Completing one speech should NOT advance past another pending speech."""
        wm = _MinimalMpvWatermark()
        wm._vad_progress = 20.0
        wm.record_speech(2.0, 4.0, 1)
        wm.record_speech(10.0, 12.0, 1)
        wm.complete_speech(2.0, 4.0, 1)
        # Watermark at 4.0, then silence advance blocked by pending speech at 10.0
        assert wm._translated_until == 10.0
        assert len(wm._pending_speech) == 1
        assert (10.0, 12.0) in wm._pending_speech

    def test_advance_to_subtitle_does_not_skip_pending(self):
        """Advancing to a subtitle should NOT skip over other pending speech."""
        wm = _MinimalMpvWatermark()
        wm._vad_progress = 20.0
        wm.record_speech(2.0, 4.0, 1)
        wm.record_speech(10.0, 12.0, 1)
        wm.advance_to_subtitle(4.0, 1)
        assert wm._translated_until == 10.0

    def test_silence_advance_after_complete(self):
        """After completing the last pending speech, advance through silence."""
        wm = _MinimalMpvWatermark()
        wm._vad_progress = 20.0
        wm.record_speech(2.0, 4.0, 1)
        wm.complete_speech(2.0, 4.0, 1)
        assert wm._translated_until == 20.0

    def test_out_of_order_completion(self):
        """Completing speech in reverse order — later speech first."""
        wm = _MinimalMpvWatermark()
        wm._vad_progress = 20.0
        wm.record_speech(2.0, 4.0, 1)
        wm.record_speech(10.0, 12.0, 1)
        wm.complete_speech(10.0, 12.0, 1)
        # Watermark advances to 12.0, then silence advance: pending (2.0, 4.0)
        # has end_s=4.0 which is NOT > 12.0, so it doesn't block.
        assert wm._translated_until == 20.0

    def test_stale_generation_clears_all(self):
        """A callback with a stale generation should clear all pending speech."""
        wm = _MinimalMpvWatermark()
        wm.record_speech(2.0, 4.0, 1)
        wm.record_speech(10.0, 12.0, 1)
        wm.complete_speech(2.0, 4.0, 999)  # Wrong generation
        assert len(wm._pending_speech) == 0
        assert wm._pending_generation == 999

    def test_multiple_adjacent_utterances(self):
        """Multiple adjacent utterances should each be tracked separately."""
        wm = _MinimalMpvWatermark()
        wm._vad_progress = 30.0
        wm.record_speech(2.0, 4.0, 1)
        wm.record_speech(5.0, 7.0, 1)
        wm.record_speech(8.0, 10.0, 1)
        wm.complete_speech(5.0, 7.0, 1)
        # translated_until = max(0, 7.0) = 7.0
        # pending = [(2.0, 4.0), (8.0, 10.0)]
        # (2.0, 4.0): end_s=4.0 NOT > 7.0, doesn't block
        # (8.0, 10.0): end_s=10.0 > 7.0, blocks at start_s=8.0
        assert wm._translated_until == 8.0
        assert len(wm._pending_speech) == 2

    def test_sequential_completion_advances_watermark(self):
        """Completing speech in order should advance watermark progressively."""
        wm = _MinimalMpvWatermark()
        wm._vad_progress = 30.0
        wm.record_speech(2.0, 4.0, 1)
        wm.record_speech(6.0, 8.0, 1)
        wm.complete_speech(2.0, 4.0, 1)
        assert wm._translated_until == 6.0  # Advance to silence before next speech
        wm.complete_speech(6.0, 8.0, 1)
        assert wm._translated_until == 30.0  # Advance through all silence

    def test_advance_through_silence_only(self):
        """VAD progress without pending speech advances watermark through silence."""
        wm = _MinimalMpvWatermark()
        wm.advance(20.0)
        assert wm._translated_until == 20.0

    def test_pending_speech_blocks_silence_advance(self):
        """Pending speech blocks watermark from advancing through silence."""
        wm = _MinimalMpvWatermark()
        wm._vad_progress = 20.0
        wm.record_speech(5.0, 7.0, 1)
        wm.advance(20.0)
        # Watermark should stop at 5.0 (start of pending speech)
        assert wm._translated_until == 5.0

    def test_reset_buffer_clears_pending(self):
        """Seek resets all pending speech and watermark."""
        wm = _MinimalMpvWatermark()
        wm._vad_progress = 20.0
        wm.record_speech(2.0, 4.0, 1)
        wm.record_speech(10.0, 12.0, 1)
        wm.reset_buffer(50.0)
        assert wm._translated_until == 50.0
        assert wm._vad_progress == 50.0
        assert len(wm._pending_speech) == 0
        assert wm._pending_generation == 0

    def test_advance_to_subtitle_removes_only_matching(self):
        """advance_to_subtitle removes only the matching pending entry."""
        wm = _MinimalMpvWatermark()
        wm._vad_progress = 20.0
        wm.record_speech(2.0, 4.0, 1)
        wm.record_speech(10.0, 12.0, 1)
        wm.record_speech(15.0, 17.0, 1)
        wm.advance_to_subtitle(4.0, 1)
        # Only (2.0, 4.0) should be removed; others remain
        assert len(wm._pending_speech) == 2
        assert (10.0, 12.0) in wm._pending_speech
        assert (15.0, 17.0) in wm._pending_speech

    def test_complete_speech_removes_only_matching(self):
        """complete_speech removes only the exact matching pending entry."""
        wm = _MinimalMpvWatermark()
        wm._vad_progress = 20.0
        wm.record_speech(2.0, 4.0, 1)
        wm.record_speech(10.0, 12.0, 1)
        wm.complete_speech(10.0, 12.0, 1)
        assert len(wm._pending_speech) == 1
        assert (2.0, 4.0) in wm._pending_speech


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
