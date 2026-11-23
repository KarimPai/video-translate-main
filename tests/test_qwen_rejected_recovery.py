"""CPU-only tests for opt-in selective Qwen3-ASR rejection recovery.

The real CUDA model, ffmpeg, and llama-server are never launched.
"""
from __future__ import annotations

import threading
from collections import deque
from unittest.mock import patch

import numpy as np
import pytest

from auto_translate.asr.qwen3_asr import Qwen3ASR
from auto_translate.config import AsrConfig, Config, get_backend_config


def backend(*, enabled: bool) -> Qwen3ASR:
    instance = Qwen3ASR.__new__(Qwen3ASR)
    instance.cfg = AsrConfig(response_timeout_s=15, recover_rejected_speech=enabled)
    instance._generation = 0
    instance._inference_lock = threading.Lock()
    instance._state_lock = threading.Lock()
    instance._closed = False
    instance._failures = 0
    instance._restarts = 0
    instance._terminal_failure = False
    instance._recent = deque(maxlen=4)
    instance._accepted_keys = deque(maxlen=32)
    instance.last_raw_transcript = ""
    instance.last_detected_language = "und"
    instance.last_rejection = None
    return instance


def fake_decoder(instance, responses):
    """Return (audio length, max tokens, probe) for each decode."""
    calls = []
    iterator = iter(responses)

    def decode(audio, probe=False, timeout=None, max_tokens=256, **kwargs):
        calls.append((len(audio), max_tokens, probe))
        text, language = next(iterator)
        instance.last_detected_language = language
        instance.last_raw_transcript = text
        return text

    return decode, calls


def test_disabled_by_default_rejects_mechanical_loop_without_extra_inference():
    asr = backend(enabled=False)
    decoder, calls = fake_decoder(asr, [("あ" * 12, "ja")])
    with patch.object(asr, "_transcribe_impl", side_effect=decoder):
        result = asr.transcribe(np.zeros(12 * 16000, np.float32))
    assert result == ""
    assert asr.last_rejection == "repeated_character"
    assert len(calls) == 1


def test_opt_in_recovers_only_when_both_short_japanese_windows_are_valid():
    asr = backend(enabled=True)
    decoder, calls = fake_decoder(asr, [
        ("あ" * 12, "ja"), ("みこさんは", "ja"), ("今日も元気です", "ja"),
    ])
    with patch.object(asr, "_transcribe_impl", side_effect=decoder):
        result = asr.transcribe(np.zeros(12 * 16000, np.float32))
    assert result == "みこさんは今日も元気です"
    assert asr.last_rejection is None
    assert asr.last_detected_language == "ja"
    assert asr.last_raw_transcript == result
    assert [c[1] for c in calls] == [256, 512, 512]
    assert calls[1][0] + calls[2][0] == 12 * 16000


def test_opt_in_preserves_normal_recognition_without_extra_decode():
    asr = backend(enabled=True)
    decoder, calls = fake_decoder(asr, [("今日はみこさんと話したよ", "ja")])
    with patch.object(asr, "_transcribe_impl", side_effect=decoder):
        result = asr.transcribe(np.zeros(13 * 16000, np.float32))
    assert result == "今日はみこさんと話したよ"
    assert len(calls) == 1


def test_opt_in_does_not_retry_legitimate_short_reaction():
    asr = backend(enabled=True)
    decoder, calls = fake_decoder(asr, [("え？", "ja")])
    with patch.object(asr, "_transcribe_impl", side_effect=decoder):
        assert asr.transcribe(np.zeros(2 * 16000, np.float32)) == "え？"
    assert len(calls) == 1


@pytest.mark.parametrize("language, duration_s, probe", [
    ("ja", 5, False), ("ja", 10, True), ("en", 10, False),
])
def test_no_recovery_for_short_probe_or_foreign_speech(language, duration_s, probe):
    asr = backend(enabled=True)
    decoder, calls = fake_decoder(asr, [("あ" * 12, language)])
    with patch.object(asr, "_transcribe_impl", side_effect=decoder):
        assert asr.transcribe(np.zeros(duration_s * 16000, np.float32), probe=probe) == ""
    assert asr.last_rejection == "repeated_character"
    assert len(calls) == 1


def test_failed_second_half_restores_original_rejection_and_metadata():
    asr = backend(enabled=True)
    original = "あ" * 12
    decoder, calls = fake_decoder(asr, [
        (original, "ja"), ("今日は話します", "ja"), ("", "none"),
    ])
    with patch.object(asr, "_transcribe_impl", side_effect=decoder):
        result = asr.transcribe(np.zeros(12 * 16000, np.float32))
    assert result == ""
    assert asr.last_rejection == "repeated_character"
    assert asr.last_raw_transcript == original
    assert asr.last_detected_language == "ja"
    assert len(calls) == 3


def test_language_mismatch_does_not_publish_partial_recovery():
    asr = backend(enabled=True)
    original = "あ" * 12
    decoder, calls = fake_decoder(asr, [
        (original, "ja"), ("今日は話す", "ja"), ("hello", "en"),
    ])
    with patch.object(asr, "_transcribe_impl", side_effect=decoder):
        assert asr.transcribe(np.zeros(12 * 16000, np.float32)) == ""
    assert asr.last_rejection == "repeated_character"
    assert asr.last_raw_transcript == original
    assert asr.last_detected_language == "ja"
    assert len(calls) == 3


def test_uncaught_recovery_exception_stays_local_and_preserves_original():
    asr = backend(enabled=True)
    original = "あ" * 12
    decoder, calls = fake_decoder(asr, [(original, "ja")])
    with patch.object(asr, "_transcribe_impl", side_effect=decoder), patch.object(
        asr, "_recover_truncated_utterance", side_effect=RuntimeError("unavailable")
    ):
        assert asr.transcribe(np.zeros(10 * 16000, np.float32)) == ""
    assert asr.last_rejection == "repeated_character"
    assert asr.last_raw_transcript == original
    assert asr.last_detected_language == "ja"
    assert len(calls) == 1


def test_generation_cancellation_prevents_retry():
    asr = backend(enabled=True)

    def cancel_in_first_decode(audio, probe=False, timeout=None, max_tokens=256, **kwargs):
        asr._generation += 1
        asr.last_detected_language = "ja"
        asr.last_raw_transcript = "あ" * 12
        return "あ" * 12

    with patch.object(asr, "_transcribe_impl", side_effect=cancel_in_first_decode) as decode:
        assert asr.transcribe(np.zeros(10 * 16000, np.float32)) == ""
    decode.assert_called_once()


def test_recovery_config_is_boolean_and_backend_profile_isolated():
    with pytest.raises(ValueError, match="recover_rejected_speech"):
        AsrConfig(recover_rejected_speech="true")
    cfg = Config()
    assert cfg.asr.recover_rejected_speech is False
    cfg.asr_profiles["qwen3-asr"] = {"recover_rejected_speech": True}
    selected = get_backend_config(cfg, "asr", "qwen3-asr")
    haas = get_backend_config(cfg, "asr", "haas")
    assert selected["recover_rejected_speech"] is True
    assert haas["recover_rejected_speech"] is False



def test_cancellation_during_second_half_never_publishes_joined_audio():
    asr = backend(enabled=True)
    original = "あ" * 12
    calls = []

    def decode(audio, probe=False, timeout=None, max_tokens=256, **kwargs):
        calls.append(max_tokens)
        asr.last_detected_language = "ja"
        if len(calls) == 1:
            asr.last_raw_transcript = original
            return original
        if len(calls) == 2:
            asr.last_raw_transcript = "こんにちは"
            return "こんにちは"
        # A seek/cancel that lands while the second recovery call is in flight.
        asr._generation += 1
        asr.last_raw_transcript = "みこさん"
        return "みこさん"

    with patch.object(asr, "_transcribe_impl", side_effect=decode):
        assert asr.transcribe(np.zeros(12 * 16000, np.float32)) == ""
    assert calls == [256, 512, 512]
    assert asr.last_rejection == "repeated_character"
    assert asr.last_raw_transcript == original
