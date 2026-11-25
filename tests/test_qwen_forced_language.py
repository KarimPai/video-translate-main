"""CPU-only tests for forced-Japanese Qwen3-ASR decoding (asr.language = "ja")."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from auto_translate.asr.qwen3_asr import Qwen3ASR
from auto_translate.config import AsrConfig

PREFIX = "language Japanese<asr_text>"
AUDIO = np.zeros(16000, np.float32)


def fake_backend(language: str = "ja", reply: str = "みこさん"):
    instance = Qwen3ASR.__new__(Qwen3ASR)
    instance.cfg = AsrConfig(language=language)
    instance._generation, instance._closed = 0, False
    instance._inference_lock = threading.Lock()
    instance._state_lock = threading.Lock()
    instance.last_detected_language = "und"
    instance.last_raw_transcript = ""
    sent: list[list[dict]] = []

    def post(url, *, json, timeout):
        sent.append(json["messages"])
        return SimpleNamespace(
            json=lambda: {
                "choices": [{
                    "message": {"content": reply},
                    "finish_reason": "stop",
                }]
            },
            raise_for_status=lambda: None,
        )

    instance.server = SimpleNamespace(
        base_url="http://127.0.0.1:0",
        client=SimpleNamespace(post=post),
    )
    return instance, sent


def test_default_language_is_ja():
    assert AsrConfig().language == "ja"


def test_invalid_language_is_rejected():
    with pytest.raises(ValueError, match="asr.language"):
        AsrConfig(language="en")


def test_forced_japanese_prefills_assistant_turn():
    # llama-server normally returns only the continuation of a prefilled turn.
    model, sent = fake_backend(reply="みこさん")
    assert model._transcribe_impl(AUDIO, timeout=10) == "みこさん"
    assert model.last_detected_language == "ja"
    assert [m["role"] for m in sent[0]] == ["user", "assistant"]
    assert sent[0][-1]["content"] == PREFIX


def test_forced_japanese_accepts_server_that_echoes_the_header():
    model, _ = fake_backend(reply=PREFIX + "みこさん")
    assert model._transcribe_impl(AUDIO, timeout=10) == "みこさん"
    assert model.last_detected_language == "ja"


def test_forced_japanese_empty_continuation_is_not_speech():
    model, _ = fake_backend(reply="")
    assert model._transcribe_impl(AUDIO, timeout=10) == ""


def test_auto_config_sends_audio_only():
    model, sent = fake_backend(language="auto", reply="language Japanese<asr_text>みこさん")
    assert model._transcribe_impl(AUDIO, timeout=10) == "みこさん"
    assert [m["role"] for m in sent[0]] == ["user"]


def test_probe_and_explicit_auto_override_stay_automatic():
    model, sent = fake_backend(reply="language Japanese<asr_text>みこさん")
    model._transcribe_impl(AUDIO, timeout=10, probe=True)
    model._transcribe_impl(AUDIO, timeout=10, language="auto")
    assert [m["role"] for m in sent[0]] == ["user"]
    assert [m["role"] for m in sent[1]] == ["user"]
