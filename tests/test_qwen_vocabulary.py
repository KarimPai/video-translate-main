"""CPU-only safety and A/B/A tests for optional Qwen3-ASR vocabulary hints."""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from auto_translate.asr.qwen3_asr import Qwen3ASR
from auto_translate.config import AsrConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_haas_qwen_windows as bench


def fake_backend():
    instance = Qwen3ASR.__new__(Qwen3ASR)
    instance.cfg = AsrConfig()
    instance._generation, instance._closed = 0, False
    instance._inference_lock = threading.Lock()
    instance._state_lock = threading.Lock()
    instance.last_detected_language = "und"
    instance.last_raw_transcript = ""
    messages = []

    def post(url, *, json, timeout):
        messages.append(json["messages"])
        assert json["temperature"] in (0.0, 0.3)
        return SimpleNamespace(
            json=lambda: {
                "choices": [{
                    "message": {"content": "language Japanese<asr_text>みこさん"},
                    "finish_reason": "stop",
                }]
            },
            raise_for_status=lambda: None,
        )

    instance.server = SimpleNamespace(
        base_url="http://127.0.0.1:0",
        client=SimpleNamespace(post=post),
    )
    return instance, messages


def test_default_payload_stays_audio_only():
    model, messages = fake_backend()
    audio = np.zeros(16000, np.float32)
    assert model._transcribe_impl(audio, timeout=10) == "みこさん"
    assert len(messages[0]) == 1
    assert messages[0][0]["role"] == "user"
    assert messages[0][0]["content"][0]["type"] == "input_audio"


def test_vocabulary_system_message_does_not_force_japanese_language():
    model, messages = fake_backend()
    model.set_vocabulary_hint(" みこ, ラミィ, ルイ ")
    assert model._transcribe_impl(np.zeros(16000, np.float32), timeout=10) == "みこさん"
    assert [m["role"] for m in messages[0]] == ["system", "user"]
    assert messages[0][0]["content"] == [
        {"type": "text", "text": "Vocabulary: みこ, ラミィ, ルイ"},
    ]
    assert messages[0][1]["content"][0]["type"] == "input_audio"
    assert all(m["role"] != "assistant" for m in messages[0])


@pytest.mark.parametrize("kwargs", [
    {"no_context": True},
    {"probe": True},
])
def test_unprompted_recovery_and_probe_remove_vocabulary(kwargs):
    model, messages = fake_backend()
    model.set_vocabulary_hint("みこ, ルイ")
    model._transcribe_impl(np.zeros(16000, np.float32), timeout=10, **kwargs)
    assert [m["role"] for m in messages[0]] == ["user"]


def test_clear_hint_and_bounds_enforced():
    model, messages = fake_backend()
    model.set_vocabulary_hint("みこ\nルイ\nラミィ")
    model._transcribe_impl(np.zeros(16000, np.float32), timeout=10)
    assert messages[0][0]["content"][0]["text"] == "Vocabulary: みこ, ルイ, ラミィ"
    model.set_vocabulary_hint("")
    model._transcribe_impl(np.zeros(16000, np.float32), timeout=10)
    assert [m["role"] for m in messages[1]] == ["user"]
    with pytest.raises(ValueError, match="12 terms"):
        model.set_vocabulary_hint(",".join(f"term{i}" for i in range(13)))
    with pytest.raises(TypeError, match="string"):
        model.set_vocabulary_hint(["みこ"])


def test_echo_recovery_sends_unprompted_decode_not_hints():
    model, messages = fake_backend()
    model.set_vocabulary_hint("みこ, ルイ")
    assert model.recover_context_echo(np.zeros(16000, np.float32), timeout=10) == "みこさん"
    assert [m["role"] for m in messages[0]] == ["user"]


def test_aba_runs_baseline_hint_baseline_and_restores_setting():
    class FakeQwen:
        last_rejection = None
        last_detected_language = "ja"
        _prefer_split_on_truncation = True

        def __init__(self):
            self._vocabulary_hint = "previous"
            self.hints = []

        def set_vocabulary_hint(self, value):
            self._vocabulary_hint = value

        def forget(self):
            pass

        def transcribe(self, audio, probe=False):
            self.hints.append(self._vocabulary_hint)
            if self._vocabulary_hint == bench.DEFAULT_QWEN_VOCAB:
                return "ルイさん"
            return "ルイスさん"

    model = FakeQwen()
    rows = bench.run_model(
        model, "qwen3-asr",
        {"HG07": (254.18, np.zeros(14 * bench.SAMPLE_RATE, np.float32))},
        qwen_vocabulary=bench.DEFAULT_QWEN_VOCAB,
    )
    assert [r["variant"] for r in rows] == [
        "full", "vocab_hint", "control_repeat", "split", "split",
    ]
    assert model.hints[:3] == ["", bench.DEFAULT_QWEN_VOCAB, ""]
    assert model._vocabulary_hint == "previous"
    assert [r["text"] for r in rows[:3]] == ["ルイスさん", "ルイさん", "ルイスさん"]
    report = bench.english_report(rows, "clip.flac")
    assert "vocabulary hint experiment (A/B/A)" in report
    assert "Hint-only change" in report
    assert "yes | yes |" in report


def test_aba_marks_baseline_variability_without_claiming_hint_improvement():
    class FakeQwen:
        last_rejection = None
        last_detected_language = "ja"
        _prefer_split_on_truncation = True

        def __init__(self):
            self._vocabulary_hint = ""
            self.calls = 0

        def set_vocabulary_hint(self, value):
            self._vocabulary_hint = value

        def forget(self):
            pass

        def transcribe(self, audio, probe=False):
            self.calls += 1
            return ["ルイさん", "ルイさん", "ルイスさん", "ルイさん", "ルイさん"][self.calls - 1]

    rows = bench.run_model(
        FakeQwen(), "qwen3-asr",
        {"HG07": (254.18, np.zeros(14 * bench.SAMPLE_RATE, np.float32))},
        qwen_vocabulary=bench.DEFAULT_QWEN_VOCAB,
    )
    report = bench.english_report(rows, "clip.flac")
    assert "baseline_varies" in report
    assert "NO | no | baseline_varies" in report


def test_hint_on_unverified_silence_is_flagged_not_called_success():
    rows = [
        {"window": "HG27", "model": "qwen3-asr", "variant": v,
         "ja_chars": len(text), "text": text, "language": "ja",
         "rejection": "", "elapsed_s": 0.1}
        for v, text in [
            ("full", ""), ("vocab_hint", "みこ"), ("control_repeat", ""),
        ]
    ]
    report = bench.english_report(rows, "clip.flac")
    assert "hint_on_unverified_silence" in report


def test_experimental_vocabulary_flag_conflicts_with_other_arms(tmp_path, capsys):
    clip = tmp_path / "test.flac"
    config = tmp_path / "config.toml"
    clip.write_bytes(b"fake")
    config.write_text("", encoding="utf-8")
    assert bench.main([
        str(clip), "--config", str(config), "--dry-run",
        "--qwen-vocabulary", "--qwen-recover-rejected",
    ]) == 2
    assert "separately" in capsys.readouterr().err



def test_vocab_flag_requires_qwen_and_nonempty_terms(tmp_path, capsys):
    clip = tmp_path / "test.flac"
    config = tmp_path / "config.toml"
    clip.write_bytes(b"fake")
    config.write_text("", encoding="utf-8")
    assert bench.main([
        str(clip), "--config", str(config), "--dry-run",
        "--model", "haas", "--qwen-vocabulary",
    ]) == 2
    assert "requires --model qwen3-asr" in capsys.readouterr().err
    assert bench.main([
        str(clip), "--config", str(config), "--dry-run",
        "--model", "qwen3-asr", "--qwen-vocabulary",
        "--qwen-vocabulary-terms", "  ",
    ]) == 2
    assert "must not be blank" in capsys.readouterr().err
