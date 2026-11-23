"""CPU-only Qwen truncated-audio recovery overlap tests; no servers or GPU."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from auto_translate.asr.qwen3_asr import Qwen3ASR, recovery_split_ranges

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_haas_qwen_windows as bench  # noqa: E402


@pytest.mark.parametrize("seconds", [4, 8, 15, 16, 20])
def test_zero_overlap_preserves_existing_boundary_samples(seconds):
    audio = np.ones(seconds * 16000, dtype=np.float32)
    (a, b), (c, d) = recovery_split_ranges(audio)
    center = len(audio) // 2
    radius = len(audio) // 10
    frame = 1280
    legacy_best = center
    legacy_energy = float("inf")
    for position in range(max(frame, center - radius),
                          min(len(audio) - frame, center + radius) + 1, frame):
        window = audio[position - frame // 2:position + frame // 2]
        energy = float(np.mean(np.square(window.astype(np.float64))))
        if energy < legacy_energy:
            legacy_energy, legacy_best = energy, position
    assert (a, b, c, d) == (0, legacy_best, legacy_best, len(audio))


@pytest.mark.parametrize("overlap_s", [0.16, 0.24])
def test_overlap_keeps_all_samples_and_shares_exact_boundary(overlap_s):
    audio = np.ones(16 * 16000, dtype=np.float32)
    audio[round(8.0 * 16000):round(8.18 * 16000)] = 0
    (a, b), (c, d) = recovery_split_ranges(audio, overlap_s=overlap_s)
    assert a == 0 and d == len(audio)
    assert b > c
    assert b - c == round(overlap_s * 16000 / 2) * 2
    assert np.array_equal(audio[a:b][-b+c:], audio[c:d][:b-c])
    assert 7.6 <= (b+c) / (2*16000) <= 8.4


@pytest.mark.parametrize("overlap_s", [-0.1, 0.31, float("nan")])
def test_overlap_rejects_invalid_bounds(overlap_s):
    audio = np.ones(16 * 16000, dtype=np.float32)
    with pytest.raises(ValueError, match="overlap"):
        recovery_split_ranges(audio, overlap_s)


def test_overlap_rejects_too_short_audio():
    with pytest.raises(ValueError, match="four seconds"):
        recovery_split_ranges(np.ones(3 * 16000, dtype=np.float32))


def test_real_recovery_path_uses_opt_in_overlap_only():
    model = Qwen3ASR.__new__(Qwen3ASR)
    model._closed = False
    model._generation = 0
    model._truncation_recovery_overlap_s = 0.16
    model.last_detected_language = "und"
    model.last_raw_transcript = ""
    audio = np.zeros(8 * 16000, dtype=np.float32)
    calls = []

    def fake_decoder(chunk, probe=False, timeout=None, max_tokens=256, **kwargs):
        calls.append((len(chunk), max_tokens))
        model.last_detected_language = "ja"
        return ("みこさんが" if len(calls) == 1 else "話しています")

    with patch.object(model, "_transcribe_impl", side_effect=fake_decoder):
        result = model._recover_truncated_utterance(
            audio, probe=False, deadline=time.monotonic() + 5.0, generation=0,
        )
    assert result == "みこさんが話しています"
    assert len(calls) == 2
    assert calls[0][0] + calls[1][0] == len(audio) + round(0.16*16000/2)*2
    assert calls[0][1] == calls[1][1] == 512
    assert model.last_detected_language == "ja"


def test_forced_benchmark_uses_same_audio_and_restores_metadata_and_overlap():
    class FakeQwen:
        _generation = 0
        _closed = False
        last_rejection = "previous"
        last_detected_language = "previous"
        last_raw_transcript = "previous"
        _truncation_recovery_overlap_s = 0.05
        _prefer_split_on_truncation = True
        cfg = SimpleNamespace(response_timeout_s=8.0)

        def __init__(self):
            self.overlaps = []

        def forget(self):
            pass

        def transcribe(self, chunk, probe=False):
            self.last_detected_language = "ja"
            self.last_rejection = None
            return "通常の日本語"

        def _recover_truncated_utterance(self, chunk, *, probe, deadline, generation):
            assert generation == 0
            assert deadline >= time.monotonic()
            self.overlaps.append(self._truncation_recovery_overlap_s)
            self.last_detected_language = "ja"
            self.last_raw_transcript = "前後の会話"
            return "前後の会話"

    qwen = FakeQwen()
    audio = np.zeros(16 * 16000, np.float32)
    qwen._truncation_recovery_overlap_s = 0.05
    rows = bench.run_model(
        qwen, "qwen3-asr", {"HG11": (385.0, audio)},
        qwen_trunc_overlap_s=0.16,
    )
    assert [r["variant"] for r in rows] == [
        "full", "trunc_control", "trunc_overlap", "trunc_control_repeat", "split", "split",
    ]
    assert qwen.overlaps == [0.0, 0.16, 0.0]
    assert qwen._truncation_recovery_overlap_s == 0.05
    # Recovery test does not leak language/raw text/last rejection into the
    # next variant, even when the previous full-window run set new metadata.
    assert qwen.last_detected_language == "ja"
    assert qwen.last_raw_transcript == "previous"
    assert qwen.last_rejection is None
    report = bench.english_report(rows, "clip.flac")
    assert "Experimental Qwen token-limit recovery overlap" in report
    assert "Control stable" in report
    assert "yes | no" in report


def test_overlap_experiment_is_separate_from_hints_and_other_variants(tmp_path, capsys):
    audio = tmp_path / "test.flac"
    config = tmp_path / "config.toml"
    audio.write_bytes(b"placeholder")
    config.write_text("", encoding="utf-8")
    assert bench.main([
        str(audio), "--config", str(config), "--model", "qwen3-asr",
        "--dry-run", "--qwen-trunc-overlap-s", "0.16",
        "--qwen-vocabulary",
    ]) == 2
    assert "separately" in capsys.readouterr().err
    assert bench.main([
        str(audio), "--config", str(config), "--model", "haas",
        "--dry-run", "--qwen-trunc-overlap-s", "0.16",
    ]) == 2
    assert "requires --model qwen3-asr" in capsys.readouterr().err
