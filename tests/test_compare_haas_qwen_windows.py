"""CPU-only tests: no Haas, Qwen, ffmpeg process or GPU is started."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_haas_qwen_windows as bench  # noqa: E402


def test_two_halves_overlap_without_leaving_a_gap():
    n = 16 * bench.SAMPLE_RATE
    pairs = bench.split_ranges(n)
    assert pairs[0][0] == 0
    assert pairs[1][1] == n
    assert pairs[0][1] > pairs[1][0]
    assert pairs[0][1] - pairs[1][0] == int(0.4 * bench.SAMPLE_RATE)
    assert all((end - start) / bench.SAMPLE_RATE <= 9.0 for start, end in pairs)


def test_reject_tiny_segments():
    with pytest.raises(ValueError, match="3 seconds"):
        bench.split_ranges(2 * bench.SAMPLE_RATE)


def test_similarity_is_a_disagreement_proxy_not_accuracy():
    assert bench.similarity("おはよう！", "おはよう。") == 1.0
    assert bench.similarity("こんにちは", "ありがとう") < 1.0
    assert bench.similarity("", "ありがとう") is None


def test_decoding_exact_window_rejects_invalid_pcm(monkeypatch, tmp_path):
    clip = tmp_path / "clip.flac"
    clip.write_bytes(b"not-real-flac")
    samples = np.arange(16000, dtype=np.float32) / 16000
    calls = []

    def fake_run(cmd, capture_output, check):
        assert capture_output and not check
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout=samples.tobytes(), stderr=b"")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    decoded = bench.decode_window(clip, 385.0, 401.0, "ffmpeg")
    assert np.array_equal(decoded, samples)
    assert "-ss" in calls[0]
    assert "385.000" in calls[0]
    assert "16.000" in calls[0]


def test_run_model_is_model_independent_and_forgets_context():
    class FakeASR:
        last_rejection = None

        def __init__(self):
            self.forgot = 0
            self.requests = []

        def forget(self):
            self.forgot += 1

        def transcribe(self, audio, probe=False):
            self.requests.append((len(audio), probe))
            return "聞こえます"

    model = FakeASR()
    audio = np.zeros(16 * bench.SAMPLE_RATE, np.float32)
    rows = bench.run_model(model, "haas", {"HG11": (385.0, audio)})
    assert len(rows) == 3
    assert [r["variant"] for r in rows] == ["full", "split", "split"]
    assert model.forgot == 3
    assert rows[0]["start_s"] == pytest.approx(385.0)
    assert rows[0]["end_s"] == pytest.approx(401.0)
    assert rows[1]["end_s"] > rows[2]["start_s"]
    assert all(r["ja_chars"] == len("聞こえます") for r in rows)


def test_english_report_keeps_separate_halves_and_warns_no_accuracy():
    rows = [
        {
            "window": "HG11", "model": model, "variant": "full",
            "part": 0, "ja_chars": 4, "text": "聞こえます",
            "rejection": "",
        }
        for model in bench.MODELS
    ]
    report = bench.english_report(rows, "clip.flac")
    assert "HG11" in report
    assert "Not an accuracy ranking" in report
    assert "Japanese reading is required" in report  # "no Japanese reading is required"


def test_dry_run_does_not_require_models_or_ffmpeg(tmp_path):
    clip, config = tmp_path / "clip.flac", tmp_path / "config.toml"
    clip.write_bytes(b"x")
    config.write_text("", encoding="utf-8")
    assert bench.main([str(clip), "--config", str(config), "--dry-run"]) == 0
