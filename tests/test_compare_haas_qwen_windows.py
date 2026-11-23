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


def test_qwen_ab_mode_completes_both_full_decodes_then_short_halves():
    class FakeQwen:
        last_rejection = None
        last_detected_language = "ja"
        _prefer_split_on_truncation = False

        def __init__(self):
            self.flags = []
            self.forgot = 0

        def forget(self):
            self.forgot += 1

        def transcribe(self, audio, probe=False):
            self.flags.append(self._prefer_split_on_truncation)
            return "次回ありがとうございます"

    qwen = FakeQwen()
    audio = np.zeros(16 * bench.SAMPLE_RATE, np.float32)
    rows = bench.run_model(
        qwen, "qwen3-asr", {"HG13": (425.0, audio)},
        qwen_split_first=True,
    )
    assert [r["variant"] for r in rows] == [
        "full", "split_first", "split", "split",
    ]
    assert qwen.flags == [False, True, False, False]
    assert not qwen._prefer_split_on_truncation
    assert qwen.forgot == 4
    report = bench.english_report(rows, "clip.flac")
    assert "Qwen experimental split-first" in report


def test_qwen_truncation_opt_in_skips_redundant_512_long_decode():
    from auto_translate.asr.qwen3_asr import Qwen3ASR, Qwen3ASRTruncationError

    calls = []
    responses = [
        {"choices": [{"message": {"content": "language Japanese<asr_text>途中"}, "finish_reason": "length"}]},
    ]

    def post(url, *, json, timeout):
        calls.append(json["max_tokens"])
        return SimpleNamespace(json=lambda: responses.pop(0),
                               raise_for_status=lambda: None)

    backend = Qwen3ASR.__new__(Qwen3ASR)
    backend.server = SimpleNamespace(base_url="http://127.0.0.1:0",
                                     client=SimpleNamespace(post=post))
    backend._prefer_split_on_truncation = True
    with pytest.raises(Qwen3ASRTruncationError, match="experimental split-first"):
        backend._transcribe_impl(np.zeros(16 * bench.SAMPLE_RATE, np.float32),
                                 timeout=10.0)
    assert calls == [256]


def test_qwen_short_truncation_remains_on_original_512_retry():
    from auto_translate.asr.qwen3_asr import Qwen3ASR

    calls = []
    responses = [
        {"choices": [{"message": {"content": "途中"}, "finish_reason": "length"}]},
        {"choices": [{"message": {"content": "language Japanese<asr_text>ありがとう"},
                      "finish_reason": "stop"}]},
    ]

    def post(url, *, json, timeout):
        calls.append(json["max_tokens"])
        return SimpleNamespace(json=lambda: responses.pop(0),
                               raise_for_status=lambda: None)

    backend = Qwen3ASR.__new__(Qwen3ASR)
    backend.server = SimpleNamespace(base_url="http://127.0.0.1:0",
                                     client=SimpleNamespace(post=post))
    backend._prefer_split_on_truncation = True
    assert backend._transcribe_impl(np.zeros(8 * bench.SAMPLE_RATE, np.float32),
                                    timeout=10.0) == "ありがとう"
    assert calls == [256, 512]


def test_qwen_experimental_flag_is_opt_in():
    from auto_translate.asr.qwen3_asr import Qwen3ASR

    calls = []
    responses = [
        {"choices": [{"message": {"content": "途中"}, "finish_reason": "length"}]},
        {"choices": [{"message": {"content": "language Japanese<asr_text>続き"},
                      "finish_reason": "stop"}]},
    ]

    def post(url, *, json, timeout):
        calls.append(json["max_tokens"])
        return SimpleNamespace(json=lambda: responses.pop(0),
                               raise_for_status=lambda: None)

    backend = Qwen3ASR.__new__(Qwen3ASR)
    backend.server = SimpleNamespace(base_url="http://127.0.0.1:0",
                                     client=SimpleNamespace(post=post))
    assert backend._transcribe_impl(np.zeros(16 * bench.SAMPLE_RATE, np.float32),
                                    timeout=10.0) == "続き"
    assert calls == [256, 512]


def test_qwen_real_constructor_enables_guarded_split_first(monkeypatch):
    """Production default is set without spawning models, processes or GPU."""
    from auto_translate.asr import qwen3_asr as qwen_mod
    from auto_translate.config import AsrConfig

    class FakeServer:
        def __init__(self, **kwargs):
            self.started = False
            self.closed = False

        def start(self):
            self.started = True

        def close(self):
            self.closed = True

    monkeypatch.setattr(qwen_mod, "require_gguf",
                        lambda name, purpose: Path(name))
    monkeypatch.setattr(qwen_mod, "resolve_executable",
                        lambda name: Path("unused-llama-server.exe"))
    monkeypatch.setattr(qwen_mod, "detect_device",
                        lambda executable, device: "CUDA0")
    monkeypatch.setattr(qwen_mod, "LlamaServer", FakeServer)
    monkeypatch.setattr(qwen_mod.Qwen3ASR, "_warmup", lambda self: None)
    monkeypatch.setattr(qwen_mod, "check_running", lambda: None)
    monkeypatch.setattr(qwen_mod, "update_metrics", lambda **kwargs: None)

    model = qwen_mod.Qwen3ASR(AsrConfig(
        model_path="unused-asr.gguf", mmproj_path="unused-mmproj.gguf",
    ))
    try:
        assert model._prefer_split_on_truncation is True
        assert model.server.started
    finally:
        model.close()
    assert model.server.closed


def test_qwen_ab_control_disables_production_default():
    class ProductionDefaultQwen:
        last_rejection = None
        last_detected_language = "ja"
        _prefer_split_on_truncation = True

        def __init__(self):
            self.flags = []

        def forget(self):
            pass

        def transcribe(self, audio, probe=False):
            self.flags.append(self._prefer_split_on_truncation)
            return "日本語の会話"

    model = ProductionDefaultQwen()
    audio = np.zeros(16 * bench.SAMPLE_RATE, np.float32)
    rows = bench.run_model(
        model, "qwen3-asr", {"HG13": (425.0, audio)},
        qwen_split_first=True,
    )
    assert len(rows) == 4
    assert model.flags == [False, True, False, False]
