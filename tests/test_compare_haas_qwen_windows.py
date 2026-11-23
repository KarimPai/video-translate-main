"""CPU-only tests: no Haas, Qwen, ffmpeg process or GPU is started."""
from __future__ import annotations

import sys
import threading
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


@pytest.mark.parametrize("duration_s", [15.0, 15.8, 16.0])
def test_qwen_truncation_opt_in_skips_redundant_512_long_decode(duration_s):
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
    backend._generation, backend._closed = 0, False
    backend.server = SimpleNamespace(base_url="http://127.0.0.1:0",
                                     client=SimpleNamespace(post=post))
    backend._prefer_split_on_truncation = True
    with pytest.raises(Qwen3ASRTruncationError, match="experimental split-first"):
        backend._transcribe_impl(np.zeros(int(duration_s * bench.SAMPLE_RATE), np.float32),
                                 timeout=10.0)
    assert calls == [256]


@pytest.mark.parametrize("duration_s", [8.0, 14.4, 14.99])
def test_qwen_short_truncation_remains_on_original_512_retry(duration_s):
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
    backend._generation, backend._closed = 0, False
    backend._state_lock = threading.Lock()
    backend.server = SimpleNamespace(base_url="http://127.0.0.1:0",
                                     client=SimpleNamespace(post=post))
    backend._prefer_split_on_truncation = True
    assert backend._transcribe_impl(np.zeros(int(duration_s * bench.SAMPLE_RATE), np.float32),
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
    backend._generation, backend._closed = 0, False
    backend._state_lock = threading.Lock()
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
    monkeypatch.setattr(qwen_mod.Qwen3ASR, "_warmup", lambda self, deadline, generation: None)
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


def test_qwen_split_first_failure_falls_back_to_full_512(monkeypatch):
    """Do not drop a long utterance if split recovery fails but full retry works."""
    from collections import deque
    from auto_translate.config import AsrConfig
    from auto_translate.asr.qwen3_asr import Qwen3ASR, Qwen3ASRTruncationError

    model = Qwen3ASR.__new__(Qwen3ASR)
    model.cfg = AsrConfig(response_timeout_s=15.0)
    model._generation = 0
    model._inference_lock = threading.Lock()
    model._state_lock = threading.Lock()
    model._closed = False
    model._failures = 0
    model._restarts = 0
    model._terminal_failure = False
    model._recent = deque(maxlen=4)
    model._accepted_keys = deque(maxlen=32)
    model._prefer_split_on_truncation = True
    model.last_raw_transcript = ""
    model.last_detected_language = "und"
    model.last_rejection = None

    calls = []

    def decode(samples, probe=False, *, timeout=None, max_tokens=256, **kwargs):
        calls.append((len(samples), max_tokens))
        if len(calls) <= 2:
            raise Qwen3ASRTruncationError("token limit")
        model.last_detected_language = "ja"
        model.last_raw_transcript = "ありがとう"
        return "ありがとう"

    monkeypatch.setattr(model, "_transcribe_impl", decode)
    assert model.transcribe(np.zeros(16 * bench.SAMPLE_RATE, np.float32)) == "ありがとう"
    assert [tokens for _, tokens in calls] == [256, 512, 512]
    assert calls[0][0] == calls[2][0] == 16 * bench.SAMPLE_RATE
    assert calls[1][0] < calls[0][0]
    assert model.last_detected_language == "ja"
    assert model.last_rejection is None



def test_qwen_pause_guided_segments_cover_audio_with_bounded_overlap():
    audio = np.ones(16 * bench.SAMPLE_RATE, np.float32)
    # Intentionally low-energy region near the midpoint; other audio is loud.
    audio[int(7.90 * bench.SAMPLE_RATE):int(8.15 * bench.SAMPLE_RATE)] = 0
    ranges = bench.pause_guided_ranges(audio, target_s=10.0, overlap_s=0.24)
    assert len(ranges) == 2
    assert ranges[0][0] == 0 and ranges[-1][1] == len(audio)
    assert ranges[0][1] > ranges[1][0]
    assert all(right > left and (right - left) <= 10 * bench.SAMPLE_RATE
               for left, right in ranges)
    assert 7.8 <= (ranges[0][1] + ranges[1][0]) / (2 * bench.SAMPLE_RATE) <= 8.2


@pytest.mark.parametrize("length_s,target_s", [(7, 10.0), (11, 10.0),
                                                (14, 10.0), (16, 8.0),
                                                (19, 10.0), (27, 10.0),
                                                (30, 10.0)])
def test_qwen_pause_guided_segments_do_not_drop_samples(length_s, target_s):
    audio = np.zeros(length_s * bench.SAMPLE_RATE, np.float32)
    ranges = bench.pause_guided_ranges(audio, target_s=target_s)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == len(audio)
    assert all(ranges[i][1] >= ranges[i + 1][0] for i in range(len(ranges) - 1))
    assert all((right - left) / bench.SAMPLE_RATE <= target_s + 0.002
               for left, right in ranges)
    if length_s <= target_s:
        assert ranges == [(0, len(audio))]
    else:
        assert len(ranges) >= 2


def test_qwen_pause_guided_segments_reject_invalid_parameters():
    audio = np.zeros(16 * bench.SAMPLE_RATE, np.float32)
    with pytest.raises(ValueError, match="Invalid"):
        bench.pause_guided_ranges(audio, target_s=0.0)
    with pytest.raises(ValueError, match="Invalid"):
        bench.pause_guided_ranges(audio, overlap_s=1.0)


def test_qwen_pause_guided_ab_keeps_production_full_control_and_no_auto_merge():
    class FakeQwen:
        _prefer_split_on_truncation = False
        last_detected_language = "ja"
        last_rejection = None

        def __init__(self):
            self.flags = []
            self.requests = []

        def forget(self):
            pass

        def transcribe(self, audio, probe=False):
            self.flags.append(self._prefer_split_on_truncation)
            self.requests.append(len(audio))
            return "みこさん、ありがとう"

    model = FakeQwen()
    audio = np.zeros(16 * bench.SAMPLE_RATE, np.float32)
    rows = bench.run_model(
        model, "qwen3-asr", {"HG11": (385.0, audio)},
        qwen_short_target_s=10.0,
    )
    assert [row["variant"] for row in rows] == [
        "full", "short", "short", "split", "split",
    ]
    assert rows[0]["audio_s"] == pytest.approx(16.0)
    assert all(row["audio_s"] <= 10.002 for row in rows if row["variant"] == "short")
    assert all(model.flags)  # full control uses current production recovery default
    assert sum(row["ja_chars"] for row in rows if row["variant"] == "short") > rows[0]["ja_chars"]
    report = bench.english_report(rows, "clip.flac")
    assert "pause-guided" in report
    assert "Overlapping cuts can duplicate words" in report
    assert "NOT a production VAD comparison" in report


def test_qwen_shortening_does_not_trigger_on_short_audio():
    class FakeQwen:
        last_rejection = None
        last_detected_language = "ja"

        def forget(self):
            pass

        def transcribe(self, audio, probe=False):
            return "ありがとう"

    audio = np.zeros(5 * bench.SAMPLE_RATE, np.float32)
    rows = bench.run_model(
        FakeQwen(), "qwen3-asr", {"HG02": (15.9, audio)},
        qwen_short_target_s=10.0,
    )
    assert not any(row["variant"] == "short" for row in rows)


def test_standard_qwen_preflight_is_explicit_when_ja_is_also_installed(
        tmp_path, capsys):
    clip = tmp_path / "clip.flac"
    config = tmp_path / "config.toml"
    clip.write_bytes(b"audio-placeholder")
    config.write_text("", encoding="utf-8")
    model_dir = tmp_path / "models" / "qwen3-asr"
    assert bench.main([
        str(clip), "--config", str(config), "--model", "qwen3-asr",
        "--qwen-model-dir", str(model_dir), "--qwen-short-target-s", "10",
    ]) == 2
    assert "Standard Qwen requires" in capsys.readouterr().err




@pytest.mark.parametrize("scene", ["HG06", "HG10", "HG13", "Q16"])
def test_crosstalk_focus_has_nested_audio_and_repeatable_tight_pcm(scene):
    duration = int((bench.WINDOWS[scene][1] - bench.WINDOWS[scene][0]) * bench.SAMPLE_RATE)
    center = bench.CROSSTALK_FOCUS_S[scene][0]
    (a, b), (c, d) = bench.crosstalk_focus_ranges(duration, center)
    assert 0 <= c <= a < b <= d <= duration
    assert (b - a) / bench.SAMPLE_RATE <= 6.002
    assert (d - c) / bench.SAMPLE_RATE <= 10.002
    assert b - a >= 2 * bench.SAMPLE_RATE
    assert d - c > b - a


@pytest.mark.parametrize("center,tight,wide", [
    (float("nan"), 6, 10), (5, 0, 10), (5, 10, 6),
    (0, 6, 10), (30, 6, 10), (5, 6, float("inf")),
])
def test_crosstalk_focus_rejects_unsafe_ranges(center, tight, wide):
    with pytest.raises(ValueError, match="Invalid"):
        bench.crosstalk_focus_ranges(16 * bench.SAMPLE_RATE, center, tight, wide)


def test_crosstalk_focus_three_calls_and_nothing_autopublished():
    class FakeQwen:
        last_rejection = None
        last_detected_language = "ja"
        _prefer_split_on_truncation = True

        def __init__(self):
            self.requests = []
            self.forgot = 0

        def forget(self):
            self.forgot += 1

        def transcribe(self, audio, probe=False):
            self.requests.append(audio.copy())
            return "練習見に来ました" if len(audio) < 7 * bench.SAMPLE_RATE else "練習見に来ました。何々"

    model = FakeQwen()
    audio = np.arange(16 * bench.SAMPLE_RATE, dtype=np.float32)
    rows = bench.run_model(
        model, "qwen3-asr", {"HG06": (bench.WINDOWS["HG06"][0], audio)},
        qwen_crosstalk_focus=True,
    )
    assert [r["variant"] for r in rows] == [
        "full", "focus_tight", "focus_wide", "focus_tight_repeat", "split", "split",
    ]
    assert np.array_equal(model.requests[1], model.requests[3])
    assert model.forgot == len(rows)
    assert rows[1]["text"] == rows[3]["text"] == "練習見に来ました"
    assert rows[1]["start_s"] >= rows[2]["start_s"]
    assert rows[1]["end_s"] <= rows[2]["end_s"]
    report = bench.english_report(rows, "clip.flac")
    assert "targeted crosstalk context-window experiment" in report
    assert "spans differ" in report
    assert "Tight A/A agree" in report
    assert "yes | -" in report


def test_crosstalk_focus_experiment_guards_wrong_backend_and_unknown_windows(
        tmp_path, capsys):
    clip = tmp_path / "clip.flac"
    config = tmp_path / "config.toml"
    clip.write_bytes(b"fake audio")
    config.write_text("", encoding="utf-8")

    assert bench.main([
        str(clip), "--config", str(config), "--dry-run",
        "--model", "haas", "--window", "HG06", "--qwen-crosstalk-focus",
    ]) == 2
    assert "requires --model qwen3-asr only" in capsys.readouterr().err

    assert bench.main([
        str(clip), "--config", str(config), "--dry-run",
        "--model", "qwen3-asr", "--window", "HG27", "--qwen-crosstalk-focus",
    ]) == 2
    assert "No audited crosstalk anchor" in capsys.readouterr().err

    assert bench.main([
        str(clip), "--config", str(config), "--dry-run",
        "--model", "qwen3-asr", "--window", "HG06", "--qwen-crosstalk-focus",
        "--qwen-vocabulary",
    ]) == 2
    assert "separately" in capsys.readouterr().err


def test_qwen_rejected_recovery_ab_preserves_baseline_and_config():
    """A/B runs identical PCM and restores any user-chosen config value."""
    class FakeQwen:
        last_detected_language = "ja"
        last_rejection = None
        _prefer_split_on_truncation = True

        def __init__(self):
            self.cfg = SimpleNamespace(recover_rejected_speech=True)
            self.calls = []

        def forget(self):
            pass

        def transcribe(self, audio, probe=False):
            assert not probe
            self.calls.append((len(audio), self.cfg.recover_rejected_speech,
                               self._prefer_split_on_truncation))
            # The first 16s request is deliberately rejected; only the
            # experimental arm can salvage the same audio.
            if len(audio) == 16 * bench.SAMPLE_RATE:
                if self.cfg.recover_rejected_speech:
                    self.last_rejection = None
                    return "みこさんが話しています"
                self.last_rejection = "repeated_character"
                return ""
            self.last_rejection = None
            return "会話"

    model = FakeQwen()
    audio = np.zeros(16 * bench.SAMPLE_RATE, np.float32)
    rows = bench.run_model(
        model, "qwen3-asr", {"HG11": (385.0, audio)},
        qwen_quality_recovery=True,
    )
    assert [row["variant"] for row in rows] == [
        "full", "quality_recovery", "split", "split",
    ]
    assert rows[0]["rejection"] == "repeated_character"
    assert rows[0]["text"] == ""
    assert rows[1]["rejection"] == ""
    assert rows[1]["text"] == "みこさんが話しています"
    assert model.calls[:2] == [
        (16 * bench.SAMPLE_RATE, False, True),
        (16 * bench.SAMPLE_RATE, True, True),
    ]
    assert model.cfg.recover_rejected_speech is True
    assert "## Qwen rejected-speech recovery A/B" in bench.english_report(rows, "clip.flac")


def test_qwen_recovery_flag_must_be_separate_from_segmentation(tmp_path, capsys):
    clip, config = tmp_path / "clip.flac", tmp_path / "config.toml"
    clip.write_bytes(b"not-decoded")
    config.write_text("", encoding="utf-8")
    assert bench.main([
        str(clip), "--config", str(config),
        "--qwen-recover-rejected", "--qwen-short-target-s", "10",
        "--dry-run",
    ]) == 2
    assert "separately" in capsys.readouterr().err
