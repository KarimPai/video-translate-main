"""CPU-only tests for exact-window Qwen3-ASR vs Japanese fine-tune comparison."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_qwen_ja as cmp


def record(label: str, text: str, *, language: str = "ja",
           rejection: str = "", window: str = "HG02") -> dict:
    return {
        "window": window, "model": label, "text": text,
        "language": language, "elapsed_s": 0.2, "rejection": rejection,
        "ja_chars": len(text),
    }


def test_reference_windows_validate_fixed_boundaries():
    windows = cmp.load_reference_windows()
    assert len(windows) == 30
    assert windows["HG02"]["start_s"] == pytest.approx(15.90)
    assert windows["HG02"]["end_s"] == pytest.approx(19.52)
    assert "money" in windows["HG12"]["focus"]
    assert "HG27" in cmp.DEFAULT_WINDOWS
    assert all(x in windows for x in cmp.DEFAULT_WINDOWS)


def test_reference_rejects_duplicate_and_invalid_range(tmp_path):
    f = tmp_path / "windows.csv"
    f.write_text("id,start_s,end_s,focus\nHG02,10,5,name\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid"):
        cmp.load_reference_windows(f)
    f.write_text(
        "id,start_s,end_s,focus\nHG02,1,5,name\nHG02,5,10,name\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        cmp.load_reference_windows(f)


def test_agreement_preserves_japanese_numbers_and_english_names():
    assert cmp.normalize_for_agreement("30万円、Miko！ ") == "30万円Miko"
    assert cmp.agreement("ありがとう。", "ありがとう") == 1.0
    assert cmp.agreement("30万円", "300万円") < 1.0
    assert cmp.agreement("", "") is None


def test_high_disagreement_is_not_called_accuracy():
    windows = {"HG02": {"start_s": 15.9, "end_s": 19.52,
                        "focus": "proper-name;Miko"}}
    rows = [record("current", "美子さん"), record("qwen-ja", "さくらみこ")]
    review = cmp.build_review(rows, windows)
    assert review[0]["current_ja"] == "美子さん"
    assert review[0]["qwen_ja_ja"] == "さくらみこ"
    assert review[0]["needs_review"] in ("high_disagreement", "review_names_or_meaning")
    report = cmp.report_markdown(review, "clip.flac")
    assert "Not an accuracy ranking" in report
    assert "Agreement is not CER" in report
    assert "HG02" in report


def test_missing_output_not_mistaken_for_silence():
    windows = {"HG27": {"start_s": 1076.8, "end_s": 1083.5,
                        "focus": "speech-presence-check;unverified"}}
    review = cmp.build_review([
        record("current", "", window="HG27"),
        record("qwen-ja", "", window="HG27"),
    ], windows)
    assert review[0]["needs_review"] == "both_empty_unverified"


def test_model_exception_is_visible():
    windows = {"HG11": {"start_s": 385.82, "end_s": 401.38,
                        "focus": "dialogue"}}
    review = cmp.build_review([
        record("current", "", language="error", rejection="timeout", window="HG11"),
        record("qwen-ja", "ありがとう", window="HG11"),
    ], windows)
    assert review[0]["needs_review"] == "model_error"
    assert review[0]["current_rejection"] == "timeout"


def test_results_save_comparison_provenance(tmp_path):
    media = tmp_path / "sample.flac"
    media.write_bytes(b"fake")
    windows = {"HG02": {"start_s": 15.90, "end_s": 19.52, "focus": "Miko"}}
    rows = [record("current", "みこさん"), record("qwen-ja", "美子さん")]
    paths = {
        "current": {"model_path": "standard.gguf", "mmproj_path": "standard_mm.gguf"},
        "qwen-ja": {"model_path": "ja.gguf", "mmproj_path": "ja_mm.gguf"},
    }
    cmp.write_results(tmp_path, rows, windows, media, paths)
    with (tmp_path / "review.csv").open(encoding="utf-8-sig", newline="") as file:
        data = list(csv.DictReader(file))
    assert len(data) == 1
    assert data[0]["current_ja"] == "みこさん"
    assert data[0]["qwen_ja_ja"] == "美子さん"
    provenance = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert provenance["clip_bytes"] == 4
    assert provenance["model_paths"]["qwen-ja"]["model_path"] == "ja.gguf"
    assert len((tmp_path / "hypotheses.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_dry_run_needs_no_models_or_cuda(tmp_path, capsys):
    media = tmp_path / "sample.flac"
    config = tmp_path / "config.toml"
    media.write_bytes(b"fake")
    config.write_text("[models]\n", encoding="utf-8")
    assert cmp.main([str(media), "--config", str(config),
                     "--ja-model-dir", str(tmp_path / "missing"),
                     "--window", "HG02", "--dry-run"]) == 0
    assert "HG02" in capsys.readouterr().out


def test_missing_ja_models_reports_download_without_loading(tmp_path, capsys):
    media = tmp_path / "sample.flac"
    config = tmp_path / "config.toml"
    media.write_bytes(b"fake")
    config.write_text("[models]\n", encoding="utf-8")
    assert cmp.main([str(media), "--config", str(config),
                     "--ja-model-dir", str(tmp_path / "missing"),
                     "--window", "HG02"]) == 2
    assert "Download BOTH" in capsys.readouterr().err


def test_unknown_window_fails_before_loading_models(tmp_path):
    media = tmp_path / "sample.flac"
    config = tmp_path / "config.toml"
    media.write_bytes(b"fake")
    config.write_text("[models]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown windows"):
        cmp.main([str(media), "--config", str(config),
                  "--window", "NOPE", "--dry-run"])



def test_explicit_paths_prevent_ambiguous_registry_discovery(tmp_path):
    """Installing two Qwen models must never trigger implicit model selection."""
    from auto_translate.config import Config

    config = Config()
    standard = tmp_path / "standard"
    japanese = tmp_path / "japanese"
    selected = cmp.explicit_model_configs(config, standard, japanese)
    assert set(selected) == {"current", "qwen-ja"}
    assert selected["current"]["model_path"] == str(standard / cmp.STANDARD_MODEL)
    assert selected["current"]["mmproj_path"] == str(standard / cmp.STANDARD_MMPROJ)
    assert selected["qwen-ja"]["model_path"] == str(japanese / cmp.JA_MODEL)
    assert selected["qwen-ja"]["mmproj_path"] == str(japanese / cmp.JA_MMPROJ)
    assert selected["current"]["model_path"] != selected["qwen-ja"]["model_path"]
    assert selected["current"]["response_timeout_s"] == selected["qwen-ja"]["response_timeout_s"]


def test_both_models_are_selected_explicitly_and_closed_sequentially(
        tmp_path, monkeypatch):
    """CPU-only integration: compare runner does not use registry auto-discovery."""
    import numpy as np

    import auto_translate.config as cfg_module
    import auto_translate.registry as reg_module

    clip = tmp_path / "test.flac"
    config = tmp_path / "config.toml"
    clip.write_bytes(b"fake")
    config.write_text("[models]\n", encoding="utf-8")
    standard = tmp_path / "standard"
    japanese = tmp_path / "japanese"
    for parent, model, mmproj in (
        (standard, cmp.STANDARD_MODEL, cmp.STANDARD_MMPROJ),
        (japanese, cmp.JA_MODEL, cmp.JA_MMPROJ),
    ):
        parent.mkdir()
        (parent / model).write_bytes(b"fake")
        (parent / mmproj).write_bytes(b"fake")

    events = []

    class FakeASR:
        last_detected_language = "ja"
        last_rejection = None

        def __init__(self, identity):
            self.identity = identity

        def forget(self):
            pass

        def transcribe(self, audio, probe=False):
            assert not probe
            assert audio.size == 16000
            events.append(("decode", self.identity))
            return "みこさん"

        def close(self):
            events.append(("close", self.identity))

    class FakeRegistry:
        def create_asr(self, name, model_config):
            assert name == "qwen3-asr"
            model = Path(model_config["model_path"])
            mmproj = Path(model_config["mmproj_path"])
            assert model.exists() and mmproj.exists()
            assert model.parent == mmproj.parent
            events.append(("open", model.name))
            return FakeASR(model.name)

    monkeypatch.setattr(cfg_module, "load_config", lambda _: cfg_module.Config())
    monkeypatch.setattr(reg_module, "get_default_registry", lambda: FakeRegistry())
    monkeypatch.setattr(cmp.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(
        cmp, "decode_window",
        lambda media, start, end, ffmpeg: np.zeros(16000, dtype=np.float32),
    )
    status = cmp.main([
        str(clip), "--config", str(config),
        "--standard-model-dir", str(standard), "--ja-model-dir", str(japanese),
        "--outdir", str(tmp_path / "runs"), "--window", "HG02",
    ])
    assert status == 0
    assert events == [
        ("open", cmp.STANDARD_MODEL),
        ("decode", cmp.STANDARD_MODEL),
        ("close", cmp.STANDARD_MODEL),
        ("open", cmp.JA_MODEL),
        ("decode", cmp.JA_MODEL),
        ("close", cmp.JA_MODEL),
    ]
    output_dirs = list((tmp_path / "runs").iterdir())
    assert len(output_dirs) == 1
    report = (output_dirs[0] / "review.csv").read_text(encoding="utf-8-sig")
    assert "みこさん" in report
