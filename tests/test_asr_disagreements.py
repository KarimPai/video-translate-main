"""CPU-only regression tests for ASR disagreement triage."""
from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_asr_disagreements.py"
spec = importlib.util.spec_from_file_location("analyze_asr_disagreements", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)


def sample(**overrides):
    row = {
        "id": "HG01", "start_s": "0.26", "end_s": "15.49", "focus": "",
        "haas_ja": "ラミィありがとう", "qwen_ja": "ラミィ、ありがとう",
        "whisper_ja": "ラミィありがとう",
        "haas_boundary": "", "qwen_boundary": "", "whisper_boundary": "",
    }
    return row | overrides


def test_punctuation_does_not_cause_disagreement():
    assert mod.normalize("ラミィ、ありがとう！") == mod.normalize("ラミィありがとう")
    assert mod.difference(mod.normalize("ラミィ、ありがとう"),
                          mod.normalize("ラミィありがとう")) == 0


def test_complete_agreement_does_not_claim_truth():
    result = mod.analyze_row(sample())
    assert result["disagreement"] == 0
    assert result["priority"] == "low"
    assert result["present_models"] == "haas;qwen;whisper"
    assert "verified" not in result["signals"]


def test_one_model_empty_is_flagged_as_possible_loss_only():
    result = mod.analyze_row(sample(whisper_ja=""))
    assert "one_or_more_models_empty" in result["signals"]
    assert result["missing_models"] == "whisper"
    assert result["priority"] == "high"


def test_all_empty_cannot_prove_silence():
    result = mod.analyze_row(sample(haas_ja="", qwen_ja="", whisper_ja=""))
    assert "speech_presence_unverified" in result["signals"]
    assert result["priority"] == "low"


def test_unavailable_source_is_not_marked_missing_speech():
    result = mod.analyze_row(sample(qwen_boundary="source_unavailable", qwen_ja=""))
    assert "qwen" not in result["missing_models"]
    assert "incomplete_model_data" in result["signals"]


def test_cue_boundary_is_an_uncertainty_flag():
    result = mod.analyze_row(sample(whisper_boundary="partial_cue"))
    assert result["boundary_models"] == "whisper"
    assert "cue_boundary_mismatch" in result["signals"]


def test_english_report_does_not_require_japanese_reading():
    rows = mod.analyze([
        sample(whisper_ja=""),
        sample(id="HG02", haas_ja="", qwen_ja="", whisper_ja=""),
    ])
    report = mod.report(rows, mod.summary(rows))
    assert "NOT an accuracy ranking" in report
    assert "HG01" in report
    assert "01:25:00" in report
    assert "Unverified speech-presence" in report
    assert "ラミィ" not in report


def test_original_stream_offset_and_relative_difference():
    result = mod.analyze_row(sample(
        start_s="600",
        haas_ja="ああああああああ", qwen_ja="いいいいいいいい", whisper_ja="うううううううう",
    ))
    assert result["original_stream_time"] == "01:35:00"
    assert result["disagreement"] == 1.0
    assert result["priority"] == "medium"  # Disagreement alone is not proven error
