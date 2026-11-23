"""CPU-only tests for manual-ground-truth ASR evaluation."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_asr_reference.py"
if not SCRIPT.exists():
    SCRIPT = Path(__file__).resolve().parent / "prepare_asr_reference.py"
spec = importlib.util.spec_from_file_location("prepare_asr_reference", SCRIPT)
ref = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(ref)


def test_normalize_ignores_punctuation_but_preserves_words():
    assert ref.normalize_ja("ラミィ、ありがとう！ ") == "ラミィありがとう"
    assert ref.normalize_ja("ＡＢＣ１２ こんにちは") == "ABC12こんにちは"


def test_edit_distance():
    assert ref.edit_distance("ラミィ", "ラミィ") == 0
    assert ref.edit_distance("ラミィ", "ラミー") == 1
    assert ref.edit_distance("こんにちは", "") == 5


def test_score_only_verified_and_detects_empty_speech():
    rows = [
        {"id": "HG01", "verification_status": "verified", "speech_present": "yes",
         "reference_ja": "ラミィ、ありがとう。", "haas_ja": "ラミィありがとう",
         "qwen_ja": "ラミーありがとう", "whisper_ja": ""},
        {"id": "HG02", "verification_status": "pending", "speech_present": "yes",
         "reference_ja": "ここはまだ未確認", "haas_ja": "", "qwen_ja": "", "whisper_ja": ""},
    ]
    scores, details = ref.score_rows(rows)
    assert len(details) == 3
    assert [s["verified_windows"] for s in scores] == [1, 1, 1]
    assert [s["edit_errors"] for s in scores] == [0, 1, 8]
    assert scores[2]["empty_speech_windows"] == 1
    assert scores[1]["cer"] == 0.125


def test_silence_false_positive_does_not_change_speech_cer():
    rows = [{"id": "HG03", "verification_status": "verified", "speech_present": "no",
             "reference_ja": "", "haas_ja": "誤認識", "qwen_ja": "", "whisper_ja": ""}]
    scores, _ = ref.score_rows(rows)
    assert scores[0]["false_speech_window_rate"] == 1.0
    assert scores[1]["false_speech_window_rate"] == 0.0
    assert scores[0]["cer"] is None


def test_verified_speech_requires_reference():
    with pytest.raises(ValueError, match="reference_ja"):
        ref.score_rows([{"id": "HG01", "verification_status": "verified",
                         "speech_present": "yes", "reference_ja": ""}])


def test_verified_no_speech_rejects_nonempty_reference():
    with pytest.raises(ValueError, match="reference_ja"):
        ref.score_rows([{"id": "HG01", "verification_status": "verified",
                         "speech_present": "no", "reference_ja": "本当はある"}])


def test_midpoint_alignment_flags_partial_cue():
    cues = [{"start_s": 1, "end_s": 9, "ja": "こんにちは"},
            {"start_s": 10, "end_s": 12, "ja": "ありがとう"}]
    aligned, warning = ref.aligned_text(cues, 5, 12)
    assert aligned == "こんにちは ありがとう"
    assert warning == "partial_cue"


def test_full_cue_alignment_without_warning():
    cues = [{"start_s": 1, "end_s": 5, "ja": "ラミィ"}]
    assert ref.aligned_text(cues, 1, 5) == ("ラミィ", "")
