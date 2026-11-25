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
    assert ref.edit_distance("", "こんにちは") == 5
    assert ref.edit_distance("みこ", "みこち") == ref.edit_distance("みこち", "みこ") == 1


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


def test_find_matching_haas_run_with_non_gemma_translator(tmp_path):
    import json

    folder = tmp_path / "20261123-162100"
    folder.mkdir()
    (folder / "manifest.json").write_text(json.dumps({
        "clip_bytes": 19515475, "vad_max_utterance_s": 14.0,
        "vad_threshold": 0.28,
        "combinations": [{"asr": "haas", "mt": "hymt"}],
    }), encoding="utf-8")
    expected = folder / "01-haas-hymt.jsonl"
    expected.write_text('{"start_s":0,"end_s":1,"ja":"はい"}\n', encoding="utf-8")
    assert ref.latest_matched_run(tmp_path, "haas", 19515475) == expected
    assert ref.latest_matched_run(tmp_path, "haas", 19515476) is None
    assert ref.latest_matched_run(tmp_path, "qwen", 19515475) is None


def test_missing_optional_hypotheses_are_not_scored_as_asr_errors(tmp_path):
    from argparse import Namespace

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"dummy clip data")
    manifest = Path(__file__).resolve().parents[1] / "benchmarks/reference/hologta_30.csv"
    haas = tmp_path / "01-haas-plamo8b.jsonl"
    haas.write_text('{"start_s":0.26,"end_s":15.49,"ja":"こんにちは"}\n', encoding="utf-8")
    args = Namespace(
        clip=clip, manifest=manifest, runs_root=tmp_path / "runs",
        haas=haas, qwen=None, whisper=None,
        outdir=tmp_path / "working", extract_audio=False, overwrite=False,
    )
    assert ref.prepare(args) == 0
    rows = ref.read_csv(args.outdir / "hologta_30_review.csv")
    assert len(rows) == 30
    assert rows[0]["haas_ja"] == "こんにちは"
    assert rows[0]["qwen_boundary"] == "source_unavailable"
    assert rows[0]["whisper_boundary"] == "source_unavailable"
    assert rows[0]["verification_status"] == "pending"

    rows[0].update(verification_status="verified",
                   speech_present="yes", reference_ja="こんにちは")
    scores, details = ref.score_rows(rows)
    assert scores[0]["cer"] == 0
    assert scores[0]["verified_windows"] == 1
    assert scores[1]["verified_windows"] == 0
    assert scores[1]["cer"] is None
    assert scores[2]["verified_windows"] == 0
    assert [entry["model"] for entry in details] == ["haas"]
