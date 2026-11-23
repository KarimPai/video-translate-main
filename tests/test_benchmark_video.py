"""CPU-only checks for the local benchmark harness."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from benchmark_video import main, parse_log, seconds, summarize


FAKE_LOG = (
    "[25:01.1–25:03.5] ここにいる。    "
    "(vad 0.40s · wait 0.03s · asr 0.12s · mt 0.06s)\n"
    "                  → I'm here.\n"
    "[1:02:05.0–1:02:07.5] そうだね。    "
    "(vad 0.50s · wait 0.10s · asr 0.22s · mt 0.08s)\n"
    "                         → That's right.\n"
)


def test_parse_timestamp_formats_and_english():
    assert seconds("25:01.1") == 1501.1
    assert seconds("1:02:05.0") == 3725.0
    cues = parse_log(FAKE_LOG)
    assert len(cues) == 2
    assert cues[0]["ja"] == "ここにいる。"
    assert cues[0]["en"] == "I'm here."
    assert cues[1]["en"] == "That's right."
    assert cues[1]["asr_s"] == 0.22
    assert cues[1]["start_s"] == 3725.0


def test_summary_is_operational_not_quality():
    data = summarize(parse_log(FAKE_LOG), 20.0, 0)
    assert data["cue_count"] == 2
    assert data["translated_count"] == 2
    assert data["mean_asr_s"] == 0.17
    assert data["p95_asr_s"] == 0.22
    assert data["long_cues_ge_14s"] == 0
    assert data["long_cues_ge_10s"] == 0
    assert data["median_cue_duration_s"] == 2.45
    assert "accuracy" not in data
    assert summarize([], 5, 1)["mean_asr_s"] is None


def test_missing_clip_does_not_launch_models(tmp_path: Path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text("[models]\n", encoding="utf-8")

    def disallow(*args, **kwargs):
        raise AssertionError("must not launch subprocess")

    monkeypatch.setattr("benchmark_video.subprocess.run", disallow)
    assert main([str(tmp_path / "missing.webm"), "--config", str(config)]) == 2


def test_runs_sequentially_and_saves_distinct_results(tmp_path: Path, monkeypatch):
    clip = tmp_path / "clip.webm"
    clip.write_bytes(b"fake-data")
    config = tmp_path / "config.toml"
    config.write_text("[models]\n", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        kwargs["stdout"].write(FAKE_LOG)
        kwargs["stdout"].flush()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("benchmark_video.subprocess.run", fake_run)
    code = main([
        str(clip), "--config", str(config), "--outdir", str(tmp_path / "runs"),
        "--asr", "haas", "whisper", "--mt", "gemma4",
        "--context-lines", "8", "--vad-max-utterance-s", "10", "--vad-threshold", "0.28",
    ])
    assert code == 0
    assert len(calls) == 2
    assert "--fast" in calls[0]
    assert "--no-web" in calls[0]
    assert "--context-lines" in calls[0]
    assert "--vad-max-utterance-s" in calls[0]
    assert "--vad-threshold" in calls[0]
    assert calls[0][calls[0].index("--vad-threshold") + 1] == "0.28"
    assert calls[0].index("--asr-backend") != calls[0].index("--translate-backend")
    results = list((tmp_path / "runs").glob("*/summary.json"))
    assert len(results) == 1
    data = json.loads(results[0].read_text(encoding="utf-8"))
    assert len(data) == 2
    assert all(entry["cue_count"] == 2 for entry in data)
    assert len(list(results[0].parent.glob("*.jsonl"))) == 2
    assert len(list(results[0].parent.glob("*-review.csv"))) == 2


def test_failed_run_stops_before_next_model(tmp_path: Path, monkeypatch):
    clip = tmp_path / "clip.webm"
    clip.write_bytes(b"fake-data")
    config = tmp_path / "config.toml"
    config.write_text("[models]\n", encoding="utf-8")
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        kwargs["stdout"].write("model unavailable\n")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("benchmark_video.subprocess.run", fail)
    assert main([
        str(clip), "--config", str(config),
        "--outdir", str(tmp_path / "out"), "--asr", "haas", "whisper",
    ]) == 1
    assert len(calls) == 1
