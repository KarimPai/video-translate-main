"""Full-track Japanese caption download, JSON3 timing and reconstruction tests."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from auto_translate.caption_feed import prepare_caption_cues
from auto_translate.captions import CaptionCue, CaptionIndex, download_vod_captions
from auto_translate.config import IngestConfig


def test_json3_preserves_event_and_first_segment_timestamps():
    raw = json.dumps({
        "events": [
            {"tStartMs": 1000, "dDurationMs": 1400,
             "segs": [{"utf8": ""}, {"utf8": "おはよう", "tOffsetMs": 150},
                      {"utf8": "ございます", "tOffsetMs": 450}]},
            {"tStartMs": 2600, "dDurationMs": 300, "segs": [{"utf8": "\n"}]},
            {"tStartMs": 3000, "dDurationMs": 1200,
             "segs": [{"utf8": "美味しかっ"}, {"utf8": "た", "tOffsetMs": 500}]},
        ],
    }, ensure_ascii=False)
    index = CaptionIndex.from_json3(raw)
    assert len(index.cues) == 2
    assert index.cues[0].text == "おはようございます"
    assert abs(index.cues[0].start_s - 1.15) < 0.001
    assert abs(index.cues[0].end_s - 2.4) < 0.001
    assert index.cues[1].text == "美味しかった"


def test_vtt_rolling_previous_line_prefers_inline_new_word_onset():
    vtt = """WEBVTT

00:00:00.000 --> 00:00:01.000
今日は

00:00:00.800 --> 00:00:02.000
今日は
<00:00:01.200>ゲームをします
"""
    index = CaptionIndex.from_vtt(vtt)
    assert index.cues[0].text == "今日は"
    assert index.cues[1].text == "ゲームをします"
    assert abs(index.cues[1].start_s - 1.2) < 0.001


def test_full_file_json3_preferred_and_cached_without_redownload():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        json3 = json.dumps({"events": [
            {"tStartMs": 0, "dDurationMs": 1600,
             "segs": [{"utf8": "おはよう"}]},
        ]}, ensure_ascii=False)

        def fake_download(command, **kwargs):
            path_template = command[command.index("--output") + 1]
            path = Path(path_template.replace("%(ext)s", "ja.json3"))
            path.write_text(json3, encoding="utf-8")
            assert command[command.index("--sub-format") + 1] == "json3/vtt/best"
            return Mock(returncode=0, stderr="", stdout="")

        with patch("auto_translate.captions._caption_cache_dir", return_value=root):
            with patch("auto_translate.captions.subprocess.run",
                       side_effect=fake_download) as download:
                first = download_vod_captions(
                    "abc123", "https://www.youtube.com/watch?v=abc123",
                    IngestConfig(),
                )
                assert download.call_count == 1
            assert first is not None
            assert first.cues[0].text == "おはよう"
            assert (root / "abc123.ja.json3").exists()
            with patch("auto_translate.captions.subprocess.run") as retry:
                second = download_vod_captions(
                    "abc123", "https://www.youtube.com/watch?v=abc123",
                    IngestConfig(),
                )
            retry.assert_not_called()
            assert second is not None
            assert second.cues[0] == first.cues[0]


def test_cached_vtt_fallback_when_json3_download_unavailable():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "abc123.ja.vtt").write_text(
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nおはよう\n",
            encoding="utf-8",
        )
        with patch("auto_translate.captions._caption_cache_dir", return_value=root):
            with patch("auto_translate.captions.subprocess.run",
                       return_value=Mock(returncode=1, stderr="unavailable", stdout="")):
                index = download_vod_captions(
                    "abc123", "https://www.youtube.com/watch?v=abc123",
                    IngestConfig(),
                )
    assert index is not None
    assert index.cues[0].text == "おはよう"


def test_repeated_speech_at_boundary_is_not_lost_as_rolling_text():
    index = CaptionIndex([
        CaptionCue(1, 2, "おはよう"),
        CaptionCue(2, 3, "おはよう"),
    ])
    cues = prepare_caption_cues(index)
    assert len(cues) == 2


def test_rolling_caption_expansion_keeps_first_original_onset():
    index = CaptionIndex([
        CaptionCue(0, 1, "こんにちは"),
        CaptionCue(0.8, 2, "こんにちは世界"),
    ])
    cues = prepare_caption_cues(index)
    assert len(cues) == 1
    assert cues[0].text == "こんにちは世界"
    assert cues[0].start_s == 0
    assert cues[0].end_s == 2


def test_japanese_clipped_polarity_and_greeting_are_reconstructed():
    index = CaptionIndex([
        CaptionCue(1.0, 1.5, "あんまりそう美味しかっ"),
        CaptionCue(1.5, 2.0, "た何バえ"),
        CaptionCue(4.0, 4.6, "わかん"),
        CaptionCue(4.6, 5.2, "ない"),
        CaptionCue(8.0, 8.7, "おはようござい"),
        CaptionCue(8.7, 9.0, "ます"),
    ])
    cues = prepare_caption_cues(index)
    assert len(cues) == 3
    assert cues[0].text.startswith("あんまりそう美味しかった")
    assert cues[0].start_s == 1.0
    assert cues[1].text == "わかんない"
    assert cues[2].text == "おはようございます"



def test_hololive_and_burger_king_caption_variants_use_full_names():
    from auto_translate.glossary import Glossary

    glossary = Glossary.load(Path(__file__).resolve().parents[1] / "glossary.toml")
    assert ("ナキリあめ", "Nakiri Ayame") in glossary.find("ナキリあめかな")
    assert ("バーグキング", "Burger King") in glossary.find("バーグキング食べた")
    assert ("バーガーキング", "Burger King") in glossary.find("バーガーキング")



def test_vtt_inline_timestamp_inside_word_sequence_preserves_early_speech():
    vtt = """WEBVTT

00:00:01.000 --> 00:00:02.000
おはよう

00:00:01.900 --> 00:00:04.000
おはよう
一緒に<00:00:03.000>遊びましょう
"""
    index = CaptionIndex.from_vtt(vtt)
    assert len(index.cues) == 2
    assert index.cues[1].text == "一緒に遊びましょう"
    assert abs(index.cues[1].start_s - 1.9) < 0.001
