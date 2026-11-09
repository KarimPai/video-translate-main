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
    assert ("ペコーラ", "Pekora") in glossary.find("ペコーラをつけた")



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



def test_strong_clipped_endings_join_despite_overlapping_json3_display_times():
    # Real rolling captions overlap: the previous caption's display end
    # is several seconds AFTER the next spoken piece begins.
    index = CaptionIndex([
        CaptionCue(1.0, 5.5, "わかん"),
        CaptionCue(2.0, 6.0, "ない30秒待つと言ったな嘘"),
        CaptionCue(10.0, 14.5, "これで行こうと思い"),
        CaptionCue(11.0, 15.5, "ます話すけど"),
        CaptionCue(21.0, 25.0, "おはようござい"),
        CaptionCue(22.0, 26.0, "ますおはようございます"),
    ], source_format="json3")
    cues = prepare_caption_cues(index)
    assert len(cues) == 3
    assert cues[0].text == "わかんない30秒待つと言ったな嘘"
    assert cues[1].text == "これで行こうと思います話すけど"
    assert cues[2].text == "おはようございますおはようございます"
    assert [cue.start_s for cue in cues] == [1.0, 10.0, 21.0]


def test_strong_word_completions_do_not_cross_long_silences_or_scene_changes():
    index = CaptionIndex([
        CaptionCue(1.0, 2.0, "わかん"),
        CaptionCue(8.0, 8.5, "ない"),
        CaptionCue(11.0, 12.0, "思い"),
        CaptionCue(17.0, 17.3, "ます"),
    ])
    cues = prepare_caption_cues(index)
    assert len(cues) == 4


def test_format_provenance_on_parsed_tracks():
    assert CaptionIndex.from_vtt(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nおはよう\n"
    ).source_format == "vtt"
    assert CaptionIndex.from_json3(json.dumps({
        "events": [{"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "こんにちは"}]}]
    })).source_format == "json3"



def test_realistic_json3_long_event_boundaries_join_clipped_japanese():
    # The reported VOD had 588 raw cues over ~36 minutes: an adjacent event
    # can begin >4 seconds later without being a different spoken sentence.
    cues = CaptionIndex([
        CaptionCue(10.0, 15.0, "わかん"),
        CaptionCue(14.8, 19.4, "ない30秒待つと言ったな嘘"),
        CaptionCue(25.0, 30.0, "これで行こうと思い"),
        CaptionCue(30.2, 35.0, "ます話すけど"),
    ], source_format="json3")
    merged = prepare_caption_cues(cues)
    assert [cue.text for cue in merged] == [
        "わかんない30秒待つと言ったな嘘",
        "これで行こうと思います話すけど",
    ]
    assert [cue.start_s for cue in merged] == [10.0, 25.0]


def test_no_join_across_very_long_json3_display_or_speech_gap():
    cues = CaptionIndex([
        CaptionCue(1.0, 4.0, "わかん"),
        CaptionCue(7.0, 12.0, "ない"),
        CaptionCue(20.0, 26.0, "思い"),
        CaptionCue(22.0, 38.0, "ます"),
    ], source_format="json3")
    merged = prepare_caption_cues(cues)
    # Long silence or a 18-second span is not a reliable single utterance.
    assert [cue.text for cue in merged] == ["わかん", "ない", "思い", "ます"]



def test_gta_pekkora_stream_grammatical_joins_only():
    # Use the exact clipped patterns from the long GTA/Hololive transcript.
    # Bound the inputs to a plausible media window rather than joining all
    # adjacent conversational cues across different speakers.
    index = CaptionIndex([
        CaptionCue(10.0, 13.0, "俺少女のペコーラが貸し"),
        CaptionCue(13.1, 17.0, "てあげるからやなくなれるあんたやめ"),
        CaptionCue(17.1, 20.0, "なさいお姉ちゃんが許しませんよそんなの"),
        CaptionCue(30.0, 33.0, "何の話してんだ"),
        CaptionCue(33.1, 36.0, "俺にも貸してくれよ"),
    ], source_format="json3")
    translated = prepare_caption_cues(index)
    assert [c.text for c in translated] == [
        "俺少女のペコーラが貸してあげるからやなくなれるあんたやめなさいお姉ちゃんが許しませんよそんなの",
        "何の話してんだ",
        "俺にも貸してくれよ",
    ]
    assert translated[0].start_s == 10.0
    assert translated[0].end_s == 20.0


def test_distant_clipped_words_are_not_joined():
    index = CaptionIndex([
        CaptionCue(2.0, 4.0, "やめ"),
        CaptionCue(10.0, 12.0, "なさい"),
        CaptionCue(20.0, 22.0, "貸し"),
        CaptionCue(30.0, 32.0, "てあげる"),
    ])
    assert [cue.text for cue in prepare_caption_cues(index)] == [
        "やめ", "なさい", "貸し", "てあげる",
    ]
