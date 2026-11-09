"""Regression tests for YouTube captions-only preparation and prebuffering."""
from __future__ import annotations

from auto_translate.caption_feed import near_cues, prepare_caption_cues
from auto_translate.caption_only import CaptionOnlyPlayer
from auto_translate.captions import CaptionCue, CaptionIndex
from auto_translate.config import Config
from auto_translate.ingest import Source


class StubTranslator:
    def translate(self, ja: str, history: list[tuple[str, str]]) -> str:
        return "EN:" + ja


def test_caption_roll_forward_only_new_text():
    index = CaptionIndex([
        CaptionCue(0.0, 1.0, "こんにちは"),
        CaptionCue(0.8, 2.0, "こんにちは世界"),
        CaptionCue(3.0, 4.0, "おはよう"),
    ])
    cues = prepare_caption_cues(index)
    assert [cue.text for cue in cues] == ["こんにちは", "世界", "おはよう"]
    assert cues[1].start_s >= 1.0


def test_repeated_speech_after_a_gap_is_not_removed():
    index = CaptionIndex([
        CaptionCue(1.0, 2.0, "待って"),
        CaptionCue(5.0, 6.0, "待って"),
    ])
    assert [cue.text for cue in prepare_caption_cues(index)] == ["待って", "待って"]


def test_unfinished_fragments_can_merge_without_changing_start():
    index = CaptionIndex([
        CaptionCue(0.0, 1.0, "それで"),
        CaptionCue(1.1, 2.0, "帰る"),
    ])
    cues = prepare_caption_cues(index)
    assert len(cues) == 1
    assert cues[0].start_s == 0.0
    assert cues[0].end_s == 2.0
    assert "帰る" in cues[0].text


def test_nonverbal_labels_removed_without_dropping_spoken_content():
    index = CaptionIndex([
        CaptionCue(0.0, 1.0, "[music] こんにちは"),
        CaptionCue(2.0, 3.0, "[laughter]"),
    ])
    assert [cue.text for cue in prepare_caption_cues(index)] == ["こんにちは"]


def test_seek_prefetch_includes_ongoing_cue_and_excludes_old_ones():
    cues = [
        CaptionCue(1.0, 2.0, "one"),
        CaptionCue(5.0, 12.0, "long cue"),
        CaptionCue(20.0, 21.0, "later"),
    ]
    assert list(near_cues(cues, 10.0, 5.0, starts=[1.0, 5.0, 20.0])) == [1]
    assert list(near_cues(cues, 20.0, 5.0, starts=[1.0, 5.0, 20.0])) == [2]


def test_five_second_lead_only_needs_nearby_cues():
    cues = CaptionIndex([
        CaptionCue(0.0, 2.0, "こんにちは"),
        CaptionCue(12.0, 13.0, "元気？"),
    ])
    player = CaptionOnlyPlayer(Source("https://example.test/watch?v=x", False, duration=30),
                               Config(), cues, StubTranslator())
    assert player._lead_ready(0.0) is False
    player._completed.add(0)
    assert player._lead_ready(0.0) is True
    assert player._lead_ready(10.0) is False
    player._completed.add(1)
    assert player._lead_ready(10.0) is True
