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
    assert [cue.text for cue in cues] == ["こんにちは世界", "おはよう"]
    assert cues[0].start_s == 0.0  # never shift expanded text later
    assert cues[0].end_s == 2.0


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



def test_initial_buffer_waits_for_all_cues_in_first_twelve_seconds():
    cues = CaptionIndex([
        CaptionCue(0.0, 1.0, "one"),
        CaptionCue(4.0, 5.0, "two"),
        CaptionCue(9.0, 10.0, "three"),
        CaptionCue(14.0, 15.0, "later"),
    ])
    player = CaptionOnlyPlayer(
        Source("https://example.test/watch?v=x", False, duration=30),
        Config(), cues, StubTranslator(),
    )
    player._completed.update({0, 1})
    assert player._lead_ready(0.0) is False
    player._completed.add(2)
    assert player._lead_ready(0.0) is True
    assert player._lead_ready(0.0, 3.0) is True


def test_first_caption_renders_at_original_vod_timestamp():
    from auto_translate.types import Subtitle

    class RecordingIpc:
        def __init__(self):
            self.commands = []

        def command(self, *args, **kwargs):
            self.commands.append((args, kwargs))

    player = CaptionOnlyPlayer(
        Source("https://example.test/watch?v=x", False, duration=30),
        Config(), CaptionIndex([CaptionCue(0.0, 1.5, "こんにちは")]),
        StubTranslator(),
    )
    player._ipc = RecordingIpc()
    player.cache.add(Subtitle(0.0, 1.5, "こんにちは", "Hello!"))
    player._render(0.0)
    assert player._ipc.commands
    assert player._ipc.commands[-1][1]["format"] == "ass-events"
    assert "Hello!" in player._ipc.commands[-1][1]["data"]
    # Existing subtitle remains visible during the configured linger.
    player._render(2.0)
    assert player.cache.active_at(2.0, include_linger=True)


def test_previous_history_uses_only_contiguous_nearby_translations():
    from auto_translate.types import Subtitle
    cues = CaptionIndex([
        CaptionCue(0.0, 1.0, "old scene"),
        CaptionCue(40.0, 41.0, "new scene one"),
        CaptionCue(41.5, 42.0, "new scene two"),
        CaptionCue(42.5, 43.0, "current"),
    ])
    player = CaptionOnlyPlayer(
        Source("https://example.test/watch?v=x", False, duration=60),
        Config(), cues, StubTranslator(),
    )
    player._translated[0] = Subtitle(0.0, 1.0, "old scene", "Old scene")
    player._translated[1] = Subtitle(40.0, 41.0, "new scene one", "Scene one")
    player._translated[2] = Subtitle(41.5, 42.0, "new scene two", "Scene two")
    assert player._previous_history(3) == [
        ("new scene one", "Scene one"), ("new scene two", "Scene two")
    ]
    del player._translated[2]
    # An unprocessed immediately preceding caption must not cause a jump
    # back to an older but completed cue after seeking.
    assert player._previous_history(3) == []



def test_optional_caption_advance_changes_display_only_not_source_times():
    from auto_translate.types import Subtitle

    class RecordingIpc:
        def __init__(self):
            self.commands = []

        def command(self, *args, **kwargs):
            self.commands.append((args, kwargs))

    config = Config()
    config.mpv.caption_advance_s = 0.5
    player = CaptionOnlyPlayer(
        Source("https://example.test/watch?v=x", False, duration=10),
        config, CaptionIndex([CaptionCue(1.0, 2.0, "こんにちは")]),
        StubTranslator(),
    )
    player._ipc = RecordingIpc()
    player.cache.add(Subtitle(1.0, 2.0, "こんにちは", "Hello"))
    player._completed.add(0)
    # Audio is at 0.7s: an optional 0.5s adjustment uses cue clock 1.2.
    player._render(0.7)
    assert player._ipc.commands
    assert "Hello" in player._ipc.commands[-1][1]["data"]
    assert player.cache.active_at(0.7) == []
    assert player._lead_ready(0.7, 1.0) is True
