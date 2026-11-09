"""Regression coverage for the RAM-only, current-VOD subtitle cache."""
import queue
import threading
import unittest
from types import SimpleNamespace

import numpy as np

from auto_translate.output.mpv import MpvOutput
from auto_translate.pipeline import run_granite
from auto_translate.session_cache import SessionSubtitleCache
from auto_translate.types import Subtitle, Utterance


class SessionCacheTests(unittest.TestCase):
    def setUp(self):
        self.cache = SessionSubtitleCache()
        self.line = Subtitle(
            10.0, 12.0, "", "Watame was wrong!",
            generation=0, speech_start_s=10.1,
        )

    def test_active_cues_survive_seek_and_deduplicate(self):
        self.assertTrue(self.cache.add(self.line))
        self.assertFalse(self.cache.add(self.line))
        self.assertEqual(len(self.cache), 1)
        self.assertEqual(self.cache.active_at(10.5), [self.line])
        self.assertEqual(self.cache.active_at(9.99), [])
        self.assertEqual(self.cache.active_at(12.0), [])
        self.assertIs(self.cache.match(10.08, 11.93), self.line)
        self.assertIsNone(self.cache.match(10.5, 11.93))

    def test_processed_ranges_merge_without_claiming_gaps(self):
        self.cache.record_processed(100.0, 103.0)
        self.cache.record_processed(103.0, 107.0)
        self.cache.record_processed(110.0, 120.0)
        self.assertEqual(self.cache.covered_until(101.0), 107.0)
        self.assertEqual(self.cache.covered_until(109.0), 109.0)
        self.cache.record_processed(106.9, 110.0)
        self.assertEqual(self.cache.covered_until(101.0), 120.0)
        self.assertEqual(self.cache.covered_until(200.0), 200.0)

    def test_new_video_has_separate_empty_cache(self):
        self.cache.add(self.line)
        other_video = SessionSubtitleCache()
        self.assertEqual(len(other_video), 0)
        self.assertEqual(other_video.active_at(11.0), [])

    def test_revisited_utterance_skips_granite_speech_decode(self):
        self.cache.add(self.line)

        class MustNotDecode:
            def translate_audio(self, audio):
                raise AssertionError("Granite should not decode a cached VOD cue")

        utterances, subtitles = queue.Queue(), queue.Queue()
        utterances.put(Utterance(
            start_s=10.0, end_s=12.0,
            audio=np.zeros(160, dtype=np.float32),
            generation=3, speech_start_s=10.1,
        ))
        utterances.put(None)
        run_granite(
            MustNotDecode(), utterances, subtitles,
            get_generation=lambda: 3, session_cache=self.cache,
        )
        reused = subtitles.get_nowait()
        self.assertIsInstance(reused, Subtitle)
        self.assertEqual(reused.generation, 3)
        self.assertEqual(reused.ja, "")
        self.assertEqual(reused.en, "Watame was wrong!")
        self.assertIsNone(subtitles.get_nowait())

    def test_mpv_uses_cache_to_display_cues_even_without_current_lines(self):
        self.cache.add(self.line)
        out = object.__new__(MpvOutput)
        out.ingest = SimpleNamespace(source=SimpleNamespace(live=False))
        out._live_overlay_confirmed = False
        out._subtitle_cache = self.cache
        out._lock = threading.Lock()
        out._lines = []
        out._shown = ""
        out._logged_displayed_subtitles = set()
        out._lags = []
        out.cfg = SimpleNamespace(max_subtitle_rows=3, en_size=28, ja_size=0)
        calls = []
        out.ipc = SimpleNamespace(command=lambda *args, **named: calls.append((args, named)))
        out._render(10.5)
        self.assertIn("Watame was wrong!", out._shown)
        self.assertTrue(calls)
        self.assertEqual(calls[-1][1]["format"], "ass-events")

    def test_interrupted_speakers_remain_two_independent_granite_cues(self):
        class FakeGranite:
            def __init__(self):
                self.calls = 0
            def translate_audio(self, audio):
                self.calls += 1
                return ["I'm heading out—", "Wait a second!"][self.calls - 1]

        utterances, subtitles = queue.Queue(), queue.Queue()
        utterances.put(Utterance(
            0.0, 1.5, np.full(24000, .1, dtype=np.float32),
            speech_start_s=0.0, speaker_handoff=False,
        ))
        utterances.put(Utterance(
            1.5, 4.0, np.full(40000, -.1, dtype=np.float32),
            speech_start_s=1.5, speaker_handoff=True,
        ))
        utterances.put(None)
        granite = FakeGranite()
        run_granite(granite, utterances, subtitles)
        first, second, sentinel = (
            subtitles.get_nowait(), subtitles.get_nowait(), subtitles.get_nowait()
        )
        self.assertIsNone(sentinel)
        self.assertEqual(granite.calls, 2)
        self.assertEqual([first.en, second.en],
                         ["I'm heading out—", "Wait a second!"])
        self.assertEqual([(first.start_s, first.end_s), (second.start_s, second.end_s)],
                         [(0.0, 1.5), (1.5, 4.0)])
        self.assertEqual([first.speaker_handoff, second.speaker_handoff],
                         [False, True])

    def test_vod_cache_replay_stops_old_subtitle_at_speaker_handoff(self):
        first = Subtitle(10.0, 11.5, "先に", "I'll go first.",
                         speech_start_s=10.0)
        second = Subtitle(11.5, 13.0, "", "Wait!",
                          speech_start_s=11.5, speaker_handoff=True)
        self.cache.add(first)
        self.cache.add(second)
        self.assertEqual(len(self.cache.active_at(11.8, include_linger=True)), 2)

        out = object.__new__(MpvOutput)
        out.ingest = SimpleNamespace(source=SimpleNamespace(live=False))
        out._live_overlay_confirmed = False
        out._subtitle_cache = self.cache
        out._lock = threading.Lock()
        out._lines = []
        out._shown = ""
        out._logged_displayed_subtitles = set()
        out._lags = []
        out.cfg = SimpleNamespace(max_subtitle_rows=3, en_size=28, ja_size=0)
        out.ipc = SimpleNamespace(command=lambda *args, **named: None)
        out._render(11.8)
        self.assertIn("Wait!", out._shown)
        self.assertNotIn("I'll go first.", out._shown)

    def test_cached_handoff_metadata_survives_seek_generation(self):
        cached = Subtitle(20.0, 21.5, "", "Wait!",
                          speech_start_s=20.0, speaker_handoff=True)
        self.cache.add(cached)
        source, translated = queue.Queue(), queue.Queue()
        source.put(Utterance(
            20.0, 21.5, np.zeros(160, dtype=np.float32),
            generation=2, speaker_handoff=False,
        ))
        source.put(None)

        class MustNotDecode:
            def translate_audio(self, audio):
                raise AssertionError("should reuse matching cached subtitle")

        run_granite(MustNotDecode(), source, translated,
                    get_generation=lambda: 2, session_cache=self.cache)
        reuse = translated.get_nowait()
        self.assertIsInstance(reuse, Subtitle)
        self.assertEqual(reuse.generation, 2)
        self.assertTrue(reuse.speaker_handoff)

    def test_cached_audio_can_satisfy_five_second_playback_lead(self):
        self.cache.record_processed(45.0, 65.0)
        out = object.__new__(MpvOutput)
        out._subtitle_cache = self.cache
        out._buffer_condition = threading.Condition()
        out._translated_until = 50.0
        out._source_finished = False
        out._now = 52.0
        self.assertTrue(out._translation_ready())
        out._now = 62.0
        self.assertFalse(out._translation_ready())


if __name__ == "__main__":
    unittest.main()
