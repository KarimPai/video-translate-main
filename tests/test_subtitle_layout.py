"""Subtitle readability: reading-speed timing, wrapping, splitting and row limits."""
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from auto_translate.config import MpvConfig
from auto_translate.output.mpv import MpvOutput, _ass_many
from auto_translate.session_cache import SessionSubtitleCache
from auto_translate.subtitle_layout import chunk_at, display_until, split_cue, wrap_lines
from auto_translate.types import Subtitle

LONG = ("I went to the convenience store yesterday, and some stranger started talking to me. "
        "So in the end I came back home without buying anything at all.")


class LayoutTests(unittest.TestCase):
    def test_short_text_is_one_line_and_long_text_is_balanced(self):
        self.assertEqual(wrap_lines("Thanks for the Super Chat!", 42), ["Thanks for the Super Chat!"])
        lines = wrap_lines("Isn't it time for everyone to go to bed? You have school tomorrow, right?", 42)
        self.assertEqual(lines, ["Isn't it time for everyone to go to bed?", "You have school tomorrow, right?"])
        self.assertTrue(all(len(line) <= 42 for line in lines))
        self.assertEqual(wrap_lines("a b", 0), ["a b"])

    def test_long_cue_splits_at_sentences_into_chunks_that_fit(self):
        chunks = split_cue(LONG, 42, 2)
        self.assertEqual(chunks, ["I went to the convenience store yesterday, and some stranger started talking to me.",
                                  "So in the end I came back home without buying anything at all."])
        self.assertTrue(all(len(wrap_lines(chunk, 42)) <= 2 for chunk in chunks))
        self.assertEqual(" ".join(chunks), LONG)
        self.assertEqual(split_cue(LONG, 0, 2), [LONG])

    def test_oversized_sentence_splits_at_clauses_then_words(self):
        text = ", ".join(["one two three four five six seven eight"] * 4)
        chunks = split_cue(text, 20, 1)
        self.assertEqual(" ".join(chunks), text)
        self.assertTrue(all(len(chunk) <= 20 for chunk in chunks))

    def test_chunks_follow_speech_proportionally(self):
        chunks = ["a" * 10, "b" * 30]
        self.assertEqual(chunk_at(chunks, 0, 4, 0.5), chunks[0])
        self.assertEqual(chunk_at(chunks, 0, 4, 2), chunks[1])
        self.assertEqual(chunk_at(chunks, 0, 4, 9), chunks[1])

    def test_reading_time_extends_short_cues_up_to_cap(self):
        self.assertEqual(display_until("x" * 34, 10, 11, 1.2, 17, 7), 12)  # 2s to read
        self.assertEqual(display_until("x" * 340, 10, 11, 1.2, 17, 7), 17)  # capped
        self.assertEqual(display_until("x" * 340, 10, 20, 1.2, 17, 7), 20)  # speech is longer
        self.assertEqual(display_until("x", 10, 10.5, 1.2, 0, 7), 11.2)  # disabled: minimum only

    def test_config_validation(self):
        for bad in ({"reading_cps": -1}, {"max_line_chars": 8}, {"max_cue_lines": 0},
                    {"max_subtitle_duration_s": 11}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                MpvConfig(**bad)


class MpvReadabilityTests(unittest.TestCase):
    def make_output(self, live=False):
        output = MpvOutput.__new__(MpvOutput)
        output.ingest = SimpleNamespace(source=SimpleNamespace(live=live), generation=0)
        output.cfg = MpvConfig()
        output._lines = []
        output._shown = ""
        output._now = 0.0
        output._session_start = 0.0
        output._lock = threading.Lock()
        output._subtitle_ready = threading.Event()
        output._player_command = Mock(return_value=True)
        return output

    def test_show_extends_short_cue_to_reading_time(self):
        output = self.make_output()
        output.show(Subtitle(0, 1, "あ", "x" * 51, speech_start_s=0.5))
        self.assertAlmostEqual(output._lines[0].show_until, 3.9)  # 51 chars at 15 cps

    def test_render_wraps_rows_and_shows_one_chunk_of_long_cue(self):
        output = self.make_output()
        output.show(Subtitle(0, 8, "あ", LONG))
        output._render(1)
        self.assertIn("convenience store", output._shown)
        self.assertNotIn("buying", output._shown)
        self.assertEqual(output._shown.count(r"\N"), 1)  # two wrapped rows
        output._render(7)
        self.assertIn("buying", output._shown)
        self.assertNotIn("convenience", output._shown)

    def test_row_limit_counts_wrapped_rows(self):
        output = self.make_output()
        two_rows = "Isn't it time for everyone to go to bed? You have school tomorrow, right?"
        for start in (0, 1, 2):
            output.show(Subtitle(start, 6, "あ", two_rows + f" {start}"))
        output._render(3)
        # Four rows fit two two-row cues; the oldest is dropped.
        self.assertNotIn(" 0", output._shown)
        self.assertIn(" 1", output._shown)
        self.assertIn(" 2", output._shown)

    def test_vod_cache_uses_reading_time(self):
        output = self.make_output()
        cache = SessionSubtitleCache()
        cache.add(Subtitle(10, 10.5, "あ", "x" * 68))
        output._subtitle_cache = cache
        output._render(14.4)
        self.assertIn("xxxx", output._shown)
        output._render(14.7)
        self.assertEqual(output._shown, "")

    def test_ass_wraps_long_english(self):
        cfg = MpvConfig()
        rendered = _ass_many([Subtitle(0, 4, "あ", "Isn't it time for everyone to go to bed? Right?")], cfg)
        self.assertIn(r"bed?\NRight?", rendered)

    def test_background_box_opacity(self):
        sub = [Subtitle(0, 4, "あ", "Hello.")]
        self.assertIn(r"\4a&HB2&", _ass_many(sub, MpvConfig()))  # 30% opaque
        self.assertIn(r"\4a&HFF&", _ass_many(sub, MpvConfig(background_opacity=0)))
        for bad in (-0.1, 1.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                MpvConfig(background_opacity=bad)


if __name__ == "__main__":
    unittest.main()
