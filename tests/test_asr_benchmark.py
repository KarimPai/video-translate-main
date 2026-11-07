"""Offline smoke tests for the optional speech.cpp benchmark utility."""

import importlib.util
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "tools" / "compare_asr.py"
spec = importlib.util.spec_from_file_location("compare_asr", path)
compare_asr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_asr)


class BenchmarkHelpersTests(unittest.TestCase):
    def test_youtube_id_accepts_watch_and_short_links(self):
        self.assertEqual(
            compare_asr.youtube_id("https://www.youtube.com/watch?v=LE8iGHrakAc"),
            "LE8iGHrakAc",
        )
        self.assertEqual(
            compare_asr.youtube_id("https://youtu.be/LE8iGHrakAc?t=14"),
            "LE8iGHrakAc",
        )

    def test_youtube_id_rejects_external_or_bad_urls(self):
        for link in ("https://example.com/LE8iGHrakAc", "https://youtu.be/123", ""):
            with self.subTest(link=link), self.assertRaises(ValueError):
                compare_asr.youtube_id(link)

    def test_parser_accepts_speech_jsonlines_and_checks_count(self):
        text = (
            '{"file":"one.wav","text":"シャッター"}\n'
            '{"file":"two.wav","text":"シャッター、シャッター"}\n'
        )
        self.assertEqual(
            compare_asr.parse_lines(text, 2),
            ["シャッター", "シャッター、シャッター"],
        )
        with self.assertRaises(ValueError):
            compare_asr.parse_lines(text, 3)


if __name__ == "__main__":
    unittest.main()
