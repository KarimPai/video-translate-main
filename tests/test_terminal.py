"""Terminal formatting regressions for the single-process Windows launcher."""
from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

from auto_translate.output.terminal import TerminalOutput
from auto_translate.types import Subtitle


class TerminalOutputTests(unittest.TestCase):
    def test_simple_mode_writes_only_japanese_and_english_lines(self):
        target = StringIO()
        with redirect_stdout(target):
            TerminalOutput(simple=True).show(
                Subtitle(1.0, 2.0, "こんにちは", "Hello.")
            )
        self.assertEqual(target.getvalue().splitlines(), [
            "JP: こんにちは",
            "EN: Hello.",
        ])

    def test_simple_mode_shows_speaker_only_in_cmd(self):
        target = StringIO()
        with redirect_stdout(target):
            TerminalOutput(simple=True).show(
                Subtitle(1.0, 2.0, "危ない！", "Watch out!", speaker_id=0)
            )
            TerminalOutput(simple=True).show(
                Subtitle(2.0, 3.0, "わかった", "Got it.", speaker_id=1)
            )
        self.assertEqual(target.getvalue().splitlines(), [
            "JP [A]: 危ない！",
            "EN [A]: Watch out!",
            "JP [B]: わかった",
            "EN [B]: Got it.",
        ])

    def test_verbose_mode_shows_speaker_in_cmd(self):
        target = StringIO()
        with redirect_stdout(target):
            TerminalOutput(simple=False).show(
                Subtitle(1.0, 2.0, "危ない", "Watch out!", speaker_id=0)
            )
        self.assertIn("[A] 危ない", target.getvalue())
        self.assertIn("→ [A] Watch out!", target.getvalue())

    def test_simple_mode_flushes_once_for_bilingual_subtitle(self):
        with patch("builtins.print") as output:
            TerminalOutput(simple=True).show(
                Subtitle(1.0, 2.0, "こんにちは", "Hello.")
            )
        output.assert_called_once_with(
            "JP: こんにちは\nEN: Hello.", flush=True
        )

    def test_default_output_preserves_timestamps_and_timing_fields(self):
        target = StringIO()
        with redirect_stdout(target):
            TerminalOutput().show(Subtitle(1.0, 2.0, "日本語", "English"))
        actual = target.getvalue()
        self.assertIn("[00:01.0–00:02.0]", actual)
        self.assertIn("日本語", actual)
        self.assertIn("→ English", actual)
        self.assertIn("vad ", actual)

    def test_simple_mode_skips_empty_translation(self):
        target = StringIO()
        with redirect_stdout(target):
            TerminalOutput(simple=True).show(Subtitle(1.0, 2.0, "はい", ""))
        self.assertEqual(target.getvalue(), "JP: はい\n")


if __name__ == "__main__":
    unittest.main()
