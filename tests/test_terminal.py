"""English-only terminal regression tests for Granite direct translation."""

from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

from auto_translate.output.terminal import TerminalOutput
from auto_translate.types import Subtitle


class TerminalOutputTests(unittest.TestCase):
    def test_simple_output_english_only(self):
        buf = StringIO()
        with redirect_stdout(buf):
            TerminalOutput(simple=True).show(Subtitle(1.0, 2.0, "", "Get in the car!"))
        self.assertEqual(buf.getvalue(), "EN: Get in the car!\n")

    def test_two_lines_have_no_speaker_labels(self):
        buf = StringIO()
        with redirect_stdout(buf):
            TerminalOutput(simple=True).show(Subtitle(1.0, 2.0, "", "Watch out!"))
            TerminalOutput(simple=True).show(Subtitle(2.0, 3.0, "", "Got it."))
        self.assertEqual(buf.getvalue().splitlines(), ["EN: Watch out!", "EN: Got it."])

    def test_verbose_output_includes_timestamps_and_granite(self):
        buf = StringIO()
        with redirect_stdout(buf):
            TerminalOutput(simple=False).show(Subtitle(1.0, 2.0, "", "Watch out!"))
        self.assertIn("Granite", buf.getvalue())
        self.assertIn("Watch out!", buf.getvalue())
        self.assertNotIn("JP:", buf.getvalue())

    def test_empty_result_does_not_print(self):
        with patch("builtins.print") as output:
            TerminalOutput(simple=True).show(Subtitle(0.0, 1.0, "", ""))
        output.assert_not_called()


if __name__ == "__main__":
    unittest.main()
