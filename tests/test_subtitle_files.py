"""File interchange regressions; optional format library, no model downloads."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from auto_translate.subtitle_files import (
    export_subtitles,
    import_corrections,
    import_subtitles,
    validate_subtitles,
)
from auto_translate.types import Subtitle


class SubtitleValidationTests(unittest.TestCase):
    def test_invalid_timestamps_and_ambiguous_duplicate_rejected(self):
        for start, end in [(float("nan"), 2), (1, float("inf")), (-1, 2), (1, 1)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                validate_subtitles([Subtitle(start, end, "日", "day")])
        cue = Subtitle(1, 2, "日", "day")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_subtitles([cue, cue])

    def test_missing_optional_dependency_has_actionable_error(self):
        with (tempfile.TemporaryDirectory() as directory,
              patch.dict("sys.modules", {"pysubs2": None}),
              self.assertRaisesRegex(RuntimeError, "subtitles.*extra")):
            export_subtitles(Path(directory) / "test.srt", [Subtitle(1, 2, "日", "day")])


@unittest.skipUnless(importlib.util.find_spec("pysubs2"), "optional pysubs2 not installed")
class SubtitleFileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.cues = [Subtitle(1.005, 2.025, "こんにちは", "Hello", generation=4, speech_start_s=1.1)]

    def test_roundtrip_all_formats_and_languages(self):
        for extension in ("srt", "vtt", "ass"):
            for language in ("japanese", "english", "bilingual"):
                with self.subTest(extension=extension, language=language):
                    path = self.directory / f"cue.{extension}"
                    self.assertEqual(export_subtitles(path, self.cues, language), 1)
                    actual = import_subtitles(path, language=language)[0]
                    if language != "english":
                        self.assertEqual(actual.ja, "こんにちは")
                    if language != "japanese":
                        self.assertEqual(actual.en, "Hello")
                    tolerance = .01 if extension == "ass" else .001
                    self.assertAlmostEqual(actual.start_s, self.cues[0].start_s, delta=tolerance)

    def test_corrections_preserve_exact_original_timestamps_and_metadata(self):
        for extension in ("srt", "vtt", "ass"):
            with self.subTest(extension=extension):
                path = self.directory / f"fixed.{extension}"
                fixed = Subtitle(1.005, 2.025, "こんにちは", "Greetings")
                export_subtitles(path, [fixed], "english")
                revisions = import_corrections(path, self.cues)
                self.assertEqual(len(revisions), 1)
                self.assertEqual(revisions[0].original.en, "Hello")
                self.assertEqual(revisions[0].revised.en, "Greetings")
                corrected = import_subtitles(path, self.cues)[0]
                self.assertEqual((corrected.start_s, corrected.end_s), (1.005, 2.025))
                self.assertEqual(corrected.generation, 4)
                self.assertEqual(corrected.speech_start_s, 1.1)
                self.assertEqual(corrected.ja, "こんにちは")
                self.assertEqual(self.cues[0].en, "Hello")
                corrected.timings.audio_read = 99
                self.assertEqual(self.cues[0].timings.audio_read, 0)

    def test_retimed_corrections_rejected_and_subset_preserves_other_cues(self):
        path = self.directory / "fix.srt"
        export_subtitles(path, [Subtitle(3, 4, "次", "Next")], "english")
        with self.assertRaisesRegex(ValueError, "timestamps do not match"):
            import_corrections(path, self.cues)
        originals = self.cues + [Subtitle(3, 4, "次", "Old")]
        actual = import_subtitles(path, originals)
        self.assertEqual([cue.en for cue in actual], ["Hello", "Next"])

    def test_failed_atomic_replace_leaves_original_and_removes_temporary(self):
        path = self.directory / "keep.srt"
        path.write_text("original", encoding="utf8")
        with (patch("auto_translate.subtitle_files.os.replace", side_effect=PermissionError("locked")),
              self.assertRaises(PermissionError)):
            export_subtitles(path, self.cues)
        self.assertEqual(path.read_text(encoding="utf8"), "original")
        self.assertEqual(list(self.directory.iterdir()), [path])

    def test_ass_commands_remain_harmless_visible_text(self):
        path = self.directory / "plain.ass"
        cue = Subtitle(1, 2, "日", r"{\p1}drawing")
        export_subtitles(path, [cue], "english")
        imported = import_subtitles(path)[0]
        self.assertEqual(imported.en, "｛＼p1｝drawing")

    def test_ass_precision_collision_and_overflow_rejected(self):
        path = self.directory / "test.ass"
        with self.assertRaisesRegex(ValueError, "precision"):
            export_subtitles(path, [Subtitle(1.001, 2.001, "日", "Day"),
                                    Subtitle(1.002, 2.002, "月", "Moon")])
        with self.assertRaisesRegex(ValueError, "duration"):
            export_subtitles(path, [Subtitle(36000, 36001, "日", "Day")])

    def test_empty_or_malformed_import_and_empty_export_rejected(self):
        path = self.directory / "bad.srt"
        path.write_text("not a subtitle", encoding="utf8")
        with self.assertRaises(ValueError):
            import_subtitles(path)
        with self.assertRaises(ValueError):
            export_subtitles(path, [Subtitle(1, 2, "日", "")], "english")
        self.assertEqual(path.read_text(encoding="utf8"), "not a subtitle")


if __name__ == "__main__":
    unittest.main()
