import json
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from auto_translate.captions import CaptionCue, CaptionIndex
from auto_translate.config import Config, LanguageFilterConfig, PreviewConfig, TranslateConfig, load_config
from auto_translate.language_filter import LanguageFilter
from auto_translate.output.review import PreviewDispatcher, ReviewOutput
from auto_translate.review import DiagnosticTail, TranscriptReview
from auto_translate.types import SAMPLE_RATE, Subtitle


class AccuracyPreventionTests(unittest.TestCase):
    def test_japanese_in_unsampled_middle_is_preserved(self):
        f = object.__new__(LanguageFilter)
        f.cfg, f.retry_at = LanguageFilterConfig(), 0.0
        audio = np.zeros(16 * SAMPLE_RATE, dtype=np.float32)
        audio[5 * SAMPLE_RATE:11 * SAMPLE_RATE] = 1.0
        f._predict = lambda clip: (("ja", .99, .99, .98) if clip.max() else ("en", .99, .001, .98))
        self.assertFalse(f.classify(audio).drop)

    def test_overlapping_captions_do_not_inflate_coverage(self):
        index = CaptionIndex([CaptionCue(0, .4, "はい"), CaptionCue(.05, .45, "はい")])
        match = index.match(0, 2)
        self.assertAlmostEqual(match.coverage, .225)
        self.assertFalse(match.reliable)

    def test_preview_config_and_cue_deadline_validation(self):
        self.assertFalse(Config().preview.enabled)
        for kwargs in ({"interval_s": float("nan")}, {"min_agreement": True}, {"window_s": .1}):
            with self.assertRaises(ValueError):
                PreviewConfig(**kwargs)
        with self.assertRaises(ValueError):
            TranslateConfig(cue_timeout_s=float("inf"))
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "config.toml"
            p.write_text("[preview]\nenabled=true\n[translate]\ncue_timeout_s=55\n")
            cfg = load_config(p)
            self.assertTrue(cfg.preview.enabled)
            self.assertEqual(cfg.translate.cue_timeout_s, 55)


class ReviewTests(unittest.TestCase):
    def test_corrections_survive_automated_replays_and_track_revisions(self):
        review = TranscriptReview()
        key = review.add(dict(start_s=1, end_s=3, ja="はい", en="Yes"))
        review.correct(key, "はい", "Okay")
        review.add(dict(start_s=1, end_s=3, ja="はい", en="Yes"))
        self.assertEqual(review.snapshot()[0].en, "Okay")
        self.assertEqual(len(review.cues[key].revisions), 1)
        self.assertEqual(len(review.filtered("okay")), 1)

    def test_diagnostics_preceding_subtitle_and_abstentions_are_reviewable(self):
        review = TranscriptReview()
        review.add_diagnostic(dict(start_s=1, end_s=3, chosen_by="uncertain", flags=["negation"]))
        review.add(dict(start_s=1, end_s=3, ja="はい", en="Yes"))
        review.add_diagnostic(dict(start_s=5, end_s=8, chosen_by="abstain_disagreement", haas="不明"))
        self.assertEqual(len(review.filtered(needs_review=True)), 2)

    def test_invalid_timestamps_are_rejected(self):
        for start, end in ((float("nan"), 1), (0, float("inf")), (True, 2), (-1, 2), (3, 1)):
            with self.assertRaises(ValueError):
                TranscriptReview().add(dict(start_s=start, end_s=end, ja="", en=""))

    def test_diagnostic_tail_handles_partial_lines_and_truncation(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "comparison.jsonl"
            p.write_bytes(b'{"old":true}\n')
            tail = DiagnosticTail(p)
            with p.open("ab") as stream:
                stream.write(b'{"new":true}')
            self.assertEqual(tail.read(), [])
            with p.open("ab") as stream:
                stream.write(b'\ninvalid\n')
            self.assertEqual(tail.read(), [{"new": True}])
            p.write_bytes(b'{"x":1}\n')
            self.assertEqual(tail.read(), [{"x": 1}])

    def test_preview_dispatcher_discards_stale_generation(self):
        seen = queue.Queue()
        output = Mock()
        output.show_preview.side_effect = seen.put
        hub = PreviewDispatcher([output], lambda: 3, threading.Event())
        try:
            hub.submit(dict(generation=2, ja="old"))
            hub.submit(dict(generation=3, ja="new"))
            self.assertEqual(seen.get(timeout=1)["ja"], "new")
        finally:
            hub.close()

    def test_structured_review_events_never_persist_previews(self):
        output = ReviewOutput(gui=True)
        with patch("auto_translate.gui_control.emit_event") as emit:
            output.show(Subtitle(1, 2, "はい", "Yes"))
            self.assertEqual(emit.call_args.args, ("subtitle",))
            self.assertNotIn("timings", emit.call_args.kwargs)
            output.show_preview(dict(kind="preview", generation=1, ja="draft"))
        self.assertFalse(output._cues)

    def test_late_draft_cannot_overwrite_an_already_finalized_gui_cue(self):
        output = ReviewOutput(gui=True)
        with patch("auto_translate.gui_control.emit_event") as emit:
            output.show(Subtitle(1, 4, "はい", "Yes", generation=2))
            output.show_preview(dict(kind="preview", generation=2, start_s=1, end_s=3, ja="draft"))
            self.assertEqual(emit.call_count, 1)
            output.show_preview(dict(kind="preview", generation=2, start_s=5, end_s=6, ja="next"))
            self.assertEqual(emit.call_count, 2)

    def test_bilingual_abstention_roundtrips_without_fabricating_english(self):
        from auto_translate.subtitle_files import export_subtitles, import_subtitles
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "subtitles.srt"
            export_subtitles(path, [Subtitle(1, 2, "はい", "")])
            loaded = import_subtitles(path, language="bilingual")
            self.assertEqual((loaded[0].ja, loaded[0].en), ("はい", ""))
            with self.assertRaises(ValueError):
                export_subtitles(path, [Subtitle(1, 2, "", "English only")])


class ToolEntrypointTests(unittest.TestCase):
    def test_stale_gui_fails_with_actionable_message_before_loading_models(self):
        import io
        from contextlib import ExitStack
        from auto_translate.__main__ import _main
        error = io.StringIO()
        with patch("sys.argv", ["auto-translate", "https://youtu.be/example", "--gui-control"]), \
             patch("sys.stderr", error), \
             patch("auto_translate.cuda_dlls.setup") as setup:
            with ExitStack() as resources, self.assertRaises(SystemExit) as stopped:
                _main(resources)
        self.assertEqual(stopped.exception.code, 2)
        self.assertIn("GUI/backend version mismatch", error.getvalue())
        self.assertIn("python -m auto_translate.gui", error.getvalue())
        setup.assert_not_called()

    def test_quality_report_needs_no_media_or_model_startup(self):
        from contextlib import ExitStack
        from auto_translate.__main__ import _main
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "events.jsonl"
            p.write_text(json.dumps(dict(stage="reconciliation", session_id="test", start_s=1, end_s=2)) + "\n")
            with patch("sys.argv", ["auto-translate", "--quality-report", str(p)]):
                with ExitStack() as resources:
                    _main(resources)
            self.assertTrue(list(Path(folder).rglob("*.html")))


if __name__ == "__main__":
    unittest.main()
