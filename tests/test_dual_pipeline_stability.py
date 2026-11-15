"""Offline regressions for live backlog shedding and stalled Haas pipes."""
import queue
import subprocess
import sys
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from auto_translate.asr.haas import HaasASR
from auto_translate.config import Config
from auto_translate.pipeline import run_asr, run_translate
from auto_translate.types import Transcript, Utterance


class ExpiredLiveWorkTests(unittest.TestCase):
    def test_asr_skips_only_cues_already_past_player(self):
        primary = Mock()
        primary.transcribe.return_value = "待って"
        optional = Mock()
        optional.playback_position = lambda: 20.0
        input_q, output_q = queue.Queue(), queue.Queue()
        input_q.put(Utterance(2, 3, np.ones(16000, dtype=np.float32)))
        input_q.put(None)
        run_asr(primary, input_q, output_q, live=True, dual=optional,
                live_late_window_s=7.0)
        primary.transcribe.assert_not_called()
        self.assertEqual(list(output_q.queue), [None])

    def test_mismatched_mpv_clock_never_discards_all_live_speech(self):
        primary = Mock()
        primary.transcribe.return_value = "待って"
        optional = Mock()
        optional.playback_position = lambda: 15260.0
        optional.select_segment.return_value = None
        input_q, output_q = queue.Queue(), queue.Queue()
        input_q.put(Utterance(2, 3, np.ones(16000, dtype=np.float32)))
        input_q.put(None)
        run_asr(primary, input_q, output_q, live=True, dual=optional,
                live_late_window_s=7.0)
        primary.transcribe.assert_called_once()
        self.assertEqual(len(output_q.queue), 2)

    def test_vod_transport_failure_never_immediately_retries(self):
        cfg = Config().asr
        primary = Mock(last_failed=True)
        primary.transcribe.return_value = ""
        input_q, output_q = queue.Queue(), queue.Queue()
        input_q.put(Utterance(2, 3, np.ones(16000, dtype=np.float32),
                              vad_confidence=0.95))
        input_q.put(None)
        run_asr(primary, input_q, output_q, live=False, asr_cfg=cfg)
        primary.transcribe.assert_called_once()

    def test_vod_keeps_older_cues(self):
        primary = Mock()
        primary.transcribe.return_value = "待って"
        input_q, output_q = queue.Queue(), queue.Queue()
        input_q.put(Utterance(2, 3, np.ones(16000, dtype=np.float32)))
        input_q.put(None)
        run_asr(primary, input_q, output_q, live=False)
        primary.transcribe.assert_called_once()
        self.assertEqual(len(output_q.queue), 2)

    def test_plamo_skips_expired_live_transcript(self):
        translator = Mock()
        translator.memory_lines = 0
        input_q, output_q = queue.Queue(), queue.Queue()
        input_q.put(Transcript(2, 3, "待って"))
        input_q.put(None)
        run_translate(translator, 0, input_q, output_q, live=True,
                      playback_position=lambda: 20.0, live_late_window_s=7.0)
        translator.translate.assert_not_called()
        self.assertEqual(list(output_q.queue), [None])

    def test_uninitialized_player_preserves_cues(self):
        translator = Mock()
        translator.memory_lines = 0
        translator.translate.return_value = "Wait."
        input_q, output_q = queue.Queue(), queue.Queue()
        input_q.put(Transcript(2, 3, "待って"))
        input_q.put(None)
        run_translate(translator, 0, input_q, output_q, live=True,
                      playback_position=lambda: None, live_late_window_s=7.0)
        translator.translate.assert_called_once()
        self.assertEqual(len(output_q.queue), 2)


class StalledPipeTests(unittest.TestCase):
    def test_stdin_deadline_applies_when_child_does_not_read(self):
        # Large PCM saturates the OS pipe while the child remains alive.
        script = "import json, time; print(json.dumps({'type': 'ready'}), flush=True); time.sleep(5)"
        real_spawn = subprocess.Popen

        def fake_spawn(cmd, **kwargs):
            return real_spawn([sys.executable, "-u", "-c", script], **kwargs)

        cfg = Config().asr
        cfg.response_timeout_s = 0.15
        cfg.startup_timeout_s = 3.0
        with patch("auto_translate.asr.haas.shutil.which", return_value="speech"), patch(
                "auto_translate.asr.haas.subprocess.Popen", side_effect=fake_spawn):
            worker = HaasASR(cfg)
            try:
                start = time.monotonic()
                with self.assertRaises(TimeoutError):
                    worker.transcribe(np.ones(40000, dtype=np.float32))
                self.assertLess(time.monotonic() - start, 1.0)
            finally:
                worker.close()


class SelectorPressureTests(unittest.TestCase):
    def test_soft_pressure_releases_only_gpu_selector(self):
        import threading
        from auto_translate.asr.dual import DualASR
        dual = object.__new__(DualASR)
        dual.lock = threading.Lock()
        selector = Mock(gpu_layers=0)
        dual.selector = selector
        dual.release_selector()
        selector.close.assert_not_called()
        self.assertIs(dual.selector, selector)

        selector.gpu_layers = 99
        dual.release_selector()
        selector.close.assert_called_once()
        self.assertIsNone(dual.selector)


class BackpressureTests(unittest.TestCase):
    def test_vod_bypasses_optional_when_primary_usable_and_backlogged(self):
        from auto_translate.asr.dual import DualASR
        from auto_translate.glossary import Glossary
        cfg = Config()
        cfg.dual_asr.enabled = False
        dual = DualASR(cfg, Mock(used=3., cfg=cfg.gpu_budget), Glossary({}), False)
        dual.worker = Mock()
        utt = Utterance(2, 3, np.ones(16000, dtype=np.float32), vad_confidence=0.9)
        result = dual.select_segment(utt, "待って", "", lambda: True, lambda: 0)
        self.assertIsNone(result)
        dual.worker.transcribe.assert_not_called()


class AsrComparisonTests(unittest.TestCase):
    def test_exact_lfm_choice_and_both_japanese_candidates_are_recorded(self):
        import threading
        from auto_translate.asr.dual import DualASR
        from auto_translate.glossary import Glossary
        cfg = Config()
        cfg.dual_asr.enabled = False
        budget = Mock(pressure_level="normal")
        dual = DualASR(cfg, budget, Glossary({}), False)
        dual.worker = Mock(last_stop="complete")
        dual.worker.transcribe.return_value = "まだ行かない"
        dual.selector = Mock(last_choice="whisper_turbo")
        dual.selector.choose.return_value = "まだ行かない"
        utt = Utterance(10.0, 11.0, np.ones(16000, dtype=np.float32),
                        vad_confidence=0.96)
        with patch("auto_translate.asr.dual.append_comparison") as save:
            result = dual.choose(utt, "行く", "", lambda: False, None,
                                 "complete", 0.15)
        self.assertEqual(result, "まだ行かない")
        record = save.call_args.args[0]
        self.assertEqual(record["haas"], "行く")
        self.assertEqual(record["whisper_turbo"], "まだ行かない")
        self.assertEqual(record["chosen_by"], "whisper_turbo")
        self.assertEqual(record["chosen"], "まだ行かない")
        self.assertTrue(record["selector_called"])
        self.assertIn("negation", record["flags"])
        self.assertEqual(record["haas_elapsed_s"], 0.15)

    def test_comparison_log_contains_utf8_transcripts_and_rotates(self):
        import json
        import tempfile
        from pathlib import Path
        from auto_translate.asr.diagnostics import append_comparison, flush_comparisons
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "asr-comparison.jsonl"
            append_comparison({
                "haas": "行く", "whisper_turbo": "まだ行かない",
                "chosen_by": "whisper_turbo", "selector_called": True,
            }, path=path)
            self.assertTrue(flush_comparisons(3))
            payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(payload["haas"], "行く")
            self.assertEqual(payload["whisper_turbo"], "まだ行かない")
            self.assertEqual(payload["chosen_by"], "whisper_turbo")
            with patch("auto_translate.asr.diagnostics._MAX_FILE_BYTES", 1):
                append_comparison({"haas": "次", "whisper_turbo": "次"}, path=path)
                self.assertTrue(flush_comparisons(3))
            self.assertTrue((Path(td) / "asr-comparison.previous.jsonl").exists())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["haas"], "次")

    def test_selected_whisper_turbo_name_uses_matching_caption_evidence(self):
        # Previously selecting secondary meant choice != None and skipped
        # even exact aligned caption-based Watame name repair.
        from auto_translate.captions import CaptionMatch
        primary = Mock()
        primary.transcribe.return_value = "渡辺、そこにいるの？"
        dual = Mock()
        dual.select_segment.return_value = "渡辺、そこにいるの？"
        caption = Mock()
        caption.match.return_value = CaptionMatch(
            "わため、そこにいるの？", overlap_s=1.0, coverage=1.0, reliable=True
        )
        input_q, output_q = queue.Queue(), queue.Queue()
        input_q.put(Utterance(2, 3, np.ones(16000, dtype=np.float32),
                              vad_confidence=.9))
        input_q.put(None)
        run_asr(primary, input_q, output_q, captions=caption, dual=dual)
        transcript = output_q.get_nowait()
        self.assertEqual(transcript.ja, "わため、そこにいるの？")
        self.assertIsNone(output_q.get_nowait())
        caption.match.assert_called_once_with(2, 3)

    def test_selected_whisper_turbo_watame_name_uses_confirmed_video_context(self):
        primary = Mock()
        primary.transcribe.return_value = "渡辺、行こう"
        dual = Mock()
        dual.select_segment.return_value = "渡辺、行こう"
        input_q, output_q = queue.Queue(), queue.Queue()
        input_q.put(Utterance(2, 3, np.ones(16000, dtype=np.float32),
                              vad_confidence=.9))
        input_q.put(None)
        run_asr(primary, input_q, output_q,
                vod_title="角巻わため 配信", dual=dual)
        self.assertEqual(output_q.get_nowait().ja, "わため、行こう")
        self.assertIsNone(output_q.get_nowait())

    def test_explicit_watanabe_caption_prevents_unfounded_name_change(self):
        from auto_translate.captions import CaptionMatch
        primary = Mock()
        primary.transcribe.return_value = "渡辺さん"
        dual = Mock()
        dual.select_segment.return_value = "渡辺さん"
        captions = Mock()
        captions.match.return_value = CaptionMatch(
            "わたなべさん", overlap_s=1.0, coverage=1.0, reliable=True
        )
        input_q, output_q = queue.Queue(), queue.Queue()
        input_q.put(Utterance(2, 3, np.ones(16000, dtype=np.float32),
                              vad_confidence=.9))
        input_q.put(None)
        run_asr(primary, input_q, output_q, captions=captions,
                vod_title="角巻わため 配信", dual=dual)
        self.assertEqual(output_q.get_nowait().ja, "渡辺さん")
        self.assertIsNone(output_q.get_nowait())

    def test_invalid_secondary_is_recorded_without_changing_primary(self):
        from auto_translate.asr.dual import DualASR
        from auto_translate.glossary import Glossary
        cfg = Config()
        cfg.dual_asr.enabled = False
        dual = DualASR(cfg, Mock(pressure_level="normal"), Glossary({}), False)
        dual.worker = Mock(last_stop="complete")
        dual.worker.transcribe.return_value = ""
        utt = Utterance(2, 3, np.ones(16000, dtype=np.float32),
                        vad_confidence=0.95)
        with patch("auto_translate.asr.dual.append_comparison") as save:
            result = dual.choose(utt, "待って", "", lambda: False, None)
        self.assertIsNone(result)
        self.assertEqual(save.call_args.args[0]["chosen_by"],
                         "haas_secondary_unusable")


class ShutdownCancellationTests(unittest.TestCase):
    def test_queued_asr_is_discarded_after_player_close(self):
        import threading
        stop = threading.Event()
        stop.set()
        aq, tq = queue.Queue(), queue.Queue()
        aq.put(Utterance(2, 3, np.ones(16000, dtype=np.float32)))
        aq.put(None)
        worker = Mock()
        run_asr(worker, aq, tq, stop=stop)
        worker.transcribe.assert_not_called()
        self.assertEqual(list(tq.queue), [None])

    def test_queued_translation_is_discarded_after_player_close(self):
        import threading
        stop = threading.Event()
        stop.set()
        tq, sq = queue.Queue(), queue.Queue()
        tq.put(Transcript(2, 3, "待って"))
        tq.put(None)
        worker = Mock(memory_lines=0)
        run_translate(worker, 0, tq, sq, stop=stop)
        worker.translate.assert_not_called()
        self.assertEqual(list(sq.queue), [None])


class ShutdownOrderingTests(unittest.TestCase):
    def test_models_close_only_after_in_flight_pipeline_threads_exit(self):
        import threading
        from auto_translate.pipeline import run

        stop = threading.Event()
        stop.set()  # MPV window already closed; exercise shutdown path.
        unblock_ingest = threading.Event()
        asr_done = threading.Event()
        mt_done = threading.Event()
        cfg = Config()
        ingest = Mock()
        ingest.source.live = False
        ingest.source.start_s = 0.0
        ingest.generation = 0
        ingest.stop.side_effect = unblock_ingest.set

        def produce(q):
            if not unblock_ingest.wait(3):
                raise TimeoutError("ingest not stopped")
            q.put(None)

        def fake_vad(_cfg, audio_q, utt_q, _get_generation):
            self.assertIsNone(audio_q.get(timeout=3))
            utt_q.put(None)

        def fake_asr(_worker, utt_q, text_q, *args):
            self.assertIsNone(utt_q.get(timeout=3))
            asr_done.set()
            text_q.put(None)

        def fake_translate(_worker, _context, text_q, sub_q, *args):
            self.assertIsNone(text_q.get(timeout=3))
            mt_done.set()
            sub_q.put(None)

        ingest.run.side_effect = produce
        asr = Mock()
        translator = Mock()
        asr.close.side_effect = lambda: self.assertTrue(asr_done.is_set())
        translator.close.side_effect = lambda: self.assertTrue(mt_done.is_set())
        with patch("auto_translate.pipeline.Segmenter"), patch("auto_translate.pipeline.run_segmenter", side_effect=fake_vad), \
             patch("auto_translate.pipeline.run_asr", side_effect=fake_asr), \
             patch("auto_translate.pipeline.run_translate", side_effect=fake_translate):
            run(ingest, cfg, asr, translator, [], stop)
        self.assertTrue(asr_done.is_set())
        self.assertTrue(mt_done.is_set())
        asr.close.assert_called_once()
        translator.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
