"""Offline regressions for live expiry and clean pipeline shutdown."""
import queue
import unittest
from unittest.mock import Mock, patch

import numpy as np

from auto_translate.config import Config
from auto_translate.pipeline import run_asr, run_translate
from auto_translate.types import Transcript, Utterance


class ExpiredLiveWorkTests(unittest.TestCase):
    def test_asr_skips_only_cues_already_past_player(self):
        primary = Mock()
        primary.transcribe.return_value = "待って"
        input_q, output_q = queue.Queue(), queue.Queue()
        input_q.put(Utterance(2, 3, np.ones(16000, dtype=np.float32)))
        input_q.put(None)
        run_asr(primary, input_q, output_q, live=True, playback_position=lambda: 20.0,
                live_late_window_s=7.0)
        primary.transcribe.assert_not_called()
        self.assertEqual(list(output_q.queue), [None])

    def test_mismatched_mpv_clock_never_discards_all_live_speech(self):
        primary = Mock()
        primary.transcribe.return_value = "待って"
        input_q, output_q = queue.Queue(), queue.Queue()
        input_q.put(Utterance(2, 3, np.ones(16000, dtype=np.float32)))
        input_q.put(None)
        run_asr(primary, input_q, output_q, live=True, playback_position=lambda: 15260.0,
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
    def test_asr_cancelled_before_join_and_translator_closes_after_workers(self):
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
        translator.close.side_effect = lambda: self.assertTrue(mt_done.is_set())
        with patch("auto_translate.pipeline.Segmenter"), patch("auto_translate.pipeline.run_segmenter", side_effect=fake_vad), \
             patch("auto_translate.pipeline.run_asr", side_effect=fake_asr), \
             patch("auto_translate.pipeline.run_translate", side_effect=fake_translate):
            run(ingest, cfg, asr, translator, [], stop)
        self.assertTrue(asr_done.is_set())
        self.assertTrue(mt_done.is_set())
        self.assertGreaterEqual(asr.close.call_count, 1)
        translator.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
