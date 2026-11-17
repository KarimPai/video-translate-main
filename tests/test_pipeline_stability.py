"""Offline regressions for live expiry and clean pipeline shutdown."""
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from auto_translate.config import Config, IngestConfig
from auto_translate.ingest import Ingest, Source
from auto_translate.pipeline import run, run_asr, run_translate
from auto_translate.translate.base import TranslatorUnavailable
from auto_translate.types import AudioChunk, AudioProgress, SAMPLE_RATE, SourceEOF, Transcript, Utterance
from auto_translate.vad import FRAME, Segmenter, run_segmenter


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
        primary = Mock(last_rejection="error")
        primary.transcribe.return_value = ""
        input_q, output_q = queue.Queue(), queue.Queue()
        input_q.put(Utterance(2, 3, np.ones(16000, dtype=np.float32),
                              vad_confidence=0.95))
        input_q.put(None)
        run_asr(primary, input_q, output_q, live=False)
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

        def fake_vad(_cfg, audio_q, utt_q, _get_generation, _gate, _stop):
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


class WorkerFailureShutdownTests(unittest.TestCase):
    def test_each_failed_stage_releases_all_capacity_one_edges(self):
        for failed_stage in ("ingest", "vad", "asr", "translate"):
            with self.subTest(stage=failed_stage):
                cfg = Config()
                cfg.safe_playback.enabled = True
                cfg.safe_playback.audio_chunks = cfg.safe_playback.queue_size = 1
                stop = threading.Event()
                ready = threading.Barrier(4)
                workers = []
                thread_type = threading.Thread
                failure = (TranslatorUnavailable("fixture outage") if failed_stage == "translate"
                           else RuntimeError("fixture stage failure"))
                ingest = Mock(source=SimpleNamespace(live=False, start_s=0.0), generation=0)
                primary = Mock(last_failed=False)
                primary.transcribe.return_value = "待って"
                translator = Mock(memory_lines=0)
                translator.translate.return_value = "Wait."
                if failed_stage == "asr":
                    primary.transcribe.side_effect = failure
                if failed_stage == "translate":
                    translator.translate.side_effect = failure

                def thread_factory(*args, **kwargs):
                    worker = thread_type(*args, **kwargs)
                    workers.append(worker)
                    return worker

                def produce(audio_q):
                    ready.wait(timeout=3)
                    if failed_stage == "ingest":
                        raise failure
                    for index in range(4):
                        audio_q.put(AudioChunk(index * SAMPLE_RATE,
                                              np.ones(SAMPLE_RATE, dtype=np.float32), 0.0))
                    audio_q.put(None)

                def segment(_seg, audio_q, utt_q, *_args):
                    ready.wait(timeout=3)
                    if failed_stage == "vad":
                        raise failure
                    while True:
                        chunk = audio_q.get()
                        if chunk is None:
                            utt_q.put(None)
                            return
                        start = chunk.start_sample / SAMPLE_RATE
                        utt_q.put(Utterance(start, start + 1, chunk.pcm, vad_confidence=0.95))

                def recognize(*args):
                    ready.wait(timeout=3)
                    run_asr(*args)

                def translate(*args):
                    ready.wait(timeout=3)
                    run_translate(*args)

                ingest.run.side_effect = produce
                translator.close.side_effect = lambda: self.assertFalse(any(t.is_alive() for t in workers))
                with (patch("auto_translate.pipeline.Segmenter"),
                      patch("auto_translate.pipeline.threading.Thread", side_effect=thread_factory),
                      patch("auto_translate.pipeline.run_segmenter", side_effect=segment),
                      patch("auto_translate.pipeline.run_asr", side_effect=recognize),
                      patch("auto_translate.pipeline.run_translate", side_effect=translate),
                      patch("auto_translate.pipeline.record_event") as events):
                    with self.assertRaisesRegex(RuntimeError, f"worker '{failed_stage}' failed") as raised:
                        run(ingest, cfg, primary, translator, [], stop)
                self.assertIs(raised.exception.__cause__, failure)
                self.assertEqual(len(workers), 4)
                self.assertFalse(any(t.is_alive() for t in workers))
                self.assertTrue(stop.is_set())
                translator.close.assert_called_once()
                events.assert_any_call("pipeline_cleanup", active_threads=[])


class SourceCompletionTests(unittest.TestCase):
    def test_eof_flushes_speech_and_progress_before_nonterminal_marker(self):
        cfg = Config()
        audio_q, utt_q = queue.Queue(), queue.Queue()
        pcm = np.full(SAMPLE_RATE, 0.02, dtype=np.float32)
        vad = Mock(side_effect=lambda audio: np.full(len(audio) // FRAME, 0.95))
        with patch("auto_translate.vad.StreamingFireRedVAD", return_value=vad):
            segmenter = Segmenter(cfg.vad)
        audio_q.put(AudioChunk(0, pcm, 0.0))
        audio_q.put(SourceEOF(1))
        audio_q.put(SourceEOF(0))
        audio_q.put(None)
        run_segmenter(segmenter, audio_q, utt_q, lambda: 0)
        items = list(utt_q.queue)
        self.assertEqual([type(item) for item in items], [Utterance, AudioProgress, SourceEOF, type(None)])
        np.testing.assert_array_equal(items[0].audio, pcm)
        self.assertEqual(items[1].sample, SAMPLE_RATE)
        self.assertEqual(items[2].generation, 0)

    def test_stop_during_vad_feed_emits_no_pending_speech_or_progress(self):
        stop = threading.Event()
        segmenter = Mock()
        segmenter.feed.side_effect = lambda _chunk: (stop.set() or [Mock()])
        audio_q, utt_q = queue.Queue(), queue.Queue()
        audio_q.put(AudioChunk(0, np.zeros(SAMPLE_RATE, dtype=np.float32), 0.0))
        audio_q.put(None)
        run_segmenter(segmenter, audio_q, utt_q, lambda: 0, stop=stop)
        self.assertEqual(list(utt_q.queue), [None])
        segmenter.flush.assert_not_called()
        segmenter.advance_progress.assert_not_called()

    def test_repeated_cold_seeks_after_eof_reuse_the_entire_graph(self):
        cfg = Config()
        cfg.safe_playback.enabled = True
        cfg.safe_playback.audio_chunks = cfg.safe_playback.queue_size = 1
        stop = threading.Event()
        ingest = Ingest(Source("synthetic.wav", False), IngestConfig())
        ingest.enable_vod_restart()
        self.addCleanup(ingest.stop)
        workers = []
        thread_type = threading.Thread
        primary = Mock(last_failed=False)
        primary.transcribe.return_value = "待って"
        translator = Mock(memory_lines=0)
        translator.translate.return_value = "Wait."
        output = Mock()
        output.scheduler_failure = None
        completed = []
        read_positions = []

        def thread_factory(*args, **kwargs):
            worker = thread_type(*args, **kwargs)
            workers.append(worker)
            return worker

        def read(audio_q):
            read_positions.append(ingest.source.start_s)
            audio_q.put(AudioChunk(round(ingest.source.start_s * SAMPLE_RATE),
                                  np.full(SAMPLE_RATE, 0.02, dtype=np.float32), 0.0,
                                  generation=ingest.generation))
            return False

        def finish(generation):
            completed.append(generation)
            self.assertTrue(all(t.is_alive() for t in workers))
            shown = [call.args[0] for call in output.show.call_args_list]
            self.assertEqual([sub.generation for sub in shown], list(range(generation + 1)))
            if generation < 2:
                ingest.request_restart((generation + 1) * 20.0)
                ingest.go()
            else:
                stop.set()

        output.finish.side_effect = finish
        vad = Mock(side_effect=lambda audio: np.full(len(audio) // FRAME, 0.95))
        with (patch.object(ingest, "_read_session", side_effect=read),
              patch("auto_translate.vad.StreamingFireRedVAD", return_value=vad) as load_vad,
              patch("auto_translate.pipeline.threading.Thread", side_effect=thread_factory)):
            run(ingest, cfg, primary, translator, [output], stop)
        self.assertEqual(completed, [0, 1, 2])
        self.assertEqual(read_positions, [0.0, 20.0, 40.0])
        self.assertEqual([call.args[0].start_s for call in output.show.call_args_list], read_positions)
        self.assertEqual(primary.transcribe.call_count, 3)
        self.assertEqual(translator.translate.call_count, 3)
        load_vad.assert_called_once()
        self.assertEqual(len(workers), 4)
        self.assertFalse(any(t.is_alive() for t in workers))
        output.drain.assert_not_called()
        translator.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
