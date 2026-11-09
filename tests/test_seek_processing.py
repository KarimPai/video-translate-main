"""Seek regressions for FireRed audio progress and URL recovery."""
import queue
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import numpy as np

from auto_translate.config import IngestConfig, VadConfig
from auto_translate.ingest import Ingest, Source
from auto_translate.types import SAMPLE_RATE, AudioChunk, AudioProgress, Utterance
from auto_translate.vad import FRAME, run_segmenter


class SeekProcessingTests(unittest.TestCase):
    def test_only_current_pcm_is_processed_and_opt_in_probes_precede_progress(self):
        in_q, out_q = queue.Queue(), queue.Queue()
        start = 30 * SAMPLE_RATE
        # Several old chunks and their reset are still waiting when a new seek lands.
        pcm = np.ones(SAMPLE_RATE, dtype=np.float32)
        in_q.put(AudioChunk(0, pcm, 0, generation=0))
        in_q.put(AudioChunk(0, pcm[:0], 0, reset=True, generation=1))
        in_q.put(AudioChunk(start, pcm[:0], 0, reset=True, generation=2))
        for second in range(12):
            in_q.put(AudioChunk(start + second * SAMPLE_RATE, pcm, 0, generation=2))
        in_q.put(None)
        detector = Mock(side_effect=lambda audio: np.zeros(len(audio) // FRAME))
        with patch("auto_translate.vad.StreamingFireRedVAD", return_value=detector):
            run_segmenter(VadConfig(fallback_s=5.0), in_q, out_q, lambda: 2)
        self.assertEqual(detector.call_count, 12)
        covered = start
        probes = 0
        reached_lead_during_processing = False
        while (item := out_q.get_nowait()) is not None:
            self.assertEqual(item.generation, 2)
            if isinstance(item, Utterance):
                self.assertTrue(item.probe)
                self.assertGreaterEqual(round(item.start_s * SAMPLE_RATE), covered)
                probes += 1
            elif isinstance(item, AudioProgress):
                self.assertGreaterEqual(item.sample, covered)
                covered = item.sample
                if covered >= start + 5 * SAMPLE_RATE and covered < start + 12 * SAMPLE_RATE:
                    reached_lead_during_processing = True
        self.assertTrue(reached_lead_during_processing)
        self.assertGreater(probes, 1)
        self.assertEqual(covered, start + 12 * SAMPLE_RATE)

    def test_default_vad_skips_noise_probes_but_advances_progress(self):
        in_q, out_q = queue.Queue(), queue.Queue()
        start = 30 * SAMPLE_RATE
        pcm = np.ones(SAMPLE_RATE, dtype=np.float32)
        in_q.put(AudioChunk(start, pcm[:0], 0, reset=True, generation=3))
        for second in range(12):
            in_q.put(AudioChunk(start + second * SAMPLE_RATE, pcm, 0, generation=3))
        in_q.put(None)
        detector = Mock(side_effect=lambda audio: np.zeros(len(audio) // FRAME))
        with patch("auto_translate.vad.StreamingFireRedVAD", return_value=detector):
            run_segmenter(VadConfig(), in_q, out_q, lambda: 3)
        outputs = []
        while (item := out_q.get_nowait()) is not None:
            outputs.append(item)
            self.assertEqual(item.generation, 3)
        self.assertEqual(detector.call_count, 12)
        self.assertFalse(any(isinstance(item, Utterance) for item in outputs))
        progress = [item.sample for item in outputs if isinstance(item, AudioProgress)]
        self.assertTrue(progress)
        self.assertEqual(progress[-1], start + 12 * SAMPLE_RATE)

    def test_vod_seeks_refresh_url_then_try_aac_on_zero_pcm(self):
        ingest = Ingest(Source("https://example.org/video", live=False), IngestConfig())
        with patch.object(
            ingest, "_resolve_audio_url",
            side_effect=["first", "fresh", "aac-fresh"],
        ) as resolve:
            self.assertIn("first", ingest._audio_commands()[0])
            ingest.source = replace(ingest.source, start_s=30)
            calls = []

            def failed_session(out_q):
                calls.append(ingest._audio_commands()[0])
                return True

            with patch.object(ingest, "_read_session", side_effect=failed_session):
                with self.assertRaisesRegex(
                    RuntimeError, "after URL refresh and audio format fallback",
                ):
                    ingest.run(queue.Queue())
        self.assertEqual(resolve.call_count, 3)
        self.assertEqual(len(calls), 3)
        for cmd, url in zip(calls, ["first", "fresh", "aac-fresh"]):
            self.assertEqual(cmd[cmd.index("-i") + 1], url)
            self.assertEqual(cmd[cmd.index("-ss") + 1], "30.000")
        self.assertEqual(ingest._audio_format_override, "bestaudio[ext=m4a]/bestaudio/best")

    def test_first_vod_audio_open_failure_is_retried_twice_then_reported(self):
        ingest = Ingest(
            Source("https://www.youtube.com/watch?v=EXAMPLE", live=False,
                   duration=150.0),
            IngestConfig(),
        )
        with patch.object(ingest, "_read_session", return_value=True) as read:
            with self.assertRaisesRegex(
                RuntimeError, "audio format fallback",
            ):
                ingest.run(queue.Queue())
        self.assertEqual(read.call_count, 3)
        self.assertEqual(ingest._audio_format_override, "bestaudio[ext=m4a]/bestaudio/best")

    def test_first_vod_audio_refresh_can_succeed_without_fallback(self):
        ingest = Ingest(
            Source("https://www.youtube.com/watch?v=EXAMPLE", live=False),
            IngestConfig(),
        )
        with patch.object(ingest, "_read_session", side_effect=[True, False]) as read:
            ingest.run(queue.Queue())
        self.assertEqual(read.call_count, 2)
        self.assertIsNone(ingest._audio_format_override)

    def test_first_vod_audio_aac_fallback_can_succeed(self):
        ingest = Ingest(
            Source("https://www.youtube.com/watch?v=EXAMPLE", live=False),
            IngestConfig(),
        )
        with patch.object(
            ingest, "_read_session", side_effect=[True, True, False],
        ) as read:
            ingest.run(queue.Queue())
        self.assertEqual(read.call_count, 3)
        self.assertEqual(ingest._audio_format_override, "bestaudio[ext=m4a]/bestaudio/best")

    def test_vod_premature_audio_eof_resumes_until_real_end(self):
        ingest = Ingest(
            Source("https://www.youtube.com/watch?v=EXAMPLE", live=False,
                   duration=100.0),
            IngestConfig(),
        )
        starts = []
        lengths = iter([8.0, 12.0, 80.0])
        def fake_session(out_q):
            starts.append(ingest.source.start_s)
            ingest._last_session_audio_s = next(lengths)
            return False
        out = queue.Queue()
        with patch.object(ingest, "_read_session", side_effect=fake_session), patch(
            "auto_translate.ingest.time.sleep"
        ):
            ingest.run(out)
        self.assertEqual(starts, [0.0, 8.0, 20.0])
        self.assertIsNone(out.get_nowait())

    def test_repeated_vod_premature_eof_is_reported_not_silent(self):
        ingest = Ingest(
            Source("https://www.youtube.com/watch?v=EXAMPLE", live=False,
                   duration=100.0),
            IngestConfig(),
        )
        called = []
        def fake_session(out_q):
            called.append(ingest.source.start_s)
            ingest._last_session_audio_s = 2.0
            return False
        out = queue.Queue()
        with patch.object(ingest, "_read_session", side_effect=fake_session), patch(
            "auto_translate.ingest.time.sleep"
        ):
            with self.assertRaisesRegex(RuntimeError, "audio ended prematurely"):
                ingest.run(out)
        self.assertEqual(called, [0.0, 2.0, 4.0, 6.0])
        self.assertIsNone(out.get_nowait())



if __name__ == "__main__":
    unittest.main()
