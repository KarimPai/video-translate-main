"""Seek regressions for progress, audio URL reuse, and overlapping captions."""
import queue
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import numpy as np

from auto_translate.captions import CaptionCue, CaptionIndex
from auto_translate.config import IngestConfig, VadConfig
from auto_translate.ingest import Ingest, Source
from auto_translate.types import SAMPLE_RATE, AudioChunk, AudioProgress, Utterance
from auto_translate.vad import FRAME, run_segmenter


class SeekProcessingTests(unittest.TestCase):
    def test_only_current_pcm_is_processed_and_probes_precede_progress(self):
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
            run_segmenter(VadConfig(), in_q, out_q, lambda: 2)
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

    def test_vod_seeks_reuse_url_and_refresh_once_on_open_failure(self):
        ingest = Ingest(Source("https://example.org/video", live=False), IngestConfig())
        with patch.object(ingest, "_resolve_audio_url", side_effect=["first", "fresh"]) as resolve:
            self.assertIn("first", ingest._audio_commands()[0])
            ingest.source = replace(ingest.source, start_s=30)
            calls = []

            def failed_session(out_q):
                calls.append(ingest._audio_commands()[0])
                return True

            with patch.object(ingest, "_read_session", side_effect=failed_session):
                ingest.run(queue.Queue())
        self.assertEqual(resolve.call_count, 2)
        self.assertEqual(len(calls), 2)
        for cmd, url in zip(calls, ["first", "fresh"]):
            self.assertEqual(cmd[cmd.index("-i") + 1], url)
            self.assertEqual(cmd[cmd.index("-ss") + 1], "30.000")

    def test_late_match_includes_long_overlapping_cue(self):
        index = CaptionIndex([
            CaptionCue(1, 100, "A"), CaptionCue(2, 3, "B"),
            CaptionCue(80, 82, "C"), CaptionCue(80, 83, "C"),
        ])
        self.assertEqual(index.match(81, 82).text, "A C")
        self.assertIsNone(index.match(101, 102))
