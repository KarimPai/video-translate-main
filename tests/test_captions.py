import queue
import unittest

import numpy as np

from auto_translate.captions import CaptionIndex
from auto_translate.pipeline import run_asr
from auto_translate.types import Transcript, Utterance


VTT = """WEBVTT

00:00:01.000 --> 00:00:02.500
今日はゲームをします

00:00:02.500 --> 00:00:04.000
一緒に遊びましょう
"""


class CaptionIndexTests(unittest.TestCase):
    def setUp(self):
        self.index = CaptionIndex.from_vtt(VTT)

    def test_parses_japanese_cues(self):
        self.assertEqual(len(self.index.cues), 2)
        self.assertEqual(self.index.cues[0].text, "今日はゲームをします")

    def test_matches_using_vod_timestamps(self):
        match = self.index.match(1.2, 2.2)
        self.assertIsNotNone(match)
        self.assertEqual(match.text, "今日はゲームをします")
        self.assertTrue(match.reliable)

    def test_unrelated_time_does_not_match(self):
        self.assertIsNone(self.index.match(10.0, 11.0))

    def test_weak_overlap_is_not_reliable(self):
        match = self.index.match(0.0, 1.1)
        self.assertIsNotNone(match)
        self.assertFalse(match.reliable)


class EmptyASR:
    def transcribe(self, audio, probe=False):
        return ""


class TextASR:
    def __init__(self, text):
        self.text = text

    def transcribe(self, audio, probe=False):
        return self.text


class CaptionFallbackTests(unittest.TestCase):
    def test_aligned_caption_fills_empty_haas_result(self):
        index = CaptionIndex.from_vtt(VTT)
        in_q = queue.Queue()
        out_q = queue.Queue()
        in_q.put(Utterance(1.2, 2.2, np.zeros(16000, dtype=np.float32)))
        in_q.put(None)

        run_asr(EmptyASR(), in_q, out_q, captions=index)

        result = out_q.get_nowait()
        self.assertIsInstance(result, Transcript)
        self.assertEqual(result.ja, "今日はゲームをします")
        self.assertIsNone(out_q.get_nowait())

    def test_haas_remains_primary_when_both_sources_disagree(self):
        index = CaptionIndex.from_vtt(VTT)
        in_q = queue.Queue()
        out_q = queue.Queue()
        in_q.put(Utterance(1.2, 2.2, np.zeros(16000, dtype=np.float32)))
        in_q.put(None)

        run_asr(
            TextASR("ハースの認識結果"),
            in_q,
            out_q,
            captions=index,
        )

        result = out_q.get_nowait()
        self.assertEqual(result.ja, "ハースの認識結果")
        self.assertIsNone(out_q.get_nowait())


if __name__ == "__main__":
    unittest.main()
