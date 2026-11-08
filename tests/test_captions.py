import queue
import unittest
from unittest.mock import Mock

import numpy as np

from auto_translate.captions import CaptionCue, CaptionIndex, PendingCaptions
from auto_translate.pipeline import run_asr, run_translate
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

    def test_pending_captions_become_available_during_playback(self):
        pending = PendingCaptions()
        self.assertIsNone(pending.match(1.2, 2.2))
        pending.index = self.index
        self.assertEqual(pending.match(1.2, 2.2), self.index.match(1.2, 2.2))

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


    def test_duplicate_overlap_and_sound_labels_do_not_verify_dialogue(self):
        index = CaptionIndex([
            CaptionCue(0, 1, "[拍手]"), CaptionCue(0, .35, "行ける"),
            CaptionCue(0, .35, "行ける"),
        ])
        match = index.match(0, 1)
        self.assertAlmostEqual(match.coverage, .35)
        self.assertFalse(match.reliable)
        self.assertFalse(CaptionIndex([CaptionCue(0, 8, "長い字幕")]).match(2, 3).reliable)

    def test_rolling_vtt_does_not_repeat_previous_line_after_seek(self):
        index = CaptionIndex.from_vtt(VTT.replace(
            "一緒に遊びましょう", "今日はゲームをします\n一緒に<00:00:03.000>遊びましょう",
        ))
        self.assertEqual(index.match(2.5, 4).text, "一緒に遊びましょう")


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

    def test_nonspeech_probes_do_not_decode_or_restore_captions(self):
        t = np.arange(16000) / 16000
        clicks = np.zeros(16000, dtype=np.float32)
        clicks[::800] = .8
        for audio in (clicks, np.sin(2 * np.pi * 440 * t).astype("f"), clicks * 0):
            incoming, outgoing = queue.Queue(), queue.Queue()
            incoming.put(Utterance(1.2, 2.2, audio, probe=True))
            incoming.put(None)
            asr = Mock()
            run_asr(asr, incoming, outgoing, captions=CaptionIndex.from_vtt(VTT))
            asr.transcribe.assert_not_called()
            self.assertIsNone(outgoing.get_nowait())

    def test_sound_cue_requires_independent_evidence_to_drop_haas(self):
        audio = np.random.default_rng(7).normal(0, .1, 16000).astype("f")
        for text, expected in [("ご視聴ありがとうございました", None), ("パチパチ", None),
                               ("この音楽が好き", "この音楽が好き")]:
            incoming, outgoing = queue.Queue(), queue.Queue()
            incoming.put(Utterance(1.2, 2.2, audio, probe=True))
            incoming.put(None)
            run_asr(TextASR(text), incoming, outgoing,
                    captions=CaptionIndex([CaptionCue(1.2, 2.2, "[音楽]")]))
            item = outgoing.get_nowait()
            self.assertEqual(item.ja if item else None, expected)

    def test_laughter_caption_does_not_restore_filtered_audio(self):
        index = CaptionIndex.from_vtt(VTT.replace("今日はゲームをします", "[笑い声]"))
        in_q, out_q = queue.Queue(), queue.Queue()
        in_q.put(Utterance(1.2, 2.2, np.zeros(16000, dtype=np.float32)))
        in_q.put(None)
        run_asr(EmptyASR(), in_q, out_q, captions=index)
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


class ReactionPipelineTests(unittest.TestCase):
    @staticmethod
    def _queues(*texts):
        class SequenceASR:
            def __init__(self, items):
                self.items = iter(items)

            def transcribe(self, audio, probe=False):
                return next(self.items)

        incoming, outgoing = queue.Queue(), queue.Queue()
        for index in range(len(texts)):
            incoming.put(Utterance(
                float(index + 1), float(index + 2),
                np.zeros(16000, dtype=np.float32),
            ))
        incoming.put(None)
        run_asr(SequenceASR(texts), incoming, outgoing)
        result = []
        while (item := outgoing.get_nowait()) is not None:
            result.append(item)
        return result

    def test_reactions_do_not_become_subtitles_or_translation_requests(self):
        class SpyTranslator:
            memory_lines = 0

            def __init__(self):
                self.calls = 0

            def translate(self, ja, history):
                self.calls += 1
                return "unexpected translation"

        incoming, transcripts, subtitles = queue.Queue(), queue.Queue(), queue.Queue()
        incoming.put(Utterance(1.0, 2.0, np.zeros(16000, dtype=np.float32)))
        incoming.put(None)
        run_asr(TextASR("うわー！"), incoming, transcripts)

        spy = SpyTranslator()
        run_translate(spy, 8, transcripts, subtitles)
        self.assertEqual(spy.calls, 0)
        self.assertIsNone(subtitles.get_nowait())

    def test_following_dialogue_survives_filtered_reaction(self):
        lines = self._queues("あっ！", "あっ、ペコちゃんのサインが")
        self.assertEqual([line.ja for line in lines], ["あっ、ペコちゃんのサインが"])

    def test_reliable_caption_fallback_cannot_restore_isolated_reaction(self):
        index = CaptionIndex.from_vtt(VTT.replace("今日はゲームをします", "うわー！"))
        incoming, outgoing = queue.Queue(), queue.Queue()
        incoming.put(Utterance(1.2, 2.2, np.zeros(16000, dtype=np.float32)))
        incoming.put(None)
        run_asr(EmptyASR(), incoming, outgoing, captions=index)
        self.assertIsNone(outgoing.get_nowait())

    def test_filter_happens_after_caption_reconciliation(self):
        from unittest.mock import patch

        index = CaptionIndex.from_vtt(VTT.replace("今日はゲームをします", "わため"))
        incoming, outgoing = queue.Queue(), queue.Queue()
        incoming.put(Utterance(1.2, 2.2, np.zeros(16000, dtype=np.float32)))
        incoming.put(None)
        with patch("auto_translate.pipeline.reconcile_caption", return_value="うわー！") as reconcile:
            run_asr(TextASR("渡辺"), incoming, outgoing, captions=index)
        reconcile.assert_called_once_with("渡辺", "わため")
        self.assertIsNone(outgoing.get_nowait())

    def test_non_reaction_caption_fallback_still_works(self):
        index = CaptionIndex.from_vtt(VTT.replace("今日はゲームをします", "うわー！これすごい！"))
        incoming, outgoing = queue.Queue(), queue.Queue()
        incoming.put(Utterance(1.2, 2.2, np.zeros(16000, dtype=np.float32)))
        incoming.put(None)
        run_asr(EmptyASR(), incoming, outgoing, captions=index)
        self.assertEqual(outgoing.get_nowait().ja, "うわー！これすごい！")
        self.assertIsNone(outgoing.get_nowait())


if __name__ == "__main__":
    unittest.main()
