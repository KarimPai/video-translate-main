"""Single recognizer boundaries: original VAD times and stale-response safety."""
import queue
import unittest
from types import SimpleNamespace

import numpy as np

from auto_translate.pipeline import run_asr, run_translate
from auto_translate.types import AudioProgress, Transcript, Utterance


class SingleRecognizerPipelineTests(unittest.TestCase):
    def recognize(self, text, *, generation=None, on_call=None, duration=.2, confidence=.95):
        samples = np.full(round(duration * 16000), .15, dtype=np.float32)
        utterance = Utterance(42.1, 42.1 + duration, samples,
                              generation=3, vad_confidence=confidence, speech_start_s=42.12)
        source, output = queue.Queue(), queue.Queue()
        source.put(utterance)
        source.put(None)
        calls = []
        def transcribe(audio, probe=False):
            calls.append((audio, probe))
            if on_call:
                on_call()
            return text
        run_asr(SimpleNamespace(transcribe=transcribe), source, output,
                get_generation=generation)
        result = []
        while not output.empty():
            item = output.get_nowait()
            if item is not None:
                result.append(item)
        return calls, result

    def test_short_japanese_reaction_keeps_vad_times_and_original_audio(self):
        calls, result = self.recognize('え？')
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].ja, 'え？')
        self.assertAlmostEqual(result[0].start_s, 42.1)
        self.assertAlmostEqual(result[0].end_s, 42.3)
        self.assertAlmostEqual(result[0].speech_start_s, 42.12)
        self.assertEqual(result[0].generation, 3)
        self.assertEqual(len(calls[0][0]), 3200)

    def test_seek_during_recognition_discards_old_response(self):
        generation = [3]
        calls, result = self.recognize('聞こえた', generation=lambda: generation[0],
                                       on_call=lambda: generation.__setitem__(0, 4))
        self.assertEqual(len(calls), 1)
        self.assertEqual(result, [])

    def test_foreign_transcript_has_no_japanese_subtitle(self):
        _, result = self.recognize('How are you doing today?', duration=3)
        self.assertEqual(result, [])

    def test_decoder_loop_rejected_without_second_recognizer(self):
        calls, result = self.recognize('あいうえお' * 8, duration=1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result, [])

    def test_empty_asr_never_fabricates_dialogue_from_caption(self):
        source, output = queue.Queue(), queue.Queue()
        source.put(Utterance(0, 1, np.ones(16000, dtype=np.float32)))
        source.put(None)
        captions = SimpleNamespace(match=lambda *_: SimpleNamespace(
            reliable=True, creator_uploaded=True, text='存在しない台詞'))
        run_asr(SimpleNamespace(transcribe=lambda *_: ''), source, output, captions=captions)
        self.assertIsNone(output.get_nowait())

    def test_generation_reset_clears_translation_context(self):
        source, output = queue.Queue(), queue.Queue()
        seen = []
        def translate(text, history):
            seen.append((text, list(history)))
            return 'Yes.'
        source.put(Transcript(0, 1, 'はい', generation=1))
        source.put(AudioProgress(160000, reset=True, generation=2))
        source.put(Transcript(10, 11, 'うん', generation=2))
        source.put(None)
        run_translate(SimpleNamespace(translate=translate), 4, source, output)
        self.assertEqual([history for _, history in seen], [[], []])


if __name__ == '__main__':
    unittest.main()
