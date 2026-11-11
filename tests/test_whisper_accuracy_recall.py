"""Offline Whisper accuracy/recall checks without GPU or model downloads."""
import queue
import unittest
from unittest.mock import patch

import numpy as np

from auto_translate.config import VadConfig, WhisperConfig
from auto_translate.pipeline import run_whisper
from auto_translate.types import AudioChunk, Utterance
from auto_translate.vad import FRAME, Segmenter
from auto_translate.whisper_v3 import WhisperDecode


class AccuracyRecallTests(unittest.TestCase):
    def test_quiet_sustained_speech_is_detected_and_keeps_onset_padding(self):
        frames = 270
        probs = np.full(frames, .02, dtype=np.float32)
        probs[100:145] = .46  # below previous 0.50 gate; 450ms real speech
        pcm = np.full(frames * FRAME, .1, dtype=np.float32)

        class VAD:
            def reset(self):
                pass
            def __call__(self, audio):
                return probs

        with patch("auto_translate.vad.StreamingFireRedVAD", return_value=VAD()):
            vad = Segmenter(VadConfig())
            result = vad.feed(AudioChunk(0, pcm, 0.0)) + vad.flush()
        self.assertEqual(len(result), 1)
        self.assertGreaterEqual(result[0].vad_confidence, 0.43)
        self.assertLessEqual(result[0].start_s, .72)
        self.assertGreater(result[0].end_s, 1.45)
        self.assertFalse(result[0].probe)

    def test_tiny_noise_spike_still_is_not_speech(self):
        frames = 260
        probs = np.full(frames, .02, dtype=np.float32)
        probs[90:94] = .99  # 40ms gunshot-like probability transient
        pcm = np.full(frames * FRAME, .1, dtype=np.float32)

        class VAD:
            def reset(self):
                pass
            def __call__(self, audio):
                return probs

        with patch("auto_translate.vad.StreamingFireRedVAD", return_value=VAD()):
            vad = Segmenter(VadConfig())
            self.assertEqual(vad.feed(AudioChunk(0, pcm, 0.0)) + vad.flush(), [])

    @staticmethod
    def utterance(confidence=.95, duration_s=1.0):
        return Utterance(
            0.0, duration_s,
            np.full(int(duration_s * 16000), .1, dtype=np.float32),
            vad_confidence=confidence,
        )

    @staticmethod
    def run_case(responses, japanese="行かない", *,
                 confidence=.95, duration_s=1.0, cfg=None):
        class Fake:
            def __init__(self):
                self.cfg = cfg or WhisperConfig()
                self.responses = iter(responses)
                self.decode_calls = []
                self.ja_calls = 0

            def decode_audio(self, audio, *, prompt=None, hotwords=None, temperature=None):
                self.decode_calls.append((prompt, hotwords, temperature))
                return next(self.responses)

            def transcribe_japanese_evidence(self, audio):
                self.ja_calls += 1
                return japanese

        model = Fake()
        incoming, outgoing = queue.Queue(), queue.Queue()
        incoming.put(AccuracyRecallTests.utterance(confidence, duration_s))
        incoming.put(None)
        run_whisper(model, incoming, outgoing)
        subtitles = []
        while (item := outgoing.get_nowait()) is not None:
            subtitles.append(item)
        return model, subtitles

    def test_empty_strong_vad_japanese_evidence_rescues_good_translation(self):
        model, subs = self.run_case([
            WhisperDecode("", decoded_segments=0, no_speech_prob=None),
            WhisperDecode("I'm not going.", decoded_segments=1, avg_logprob=-.2,
                          no_speech_prob=.12, compression_ratio=1.1),
        ])
        self.assertEqual([s.en for s in subs], ["I'm not going."])
        self.assertEqual(len(model.decode_calls), 2)
        self.assertEqual(model.decode_calls[1], (None, None, .2))
        self.assertEqual(model.ja_calls, 1)

    def test_empty_zero_segment_has_no_silence_metric(self):
        # WhisperDecode created from an actual zero-segment Whisper result
        # has no no_speech_prob at all; the recovery must not be dead code.
        model, subs = self.run_case([
            WhisperDecode(""),
            WhisperDecode("Yes!", decoded_segments=1, avg_logprob=-.20,
                          no_speech_prob=.15, compression_ratio=1.1),
        ], japanese="はい")
        self.assertEqual([s.en for s in subs], ["Yes!"])
        self.assertEqual((len(model.decode_calls), model.ja_calls), (2, 1))

    def test_empty_without_japanese_evidence_is_not_rescued(self):
        model, subs = self.run_case(
            [WhisperDecode("", no_speech_prob=.15)], japanese="")
        self.assertEqual(subs, [])
        self.assertEqual(len(model.decode_calls), 1)
        self.assertEqual(model.ja_calls, 1)

    def test_low_vad_and_silence_do_not_trigger_empty_recovery(self):
        for confidence, silence in ((.70, .1), (.99, .9)):
            with self.subTest(confidence=confidence, silence=silence):
                model, subs = self.run_case(
                    [WhisperDecode("", no_speech_prob=silence)],
                    confidence=confidence)
                self.assertEqual(subs, [])
                self.assertEqual(len(model.decode_calls), 1)
                self.assertEqual(model.ja_calls, 0)

    def test_empty_retry_rejects_low_confidence_and_stock_hallucination(self):
        for candidate in (
            WhisperDecode("Thank you for watching.", decoded_segments=1,
                          avg_logprob=-.1, no_speech_prob=.1, compression_ratio=1.1),
            WhisperDecode("No!", decoded_segments=1,
                          avg_logprob=-.85, no_speech_prob=.03, compression_ratio=1.1),
        ):
            model, subs = self.run_case([
                WhisperDecode("", no_speech_prob=.1), candidate])
            self.assertEqual(subs, [])
            self.assertEqual(len(model.decode_calls), 2)
            self.assertEqual(model.ja_calls, 1)

    def test_rescue_and_neutrality_share_one_japanese_pass(self):
        model, subs = self.run_case([
            WhisperDecode("", no_speech_prob=.1),
            WhisperDecode("I don't know.", decoded_segments=1,
                          avg_logprob=-.2, no_speech_prob=.1, compression_ratio=1.1),
        ], japanese="分からない")
        self.assertEqual([s.en for s in subs], ["Not sure."])
        self.assertEqual(model.ja_calls, 1)

    def test_no_inference_when_disabled_or_audio_too_long(self):
        cfg = WhisperConfig(recover_empty_speech=False)
        model, subs = self.run_case(
            [WhisperDecode("", no_speech_prob=.1)], cfg=cfg)
        self.assertEqual(subs, [])
        self.assertEqual(model.ja_calls, 0)
        long_model, subs = self.run_case(
            [WhisperDecode("", no_speech_prob=.1)], duration_s=9.0)
        self.assertEqual(subs, [])
        self.assertEqual(long_model.ja_calls, 0)

    def test_defaults_preserve_noisy_audio_guards(self):
        cfg = WhisperConfig()
        self.assertEqual(cfg.beam_size, 6)
        self.assertEqual(cfg.max_new_tokens, 192)
        self.assertFalse(cfg.condition_on_previous_text)
        self.assertEqual(cfg.no_speech_threshold, .6)
        self.assertFalse(VadConfig().fallback_s)


if __name__ == "__main__":
    unittest.main()
