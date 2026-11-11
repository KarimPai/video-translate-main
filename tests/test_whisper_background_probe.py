"""VOD-only background speech probes; mocked VAD and Whisper, no GPU needed."""
import queue
import unittest
from unittest.mock import patch

import numpy as np

from auto_translate.config import VadConfig, WhisperConfig, vad_for_source
from auto_translate.pipeline import run_whisper
from auto_translate.types import AudioChunk, Utterance
from auto_translate.vad import FRAME, Segmenter
from auto_translate.whisper_v3 import WhisperDecode


class BackgroundProbeTests(unittest.TestCase):
    @staticmethod
    def segment(probs, pcm=None):
        if pcm is None:
            pcm = np.full(len(probs) * FRAME, .08, dtype=np.float32)
        class VAD:
            def reset(self):
                pass
            def __call__(self, audio):
                return probs
        with patch("auto_translate.vad.StreamingFireRedVAD", return_value=VAD()):
            profile = vad_for_source(VadConfig(), live=False)
            s = Segmenter(profile)
            emitted = s.feed(AudioChunk(0, pcm, 0.0)) + s.flush()
        return [u for u in emitted if u.probe]

    def test_sustained_subthreshold_background_audio_emits_probe(self):
        p = np.full(600, .03, dtype=np.float32)
        p[60:250] = .27  # below main threshold .43, 1.9s speech-like
        probes = self.segment(p)
        self.assertTrue(probes)
        self.assertTrue(all(u.vad_confidence >= .20 for u in probes))
        self.assertTrue(all(u.start_s < u.end_s for u in probes))

    def test_spike_and_music_like_silence_do_not_emit_probe(self):
        p = np.full(600, .02, dtype=np.float32)
        self.assertFalse(self.segment(p))
        p[50:55] = .95  # transient gunshot; main onset is not reached
        self.assertFalse(self.segment(p))

    @staticmethod
    def run_case(
        english, japanese="急いでください", *,
        confidence=.28, raw=None, cfg=None,
    ):
        class Fake:
            def __init__(self):
                self.cfg = cfg or WhisperConfig()
                self.ja_calls = 0
                self.en_calls = 0
                self.responses = iter(english)
            def transcribe_japanese_evidence(self, audio):
                self.ja_calls += 1
                return japanese
            def decode_audio(self, audio, *, prompt=None, hotwords=None,
                             temperature=None):
                self.en_calls += 1
                return next(self.responses)
        fake = Fake()
        incoming, outgoing = queue.Queue(), queue.Queue()
        audio = (np.full(32000, .05, dtype=np.float32) if raw is None else raw)
        incoming.put(Utterance(0.0, len(audio)/16000, audio,
                               probe=True, vad_confidence=confidence))
        incoming.put(None)
        run_whisper(fake, incoming, outgoing)
        found = []
        while (item := outgoing.get_nowait()) is not None:
            found.append(item)
        return fake, found

    @staticmethod
    def decoded(text, lp=-.15, ns=.08):
        return WhisperDecode(text, avg_logprob=lp,
                             no_speech_prob=ns, compression_ratio=1.1)

    def test_jp_plus_two_matching_en_passes_recovers_background(self):
        english = [self.decoded("Hurry up!"), self.decoded("Hurry up.")]
        fake, found = self.run_case(english)
        self.assertEqual([item.en for item in found], ["Hurry up!"])
        self.assertEqual((fake.ja_calls, fake.en_calls), (1, 2))

    def test_disagreement_blocks_phantom(self):
        english = [self.decoded("Hurry up!"), self.decoded("Look out!")]
        fake, found = self.run_case(english)
        self.assertEqual(found, [])
        self.assertEqual((fake.ja_calls, fake.en_calls), (1, 2))

    def test_no_japanese_evidence_skips_en_decode(self):
        fake, found = self.run_case([], japanese="")
        self.assertEqual(found, [])
        self.assertEqual(fake.en_calls, 0)

    def test_low_confidence_or_vad_zero_skips_probe(self):
        for conf in (0.0, .1):
            fake, found = self.run_case([], confidence=conf)
            self.assertEqual(found, [])
            self.assertEqual(fake.ja_calls, 0)
        fake, found = self.run_case([
            self.decoded("Hurry up!", lp=-.9),
            self.decoded("Hurry up!"),
        ])
        self.assertEqual(found, [])

    def test_normal_main_speech_and_live_profile_stay_unchanged(self):
        self.assertEqual(vad_for_source(VadConfig(), live=True).fallback_s, 0.0)
        self.assertEqual(VadConfig().fallback_s, 0.0)
        fake, found = self.run_case([], cfg=WhisperConfig(background_probe_enabled=False))
        self.assertEqual(found, [])
        self.assertEqual(fake.ja_calls, 0)

    def test_no_generic_stock_outro_and_short_cues(self):
        fake, found = self.run_case([
            self.decoded("Thank you for watching."),
            self.decoded("Thank you for watching."),
        ])
        self.assertEqual(found, [])
        short, found2 = self.run_case([
            self.decoded("Yes!"), self.decoded("Yes!"),
        ])
        self.assertEqual(found2, [])
        self.assertEqual(short.en_calls, 2)


if __name__ == "__main__":
    unittest.main()
