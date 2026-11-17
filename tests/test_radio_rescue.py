"""Faint, band-limited game-radio speech must survive filtering and get one rescue pass."""
import queue
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from auto_translate.asr.voxtral import _pad_for_decoder, radio_enhance
from auto_translate.config import AsrConfig, LanguageFilterConfig
from auto_translate.language_filter import LanguageFilter, radio_band_limited
from auto_translate.pipeline import run_asr
from auto_translate.types import SAMPLE_RATE, Utterance


def radio_clip(seconds=4.0, level=.05):
    t = np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    voice = sum(np.sin(2 * np.pi * f * t) for f in (400, 900, 1500, 2600))
    return (level * voice / 4).astype(np.float32)


def wideband_clip(seconds=4.0):
    rng = np.random.default_rng(7)
    return (.05 * rng.standard_normal(round(seconds * SAMPLE_RATE))).astype(np.float32)


def make_filter(result):
    filt = object.__new__(LanguageFilter)
    filt.cfg = LanguageFilterConfig()
    filt.retry_at = 0.0
    filt._predict = lambda clip: result
    return filt


class RadioLanguageFilterTests(unittest.TestCase):
    def test_band_limited_detection(self):
        self.assertTrue(radio_band_limited(radio_clip()))
        self.assertFalse(radio_band_limited(wideband_clip()))
        self.assertFalse(radio_band_limited(np.zeros(SAMPLE_RATE, np.float32)))

    def test_radio_misread_as_other_language_is_preserved(self):
        for language in ("ko", "zh", "vi"):
            with self.subTest(language=language):
                self.assertFalse(make_filter((language, .97, .01, .91)).classify(radio_clip()).drop)

    def test_radio_english_or_indonesian_is_still_dropped(self):
        for language in ("en", "id"):
            with self.subTest(language=language):
                self.assertTrue(make_filter((language, .97, .01, .91)).classify(radio_clip()).drop)

    def test_wideband_foreign_language_is_still_dropped(self):
        self.assertTrue(make_filter(("ko", .97, .01, .91)).classify(wideband_clip()).drop)


class RadioEnhanceTests(unittest.TestCase):
    def test_faint_radio_is_lifted_up_to_24_db_without_clipping(self):
        quiet = radio_clip(level=.002)
        enhanced = radio_enhance(quiet)
        self.assertEqual(enhanced.shape, quiet.shape)
        gain = np.sqrt(np.mean(enhanced ** 2)) / np.sqrt(np.mean(quiet ** 2))
        self.assertGreater(gain, 8.0)
        self.assertLessEqual(float(np.max(np.abs(enhanced))), .95 + 1e-6)

    def test_rumble_is_removed(self):
        t = np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE
        rumble = (.2 * np.sin(2 * np.pi * 30 * t)).astype(np.float32)
        mixed = rumble + radio_clip(2.0, .05)
        spectrum = np.abs(np.fft.rfft(radio_enhance(mixed)))
        freqs = np.fft.rfftfreq(mixed.size, 1 / SAMPLE_RATE)
        self.assertLess(spectrum[np.argmin(abs(freqs - 30))], .02 * spectrum[np.argmin(abs(freqs - 900))])

    def test_silence_stays_silent_and_padding_brackets_speech(self):
        self.assertFalse(np.any(radio_enhance(np.zeros(8000, np.float32))))
        padded = _pad_for_decoder(np.ones(1600, np.float32))
        self.assertEqual(padded.size, 16000 + 1600 + 9600)
        self.assertFalse(np.any(padded[:16000]))


class RescuePipelineTests(unittest.TestCase):
    def run_once(self, first, rescued, *, confidence=.8, duration=2.0, cfg=None):
        asr = SimpleNamespace(last_rejection=None, last_raw_transcript="", calls=[])

        def transcribe(audio, probe=False):
            asr.calls.append("first")
            asr.last_rejection, asr.last_raw_transcript = "not_japanese", "Copy that."
            return first

        def transcribe_rescue(audio):
            asr.calls.append("rescue")
            asr.last_rejection, asr.last_raw_transcript = None, rescued
            return rescued

        asr.transcribe, asr.transcribe_rescue = transcribe, transcribe_rescue
        source, output = queue.Queue(), queue.Queue()
        source.put(Utterance(10., 10. + duration, radio_clip(duration), vad_confidence=confidence))
        source.put(None)
        events = []
        with patch("auto_translate.pipeline.record_event",
                   side_effect=lambda name, **data: events.append((name, data))):
            run_asr(asr, source, output, asr_cfg=cfg)
        results = [item for item in output.queue if item is not None]
        return asr.calls, results, events

    def test_rejected_radio_line_is_rescued(self):
        calls, results, events = self.run_once("", "こちら本署、応答願います")
        self.assertEqual(calls, ["first", "rescue"])
        self.assertEqual([r.ja for r in results], ["こちら本署、応答願います"])
        self.assertIn("asr_rescued", [name for name, _ in events])

    def test_failed_rescue_records_rejection_with_raw_text(self):
        calls, results, events = self.run_once("", "")
        self.assertEqual(calls, ["first", "rescue"])
        self.assertEqual(results, [])
        rejected = [data for name, data in events if name == "asr_rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["reason"], "not_japanese")
        self.assertEqual(rejected[0]["raw_text"], "Copy that.")

    def test_low_confidence_noise_is_not_retried(self):
        calls, results, events = self.run_once("", "こちら本署", confidence=.2)
        self.assertEqual(calls, ["first"])
        self.assertEqual(results, [])
        self.assertIn("asr_rejected", [name for name, _ in events])

    def test_rescue_can_be_disabled(self):
        calls, _, _ = self.run_once("", "こちら本署", cfg=AsrConfig(rescue_retry=False))
        self.assertEqual(calls, ["first"])

    def test_accepted_first_pass_never_retries(self):
        calls, results, _ = self.run_once("了解です", "別の文")
        self.assertEqual(calls, ["first"])
        self.assertEqual([r.ja for r in results], ["了解です"])


if __name__ == "__main__":
    unittest.main()
