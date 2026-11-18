"""Offline dual-ASR tests: no device, model download, or speech.cpp required."""
import unittest
from unittest.mock import patch

import numpy as np

from auto_translate.asr.dual import DualASR, canonical, plausible, select_candidate
from auto_translate.config import AsrConfig


STRONG = {"segments": [{"text": "はい", "avg_logprob": -0.15}],
          "avg_logprob": -0.15, "no_speech_prob": 0.05, "compression_ratio": 1.1}
WEAK = {"segments": [{"text": "はい", "avg_logprob": -1.4}],
        "avg_logprob": -1.4, "no_speech_prob": 0.7, "compression_ratio": 1.1}


class CandidateSelectionTests(unittest.TestCase):
    def test_canonical_preserves_spoken_nicknames(self):
        self.assertEqual(canonical("ラプ 様！"), canonical("ラプ様"))
        self.assertNotEqual(canonical("ラプ"), canonical("ラプラス"))

    def test_rejects_phantom_outro_and_english_prompt(self):
        self.assertFalse(plausible("ご視聴ありがとうございました", 0.4, []))
        self.assertFalse(plausible("bank robberies store robberies", 3.5, []))
        self.assertTrue(plausible("やばい！", 0.8, []))
        self.assertTrue(plausible("え？", 0.3, []))
        self.assertFalse(plausible("禁止です", 1.2, ["禁止"]))

    def test_agree_and_fallback(self):
        self.assertEqual(select_candidate("はい", "はい！", STRONG, 1)[1], "agreement")
        self.assertEqual(select_candidate("", "はい！", STRONG, 1)[0], "はい！")
        self.assertEqual(select_candidate("", "はい！", WEAK, 1, probe=True)[0], "")

    def test_disagreement_does_not_concatenate(self):
        selected, reason = select_candidate("ラプが来た", "マリンは帰った", WEAK, 2)
        self.assertEqual(selected, "ラプが来た")
        self.assertIn("disagreement", reason)

    def test_close_agreement_with_whisper_evidence(self):
        selected, reason = select_candidate("今日はよろしくね", "今日はよろしくね！", STRONG, 2)
        self.assertEqual(selected, "今日はよろしくね")
        self.assertEqual(reason, "agreement")


class DualASRAdapterTests(unittest.TestCase):
    def fake_workers(self, primary="はい", secondary="はい"):
        p = patch("auto_translate.asr.dual.RecoveringHaas")
        w = patch("auto_translate.asr.dual.WhisperTurboASR")
        primary_cls = p.start()
        secondary_cls = w.start()
        self.addCleanup(p.stop)
        self.addCleanup(w.stop)
        primary_cls.return_value.transcribe.return_value = primary
        secondary_cls.return_value.transcribe.return_value = secondary
        secondary_cls.return_value.last_confidence = STRONG
        return primary_cls.return_value, secondary_cls.return_value

    def test_vod_checks_both_decoders(self):
        primary, secondary = self.fake_workers()
        asr = DualASR(AsrConfig(), live=False)
        self.addCleanup(asr.close)
        self.assertEqual(asr.transcribe(np.zeros(16000, np.float32)), "はい")
        primary.transcribe.assert_called_once()
        secondary.transcribe.assert_called_once()
        self.assertEqual(asr.last_language, "Japanese")

    def test_live_selective_skips_long_plausible_haas(self):
        primary, secondary = self.fake_workers(primary="今日は本当によろしくお願いします")
        asr = DualASR(AsrConfig(), live=True)
        self.addCleanup(asr.close)
        self.assertEqual(asr.transcribe(np.zeros(16000 * 3, np.float32)),
                         "今日は本当によろしくお願いします")
        secondary.transcribe.assert_not_called()

    def test_whisper_startup_failure_degrades_to_haas(self):
        self.fake_workers()
        with patch("auto_translate.asr.dual.WhisperTurboASR",
                   side_effect=RuntimeError("CUDA unavailable")):
            asr = DualASR(AsrConfig())
        self.addCleanup(asr.close)
        self.assertIn("Haas only", asr.status)
        self.assertEqual(asr.transcribe(np.zeros(16000, np.float32)), "はい")

    def test_generation_cancellation_discards_stale_asr(self):
        primary, secondary = self.fake_workers()
        asr = DualASR(AsrConfig(), live=False)
        self.addCleanup(asr.close)
        def invalidate(*args, **kwargs):
            asr.cancel_pending()
            return "はい"
        primary.transcribe.side_effect = invalidate
        self.assertEqual(asr.transcribe(np.zeros(16000, np.float32)), "")
        secondary.transcribe.assert_not_called()

    def test_secondary_timeout_retires_worker_and_keeps_primary(self):
        primary, secondary = self.fake_workers(primary="今すぐ行く")
        secondary.transcribe.side_effect = TimeoutError("deadline")
        asr = DualASR(AsrConfig())
        self.addCleanup(asr.close)
        self.assertEqual(asr.transcribe(np.zeros(16000 * 2, np.float32)), "今すぐ行く")
        secondary.close.assert_called_once()
        self.assertIn("Haas only", asr.status)


if __name__ == "__main__":
    unittest.main()
