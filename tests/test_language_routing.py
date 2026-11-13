"""Japanese-only Whisper Turbo does not attempt English language routing."""
import unittest
from unittest.mock import Mock
from auto_translate.asr.dual import DualASR
from auto_translate.asr.whisper_turbo import WhisperTurboASR
from auto_translate.config import Config, load_config
from auto_translate.glossary import Glossary
from auto_translate.types import Utterance
import numpy as np


class JapaneseOnlyRoutingTests(unittest.TestCase):
    def test_default_disables_language_routing(self):
        self.assertFalse(Config().dual_asr.language_routing)

    def test_whisper_turbo_is_transcription_only(self):
        self.assertFalse(hasattr(WhisperTurboASR, "transcribe_with_language"))

    def test_no_preflight_even_if_legacy_flag_enabled(self):
        cfg = Config()
        cfg.dual_asr.enabled = False
        dual = DualASR(cfg, Mock(), Glossary({}), False)
        dual.worker = Mock()
        cfg.dual_asr.language_routing = True
        utt = Utterance(2, 3, np.ones(16000, dtype=np.float32), vad_confidence=.95)
        self.assertIsNone(dual.preflight_language(utt))
        dual.worker.transcribe_with_language.assert_not_called()


if __name__ == "__main__":
    unittest.main()
