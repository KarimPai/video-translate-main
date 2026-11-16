"""Name evidence reaches ASR selection without rewriting or surviving seeks."""
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np

from auto_translate.asr.dual import DualASR
from auto_translate.config import Config
from auto_translate.glossary import Glossary
from auto_translate.types import Utterance


class AsrNameContextTests(unittest.TestCase):
    def setUp(self):
        cfg = Config()
        cfg.dual_asr.enabled = False
        glossary = Glossary.load(Path(__file__).resolve().parents[1] / "glossary.toml")
        self.dual = DualASR(cfg, Mock(pressure_level="normal"), glossary, False)
        self.dual.worker = Mock(last_stop="complete")
        self.dual.selector = Mock(last_choice="haas")
        self.utt = Utterance(0, 4, np.full(64000, .1, dtype=np.float32),
                             vad_confidence=.95)

    def choose(self, primary, secondary, title="", current=None):
        self.dual.worker.transcribe.return_value = secondary
        self.dual.selector.choose.return_value = primary
        with patch("auto_translate.asr.dual.append_comparison"):
            return self.dual.choose(self.utt, primary, title, lambda: False,
                                    current or (lambda: self.utt.generation))

    def evidence(self):
        return self.dual.selector.choose.call_args.args[2]

    def test_title_and_aliases_reach_selector_without_rewriting_speech(self):
        for primary in ("みこはどうした", "ミコはどうした"):
            with self.subTest(primary=primary):
                self.dual.selector.choose.reset_mock()
                self.assertEqual(self.choose(primary, "巫女はどうした", "ホロGTA"), primary)
                self.assertIn((primary[:2], "Miko"), self.evidence()["glossary"])
                self.assertEqual(self.dual.selector.choose.call_args.args[:2],
                                 (primary, "巫女はどうした"))
                self.assertNotIn("initial_prompt", self.dual.worker.transcribe.call_args.kwargs)

    def test_recent_confirmed_name_is_available_without_hololive_title(self):
        self.dual.remember(self.utt.generation, "さくらみこだよ")
        self.dual.remember(self.utt.generation, "ちょっと待って")
        self.assertEqual(self.choose("みこが来た", "巫女が来た", "Minecraft"), "みこが来た")
        self.assertIn(("みこ", "Miko"), self.evidence()["glossary"])
        self.assertEqual(self.evidence()["recent_ja"], ["さくらみこだよ", "ちょっと待って"])

    def test_literal_role_is_not_promoted_to_miko_even_with_identity_context(self):
        self.dual.remember(self.utt.generation, "さくらみこだよ")
        primary = "神社の巫女さんが来た"
        self.assertEqual(self.choose(primary, primary + "よ", "さくらみこ Minecraft"), primary)
        self.assertFalse(any(target.startswith("Miko") or target == "Sakura Miko"
                             for _, target in self.evidence()["glossary"]))

    def test_seek_drops_prior_name_evidence_before_next_cue_is_remembered(self):
        self.dual.remember(0, "さくらみこだよ")
        self.utt.generation = 1
        self.assertEqual(self.choose("みこが来た", "巫女が来た", "Minecraft"), "みこが来た")
        self.assertEqual(self.evidence()["recent_ja"], [])
        self.assertNotIn(("みこ", "Miko"), self.evidence()["glossary"])
        self.dual.remember(1, "次の話をしよう")
        self.assertEqual(list(self.dual.history), ["次の話をしよう"])

    def test_old_name_evidence_expires_with_bounded_history(self):
        self.dual.remember(self.utt.generation, "さくらみこだよ")
        for _ in range(4):
            self.dual.remember(self.utt.generation, "次の話をしよう")
        self.choose("みこが来た", "巫女が来た", "Minecraft")
        self.assertNotIn(("みこ", "Miko"), self.evidence()["glossary"])

    def test_seek_during_decode_never_sends_stale_name_context_to_selector(self):
        self.dual.remember(0, "さくらみこだよ")
        self.assertEqual(self.choose("みこが来た", "巫女が来た", current=lambda: 1), "")
        self.dual.selector.choose.assert_not_called()

    def test_one_sided_homophone_cannot_override_weak_acoustic_evidence(self):
        self.utt.vad_confidence = .55
        self.dual.selector.last_choice = "uncertain"
        # Isolate the weak-phonetics gate: a dynamically resolved homophone
        # must obey the same two-recognizer corroboration rule as a nickname.
        with patch("auto_translate.asr.dual.compare", return_value=(.2, ["glossary"])), patch(
                "auto_translate.asr.dual.suspicious_phonetics", return_value=True), patch(
                "auto_translate.asr.dual.disagreement_unresolved") as unresolved:
            self.assertEqual(self.choose("巫女が来た", "誰か来た", "さくらみこ Minecraft"), "")
        self.assertIn(("巫女", "Miko"), self.evidence()["glossary"])
        unresolved.assert_not_called()


if __name__ == "__main__":
    unittest.main()
