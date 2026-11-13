import unittest
from auto_translate.asr.quality_guard import transcript_problem, disagreement_unresolved
from auto_translate.config import DualAsrConfig

class AntiGibberishTests(unittest.TestCase):
    def test_normal_japanese_and_short_reactions_survive(self):
        for text in ("待って", "ええ", "30秒待って", "Watame、待って！"):
            self.assertIsNone(transcript_problem(text, 2.0))
    def test_loops_and_phantom_outro_are_rejected(self):
        self.assertIsNotNone(transcript_problem("ああああああああ", 2.0))
        self.assertEqual(transcript_problem("Thank you for watching.", 2.0), "phantom_outro")
    def test_unresolved_strong_disagreement_abstains(self):
        self.assertTrue(disagreement_unresolved("行かない", "行く", "行かない", .2, ["negation"], "uncertain"))
        self.assertFalse(disagreement_unresolved("はい", "はい", "はい", 1.0, [], "haas"))
    def test_legacy_config_no_third_asr(self):
        self.assertFalse(hasattr(DualAsrConfig(), "indonesian_routing"))

if __name__ == "__main__":
    unittest.main()
