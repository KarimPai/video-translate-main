import unittest
from auto_translate.asr.quality_guard import transcript_problem

class AntiGibberishTests(unittest.TestCase):
    def test_normal_japanese_and_short_reactions_survive(self):
        for text in ("待って", "ええ", "30秒待って", "Watame、待って！"):
            self.assertIsNone(transcript_problem(text, 2.0))
    def test_loops_and_phantom_outro_are_rejected(self):
        self.assertIsNotNone(transcript_problem("ああああああああ", 2.0))
        self.assertEqual(transcript_problem("Thank you for watching.", 2.0), "phantom_outro")

if __name__ == "__main__":
    unittest.main()
