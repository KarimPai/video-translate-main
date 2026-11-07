import unittest

from auto_translate.text_filter import rejection_reason


class TranscriptFilterTests(unittest.TestCase):
    def test_laughter_only_is_rejected(self):
        for text in ("ははは", "あはは", "ふふふ", "ｗｗｗ", "ハハハ", "［笑い声］", "(laughter)", "笑", "草"):
            with self.subTest(text=text):
                self.assertEqual(rejection_reason(text), "laughter_only")

    def test_speech_with_laughter_is_preserved(self):
        self.assertIsNone(rejection_reason("あはは、今日は本当に楽しかった"))
        self.assertIsNone(rejection_reason("それは笑った"))

    def test_obvious_repetition_is_rejected(self):
        self.assertEqual(rejection_reason("ああああああ"), "repeated_character")
        self.assertEqual(rejection_reason("あいあいあいあい"), "repeated_kana_loop")

    def test_short_reactions_and_normal_speech_are_preserved(self):
        for text in ("え", "うん", "そうそう", "はいはいはいはい", "ハハハ、敵来た", "笑い声が聞こえた", "今日は楽しかった", "ホロライブ"):
            with self.subTest(text=text):
                self.assertIsNone(rejection_reason(text))


if __name__ == "__main__":
    unittest.main()
