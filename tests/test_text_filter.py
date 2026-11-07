import unittest

from auto_translate.text_filter import (
    is_isolated_reaction, is_whole_utterance_blocked, rejection_reason,
)


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


class IsolatedReactionTests(unittest.TestCase):
    def test_aggressive_isolated_reactions_are_filtered(self):
        for text in (
            "あっ！", "はっ。", "はあ", "うわっ。", "うわー！うわー！",
            "うう！", "よいしょ", "よいしょよいしょ", "えーっと",
            "うん。", "はい。", "えっ？", "ん?", "すすげえ",
            "すごい！", "スゴイ！", "すげえ", "あのー", "うーん",
            "へぇ～", "はーい", "Wow!", "Hey!", "Whoa!", "おお!",
            "うわぁぁぁ！", "あっ!あっ!",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_isolated_reaction(text))

    def test_embedded_reactions_and_dialogue_are_preserved(self):
        for text in (
            "あっ、ペコちゃんのサインが", "うわー！これすごい！",
            "すごいねかっこいい", "あの人は誰？",
            "よいしょ押す", "よいしょよいしょ見せたくて",
            "えっ、何これ？", "待って。", "ありがとう",
            "はい、分かりました", "うん、そうだね", "それはすごい",
            "なんで？", "こんにちは。", "えっ！あれは何？",
            "えーっと、ここかな", "いや", "すすげえタイミング！",
            "ヘイ、みんな！", "Wow, this is huge", "くっつく", "ワタメ",
        ):
            with self.subTest(text=text):
                self.assertFalse(is_isolated_reaction(text))

    def test_custom_blocklist_requires_whole_utterance_match(self):
        self.assertTrue(is_whole_utterance_blocked("あっ！", ["あっ"]))
        self.assertTrue(is_whole_utterance_blocked("ご視聴ありがとうございました！", ["ご視聴ありがとうございました"]))
        self.assertFalse(is_whole_utterance_blocked("あっ、ペコちゃんのサインが", ["あっ"]))
        self.assertFalse(is_whole_utterance_blocked("動画の最後にご視聴ありがとうございましたと言った", ["ご視聴ありがとうございました"]))
        self.assertFalse(is_whole_utterance_blocked("あっ！", [""]))


if __name__ == "__main__":
    unittest.main()
