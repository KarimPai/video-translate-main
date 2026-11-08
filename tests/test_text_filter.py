import unittest

from auto_translate.text_filter import (
    is_isolated_reaction, is_whole_utterance_blocked, rejection_reason,
    strip_incidental_laughter,
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


class InlineLaughterTests(unittest.TestCase):
    def test_removes_laughter_without_dropping_spoken_words(self):
        samples = {
            "何それわかんないアハハ": "何それわかんない",
            "これでいこうと思いますあはは": "これでいこうと思います",
            "アハハ、今日は本当に楽しかった": "今日は本当に楽しかった",
            "ハハハ、敵来た": "敵来た",
            "行くよ、ふふふ、待って": "行くよ、待って",
            "これｗｗｗ！": "これ！",
            "あはは、何それ？": "何それ？",
            "何それ？アハハ": "何それ？",
            "これ(笑)違うよ": "これ 違うよ",
            "それ（笑）": "それ",
            "それ（草）": "それ",
        }
        for text, expected in samples.items():
            with self.subTest(text=text):
                from auto_translate.text_filter import strip_nonverbal_labels
                self.assertEqual(
                    strip_incidental_laughter(strip_nonverbal_labels(text)),
                    expected,
                )

    def test_laughter_only_goes_empty(self):
        for text in ("アハハ", "ははは", "ふふふ", "www", "笑笑", "（草）"):
            with self.subTest(text=text):
                self.assertEqual(strip_incidental_laughter(text), "")

    def test_preserve_legitimate_japanese_and_repeated_commands(self):
        for text in (
            "今日は笑った", "笑顔が好き", "草原の草",
            "来い来い来い", "待って待って", "うん、そうだね",
            "はは", "ふふ", "母は元気", "はははと笑った", "アハハと言った",
            "https://www.youtube.com", "そこに草があります",
        ):
            with self.subTest(text=text):
                self.assertEqual(strip_incidental_laughter(text), text)

    def test_short_exhale_is_rejected_only_as_whole_utterance(self):
        self.assertTrue(is_isolated_reaction("フッ"))
        self.assertTrue(is_isolated_reaction("ふっ。"))
        self.assertFalse(is_isolated_reaction("ふっと思った"))


class IsolatedReactionTests(unittest.TestCase):
    def test_aggressive_isolated_reactions_are_filtered(self):
        for text in (
            "あっ！", "はっ。", "はあ", "うわっ。", "うわー！うわー！",
            "うう！", "よいしょ", "よいしょよいしょ", "えーっと",
            "うん。", "はい。", "えっ？", "ん?", "すすげえ",
            "すごい！", "スゴイ！", "すげえ", "あのー", "うーん",
            "へぇ～", "はーい", "Wow!", "Hey!", "Whoa!", "おお!",
            "うわぁぁぁ！", "あっ!あっ!", "ッ!", "っ。", "ッッ!",
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
            "さっき、ッ!と言った", "ッ! 待って",
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
