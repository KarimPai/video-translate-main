import unittest

from auto_translate.watame import correct_asr_name, correct_translation, reconcile_caption


class WatameTests(unittest.TestCase):
    def test_matching_caption_can_resolve_surname(self):
        self.assertEqual(reconcile_caption("渡辺が来た！", "わためが来た"), "わためが来た！")
        self.assertEqual(reconcile_caption("渡辺が来た", "わためは帰った"), "渡辺が来た")
        self.assertEqual(
            reconcile_caption(
                "今の渡辺が悪い渡辺が悪いよ",
                "今のわためが悪いわためが悪いよ",
            ),
            "今のわためが悪いわためが悪いよ",
        )

    def test_title_context_repairs_repeated_watame_asr_errors(self):
        spoken = "今の渡辺が悪い渡辺が悪いよ"
        self.assertEqual(
            correct_asr_name(spoken, vod_title="角巻わためのゲーム配信"),
            "今のわためが悪いわためが悪いよ",
        )
        self.assertEqual(
            correct_asr_name(spoken, vod_title="Hololive gaming"),
            spoken,
        )
        self.assertEqual(
            correct_asr_name(
                spoken,
                vod_title="Watame live",
                caption="今のワタナベが悪い",
            ),
            spoken,
        )

    def test_translation_preserves_distinct_names(self):
        self.assertEqual(correct_translation("ワタメが来た", "Watanabe arrived."), "Watame arrived.")
        for ja in ("渡辺が来た", "わためと渡辺", "ワタメとワタナベ"):
            self.assertEqual(correct_translation(ja, "Watanabe arrived."), "Watanabe arrived.")
