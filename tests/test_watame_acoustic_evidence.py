import unittest

from auto_translate.watame import correct_asr_name, reconcile_caption


class WatameAcousticEvidenceTests(unittest.TestCase):
    def test_title_alone_does_not_replace_real_surname(self):
        for spelling in ("渡辺", "渡邊", "渡部", "わたなべ", "ワタナベ", "Watanabe"):
            with self.subTest(spelling=spelling):
                source = f"{spelling}さんが来た"
                self.assertEqual(correct_asr_name(source, vod_title="角巻わため 配信"), source)

    def test_aligned_caption_can_repair_one_existing_sound_candidate(self):
        self.assertEqual(
            correct_asr_name("渡辺、そこにいるの？", caption="わため、そこにいるの？"),
            "わため、そこにいるの？",
        )

    def test_kanji_and_phonetic_caption_surnames_are_retained(self):
        for caption in ("渡辺さんが来た", "わたなべさんが来た", "ワタナベさんが来た"):
            self.assertEqual(
                correct_asr_name("渡辺さんが来た", vod_title="わため", caption=caption),
                "渡辺さんが来た",
            )

    def test_mixed_mentions_repair_only_caption_supported_occurrence(self):
        self.assertEqual(
            reconcile_caption("渡辺が渡辺さんに会った", "わためが渡辺さんに会った"),
            "わためが渡辺さんに会った",
        )
        self.assertEqual(
            reconcile_caption("渡辺さんが渡辺に会った", "わたなべさんがワタメに会った"),
            "渡辺さんがワタメに会った",
        )

    def test_each_spoken_form_and_honorific_is_preserved(self):
        self.assertEqual(
            reconcile_caption("渡辺ちゃんと渡辺さん", "わためちゃんとワタメさん"),
            "わためちゃんとワタメさん",
        )

    def test_partial_caption_or_changed_negation_cannot_repair(self):
        source = "渡辺が来ない"
        for caption in ("わため", "わためが来た", "わためが来ないからね", "夏色まつりが来ない"):
            self.assertEqual(reconcile_caption(source, caption), source)

    def test_missing_name_or_extra_name_in_caption_cannot_introduce_it(self):
        for source, caption in (("誰か来た", "わためが来た"),
                                ("渡辺が来た", "わためとわためが来た")):
            self.assertEqual(reconcile_caption(source, caption), source)

    def test_contradictory_existing_person_abstains(self):
        source = "わためと渡辺さん"
        self.assertEqual(reconcile_caption(source, "渡辺とわためさん"), source)


if __name__ == "__main__":
    unittest.main()
