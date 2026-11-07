import unittest

from auto_translate.watame import correct_translation, reconcile_caption


class WatameTests(unittest.TestCase):
    def test_matching_caption_can_resolve_surname(self):
        self.assertEqual(reconcile_caption("渡辺が来た！", "わためが来た"), "わためが来た！")
        self.assertEqual(reconcile_caption("渡辺が来た", "わためは帰った"), "渡辺が来た")

    def test_translation_preserves_distinct_names(self):
        self.assertEqual(correct_translation("ワタメが来た", "Watanabe arrived."), "Watame arrived.")
        for ja in ("渡辺が来た", "わためと渡辺", "ワタメとワタナベ"):
            self.assertEqual(correct_translation(ja, "Watanabe arrived."), "Watanabe arrived.")
