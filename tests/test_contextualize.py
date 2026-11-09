"""Regression cases for live context repair, timestamps, seeks and greetings."""
import queue
import unittest

from auto_translate.contextualize import (
    repair_nearby_terms, run_contextualize, should_join_fragments,
)
from auto_translate.text_filter import rejection_reason
from auto_translate.translate.quality import repair_high_confidence_meaning
from auto_translate.types import AudioProgress, SAMPLE_RATE, Subtitle, Transcript


class ContextualizerTests(unittest.TestCase):
    @staticmethod
    def process(*items, title=""):
        incoming, outgoing = queue.Queue(), queue.Queue()
        for item in items:
            incoming.put(item)
        incoming.put(None)
        run_contextualize(incoming, outgoing, title=title)
        results = []
        while not outgoing.empty():
            results.append(outgoing.get_nowait())
        return results

    def test_following_explicit_brand_fixes_bad_asr_homophone(self):
        rows = self.process(
            Transcript(1, 1.6, "あの防号キング倒れた"),
            Transcript(1.7, 3.1, "バーガーキングは2回ぐらい食べた"),
        )
        self.assertEqual(rows[0].ja, "あのバーガーキング倒れた")
        self.assertEqual(rows[0].start_s, 1)
        self.assertEqual(rows[0].end_s, 1.6)

    def test_prior_hamburger_topic_fixes_brand(self):
        rows = self.process(
            Transcript(0, 0.4, "何このハンバーガー"),
            Transcript(0.5, 1.0, "あの防御キング倒れた"),
        )
        self.assertIn("バーガーキング", rows[1].ja)

    def test_unrelated_homophone_is_not_changed(self):
        self.assertEqual(
            repair_nearby_terms("あの防号キング倒れた"),
            "あの防号キング倒れた",
        )

    def test_weapon_evidence_differentiates_ammo_from_jewels(self):
        rows = self.process(
            Transcript(1, 1.5, "アサルトだけ回収"),
            Transcript(1.7, 2.3, "玉買ったりしてもいい"),
        )
        self.assertEqual(rows[1].ja, "弾買ったりしてもいい")
        self.assertEqual(repair_nearby_terms("玉買ったりしてもいい"),
                         "玉買ったりしてもいい")

    def test_hideout_uses_independent_scene_context(self):
        self.assertEqual(
            repair_nearby_terms("8時にアジと集合", ["アジトに帰る"]),
            "8時にアジト集合",
        )
        self.assertEqual(
            repair_nearby_terms("8時にアジと集合"),
            "8時にアジと集合",
        )

    def test_honorific_botan_only_not_controller_button(self):
        self.assertEqual(repair_nearby_terms("ボタンちゃんは何してる"),
                         "ぼたんちゃんは何してる")
        self.assertEqual(repair_nearby_terms("ボタンを押して"),
                         "ボタンを押して")

    def test_merge_short_adjacent_continuation_preserves_media_time(self):
        rows = self.process(
            Transcript(1.0, 1.5, "でちょっと"),
            Transcript(1.58, 2.6, "車をカスタムして"),
        )
        self.assertEqual(len(rows), 2)  # one merged cue plus end marker
        self.assertEqual(rows[0].ja, "でちょっと 車をカスタムして")
        self.assertEqual((rows[0].start_s, rows[0].end_s), (1.0, 2.6))

    def test_handoff_never_joins_different_voices(self):
        first = Transcript(1, 1.5, "でちょっと")
        second = Transcript(1.56, 2, "車をカスタムして", speaker_handoff=True)
        self.assertFalse(should_join_fragments(first, second))
        rows = self.process(first, second)
        self.assertEqual(len([x for x in rows if isinstance(x, Transcript)]), 2)

    def test_complete_sentence_is_not_joined(self):
        rows = self.process(
            Transcript(1, 1.5, "そうです。"),
            Transcript(1.55, 2, "これも持って"),
        )
        self.assertEqual(len([x for x in rows if isinstance(x, Transcript)]), 2)

    def test_seek_discards_pending_old_generation(self):
        rows = self.process(
            Transcript(1, 1.5, "でちょっと", generation=0),
            AudioProgress(50 * SAMPLE_RATE, reset=True, generation=1),
            Transcript(50, 51, "新しい場面", generation=1),
        )
        texts = [x.ja for x in rows if isinstance(x, Transcript)]
        self.assertEqual(texts, ["新しい場面"])
        self.assertTrue(rows[0].reset)

    def test_media_progress_flushes_while_quiet(self):
        rows = self.process(
            Transcript(1, 1.5, "車の修理"),
            AudioProgress(4 * SAMPLE_RATE),
        )
        self.assertIsInstance(rows[0], Transcript)
        self.assertIsInstance(rows[1], AudioProgress)

    def test_cached_subtitle_is_not_translated_or_merged(self):
        cached = Subtitle(1, 1.5, "こんにちは", "Hello.")
        rows = self.process(cached)
        self.assertIs(rows[0], cached)


class MeaningAndFilterTests(unittest.TestCase):
    def test_real_short_reactions_survive(self):
        self.assertIsNone(rejection_reason("うん。"))
        self.assertIsNone(rejection_reason("すごい！"))

    def test_laughter_only_removed(self):
        self.assertEqual(rejection_reason("ギャーハハ"), "laughter_only")
        self.assertIsNone(rejection_reason("ギャーハハ、すごい！"))

    def test_greetings_and_names(self):
        self.assertEqual(repair_high_confidence_meaning("うーす", "Usu"), "Hey!")
        self.assertIn("Okayu", repair_high_confidence_meaning(
            "もしもしおかゆ", "Hello Yuka."
        ))
        self.assertIn("Botan-chan", repair_high_confidence_meaning(
            "ぼたんちゃん", "Button-chan called."
        ))

    def test_ammo_translation_only_with_explicit_source(self):
        self.assertIn("ammo", repair_high_confidence_meaning(
            "弾買ったりしてもいい", "You could buy jewels."
        ))
        self.assertEqual(
            repair_high_confidence_meaning("宝石買ったりしてもいい", "Buy jewels."),
            "Buy jewels.",
        )


if __name__ == "__main__":
    unittest.main()
