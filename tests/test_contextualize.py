"""Regression cases for live context repair, timestamps, seeks and greetings."""
import queue
import threading
import time
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

    def test_joins_clipped_clause_to_its_explicit_continuation(self):
        rows = self.process(
            Transcript(1.0, 2.0, "降りていただければなと"),
            Transcript(2.08, 3.4, "思うんですけれどもえ"),
        )
        cues = [row for row in rows if isinstance(row, Transcript)]
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].start_s, 1.0)
        self.assertEqual(cues[0].end_s, 3.4)
        self.assertIn("思うんです", cues[0].ja)

    def test_does_not_join_clipped_clause_after_handoff_or_long_gap(self):
        a = Transcript(1, 2, "降りていただければなと")
        b = Transcript(2.1, 3.2, "思うんです", speaker_handoff=True)
        c = Transcript(3.0, 3.5, "思うんです")
        self.assertFalse(should_join_fragments(a, b))
        self.assertFalse(should_join_fragments(a, c))

    def test_only_isolated_decoder_particles_and_sfx_are_dropped(self):
        rows = self.process(
            Transcript(1.0, 1.2, "に"),
            Transcript(2.0, 2.2, "す"),
            Transcript(3.0, 3.3, "コクッ"),
            Transcript(4.0, 4.3, "ぺろ"),
            Transcript(5.0, 5.3, "うん。"),
            Transcript(6.0, 6.3, "はい。"),
            Transcript(7.0, 7.3, "すごい。"),
            Transcript(8.0, 8.3, "ぺこちゃん"),
            Transcript(9.0, 9.3, "警察が来た"),
        )
        cues = [row.ja for row in rows if isinstance(row, Transcript)]
        self.assertEqual(
            cues, ["うん。", "はい。", "すごい。", "ぺこちゃん", "警察が来た"],
        )

    def test_rocket_ammo_repair_requires_explicit_rocket_launcher(self):
        self.assertEqual(
            repair_nearby_terms("ロケランの玉一千万"),
            "ロケランの弾一千万",
        )
        self.assertEqual(
            repair_nearby_terms("野球の玉一千万"),
            "野球の玉一千万",
        )

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

    def test_wall_time_flush_releases_held_seek_progress_without_new_pcm(self):
        # An enhanced-audio stream can go quiet after its last watermark.
        # The subtitle must be translated before its held 5-second lead
        # watermark is released, even without another AudioProgress event.
        incoming, outgoing = queue.Queue(), queue.Queue()
        worker = threading.Thread(
            target=run_contextualize,
            args=(incoming, outgoing),
            kwargs={"lookahead_s": 0.15},
            daemon=True,
        )
        worker.start()
        try:
            incoming.put(Transcript(500.0, 500.2, "車を見て"))
            incoming.put(AudioProgress(round(500.3 * SAMPLE_RATE)))
            first = outgoing.get(timeout=1)
            self.assertIsInstance(first, AudioProgress)
            self.assertEqual(first.sample, 500 * SAMPLE_RATE)
            second = outgoing.get(timeout=1.5)
            third = outgoing.get(timeout=1.5)
            self.assertIsInstance(second, Transcript)
            self.assertEqual(second.ja, "車を見て")
            self.assertIsInstance(third, AudioProgress)
            self.assertEqual(third.sample, round(500.3 * SAMPLE_RATE))
        finally:
            incoming.put(None)
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())

    def test_pending_seek_progress_does_not_cross_generation_reset(self):
        rows = self.process(
            Transcript(4, 4.5, "古い字幕", generation=0),
            AudioProgress(5 * SAMPLE_RATE, generation=0),
            AudioProgress(900 * SAMPLE_RATE, reset=True, generation=1),
            Transcript(900, 900.5, "新しい字幕", generation=1),
            AudioProgress(901 * SAMPLE_RATE, generation=1),
        )
        texts = [r.ja for r in rows if isinstance(r, Transcript)]
        self.assertEqual(texts, ["新しい字幕"])
        resets = [r for r in rows if isinstance(r, AudioProgress) and r.reset]
        self.assertEqual(len(resets), 1)
        self.assertEqual(resets[0].sample, 900 * SAMPLE_RATE)
        self.assertTrue(all(
            r.sample >= 900 * SAMPLE_RATE
            for r in rows[rows.index(resets[0]):]
            if isinstance(r, AudioProgress)
        ))

    def test_pending_cue_limits_mpv_translation_progress(self):
        rows = self.process(
            Transcript(1, 1.5, "待って"),
            AudioProgress(2 * SAMPLE_RATE),
        )
        self.assertIsInstance(rows[0], AudioProgress)
        self.assertEqual(rows[0].sample, SAMPLE_RATE)
        self.assertIsInstance(rows[1], Transcript)

    def test_old_food_topic_does_not_rewrite_unrelated_scene(self):
        rows = self.process(
            Transcript(0, 0.5, "ハンバーガー食べた"),
            Transcript(21, 21.6, "あの防号キング倒れた"),
        )
        self.assertEqual(rows[1].ja, "あの防号キング倒れた")

    def test_small_vad_overlap_can_still_join_explicit_continuation(self):
        first = Transcript(1, 1.6, "でちょっと", speech_start_s=1.1)
        second = Transcript(1.5, 2.3, "車を修理して")
        self.assertTrue(should_join_fragments(first, second))
        rows = self.process(first, second)
        self.assertEqual(rows[0].speech_start_s, 1.1)
        self.assertEqual(rows[0].end_s, 2.3)

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
