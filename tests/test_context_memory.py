"""Short-term memory should help repeats without overwriting unrelated ASR."""

import queue
import unittest

from auto_translate.context_memory import (
    SceneMemory, build_recent_context, repair_near_repeated_word,
    reuse_recent_short_term,
)
from auto_translate.pipeline import run_translate
from auto_translate.types import AudioProgress, Transcript, Subtitle


class FakeGlossary:
    def find(self, text):
        return [("獅白ぼたん", "Shishiro Botan")] if "獅白ぼたん" in text else []


class FakeTranslator:
    memory_lines = 24

    def __init__(self):
        self.calls = []

    def translate(self, ja, history):
        self.calls.append((ja, list(history)))
        return "Shutter!"


class ContextMemoryTests(unittest.TestCase):
    def test_adjacent_katakana_repetition(self):
        previous = [("シャッター!", "Shutter!")]
        self.assertEqual(
            repair_near_repeated_word("シャッタ!", previous),
            "シャッター!",
        )
        self.assertEqual(
            repair_near_repeated_word("シャッター、シャッタ、シャッター", []),
            "シャッター、シャッター、シャッター",
        )
        self.assertEqual(
            repair_near_repeated_word("シャッターシャッタシャッター", []),
            "シャッターシャッターシャッター",
        )

    def test_mismatches_and_non_repeated_speech_are_not_guessed(self):
        history = [("シャッター", "Shutter!")]
        cases = [
            ("シャタ", "シャタ"),  # More than one edit away
            ("シャッターを開けて", "シャッターを開けて"),
            ("ボタン", "ボタン"),  # Different word
            ("シャッタ", "シャッタ"),  # With no history, cannot infer
        ]
        for original, expected in cases[:3]:
            with self.subTest(original=original):
                self.assertEqual(repair_near_repeated_word(original, history), expected)
        self.assertEqual(repair_near_repeated_word("シャッタ", []), "シャッタ")

    def test_reuse_only_identical_short_terms_and_respect_punctuation(self):
        history = [("シャッター!", "Shutter!")]
        self.assertEqual(
            reuse_recent_short_term("シャッター！", history),
            "Shutter!",
        )
        self.assertEqual(
            reuse_recent_short_term("シャッター?", history),
            "Shutter?",
        )
        self.assertIsNone(reuse_recent_short_term("シャッタ", history))
        self.assertIsNone(reuse_recent_short_term("シャッターを開けて", history))
        self.assertIsNone(
            reuse_recent_short_term("シャッター", [("シャッター", "Open the shutter!")])
        )
        self.assertIsNone(
            reuse_recent_short_term("シャッター", [("シャッター", "Shutter!")] + [("別の話", "Other.")] * 2)
        )

    def test_memory_remembers_names_terms_and_older_scene(self):
        history = [
            ("獅白ぼたんの配信", "Botan is streaming."),
            ("今日はレースだ", "It's a race today."),
            ("車に乗って", "Get in the car."),
            ("誰が運転するの", "Who's driving?"),
        ] + [("これはテスト", "This is a test.")] * 8 + [
            ("シャッター", "Shutter!"),
        ]
        context = build_recent_context(
            history, FakeGlossary(), context_lines=8, memory_lines=24,
        )
        self.assertIn("Shishiro Botan", context)
        self.assertIn("シャッター ~ Shutter", context)
        self.assertIn("今日はレースだ", context)
        self.assertNotIn("Who's driving?", context)  # Old English guesses excluded.

    def test_memory_is_bounded_and_empty_when_no_history(self):
        history = [("獅白ぼたん", "Botan")] + [("これはテスト", "Test")] * 30
        context = build_recent_context(
            history, FakeGlossary(), context_lines=8, memory_lines=24,
        )
        self.assertNotIn("Shishiro Botan", context)
        self.assertEqual(
            build_recent_context([], FakeGlossary()), "",
        )

    def test_pipeline_repair_is_visible_and_resets_on_seek(self):
        translator = FakeTranslator()
        incoming, outgoing = queue.Queue(), queue.Queue()
        incoming.put(Transcript(1, 2, "シャッター"))
        incoming.put(Transcript(2, 3, "シャッタ"))
        incoming.put(AudioProgress(sample=160000, reset=True, generation=1))
        incoming.put(Transcript(11, 12, "シャッタ", generation=1))
        incoming.put(None)

        run_translate(translator, 8, incoming, outgoing)
        output = []
        while True:
            item = outgoing.get_nowait()
            if item is None:
                break
            output.append(item)

        subtitles = [item for item in output if isinstance(item, Subtitle)]
        self.assertEqual(
            [item.ja for item in subtitles],
            ["シャッター", "シャッター", "シャッタ"],
        )
        self.assertEqual(
            sum(isinstance(item, AudioProgress) for item in output), 1,
        )
        self.assertEqual(translator.calls[1][0], "シャッター")
        self.assertEqual(translator.calls[2][1], [])

    def test_scene_memory_survives_many_unrelated_lines_without_hallucinating(self):
        memory = SceneMemory(max_idle_lines=24)
        self.assertEqual(memory.hint(), "")
        memory.observe("新聞買ったよ")
        self.assertEqual(memory.hint(), "")  # A newspaper alone isn't a race.
        memory.observe("馬券買おう")
        memory.observe("オッズを確認")
        for _ in range(10):
            memory.observe("そうだね")
        self.assertEqual(memory.current_topic(), "horse_racing")
        self.assertIn("betting tickets", memory.hint())
        self.assertIn("betting odds", memory.hint())
        self.assertNotIn("新聞", memory.hint())
        self.assertLess(len(memory.hint()), 200)

    def test_scene_switch_requires_evidence_and_suppresses_stale_hint(self):
        memory = SceneMemory()
        memory.observe("競馬で馬券を買う")
        self.assertEqual(memory.current_topic(), "horse_racing")
        memory.observe("恐竜がいる")
        self.assertIsNone(memory.current_topic())  # Pending ARK: no old racing hint.
        self.assertEqual(memory.hint(), "")
        memory.observe("ギガノトをテイムする")
        self.assertEqual(memory.current_topic(), "ark")
        self.assertIn("ARK survival gameplay", memory.hint())
        self.assertNotIn("horse racing", memory.hint())
        memory.reset()
        self.assertEqual(memory.hint(), "")

    def test_conflicting_scene_evidence_does_not_guess_a_topic(self):
        memory = SceneMemory()
        memory.observe("ARKとGTAを比較しよう")
        self.assertIsNone(memory.current_topic())
        memory.observe("競馬と恐竜が好き")
        self.assertIsNone(memory.current_topic())
        memory.observe("馬券買った")
        self.assertEqual(memory.current_topic(), "horse_racing")
        # Independent conflicting topics do not replace established scene.
        memory.observe("ARKとGTA")
        self.assertEqual(memory.current_topic(), "horse_racing")
        self.assertLessEqual(SceneMemory(max_idle_lines=10000).max_idle_lines, 96)

    def test_refund_or_payout_alone_does_not_invent_horse_racing(self):
        for japanese in ("払い戻し行こうか", "払戻しはどこ?", "返金して"):
            with self.subTest(japanese=japanese):
                memory = SceneMemory()
                memory.observe(japanese)
                self.assertIsNone(memory.current_topic())
                self.assertEqual(memory.hint(), "")

        # A betting ticket supplies independent evidence for racing.
        memory = SceneMemory()
        memory.observe("払い戻し行こうか")
        self.assertEqual(memory.hint(), "")
        memory.observe("馬券を買った")
        self.assertEqual(memory.current_topic(), "horse_racing")
        memory.observe("払い戻し行こうか")
        self.assertIn("payouts", memory.hint())

    def test_gta_robbery_cues_preserve_context_without_claiming_identity(self):
        memory = SceneMemory()
        self.assertEqual(memory.hint(), "")
        memory.observe("店舗強盗行ける人いますか?")
        self.assertEqual(memory.current_topic(), "gta")
        memory.observe("コンビニに行こう、無線で確認")
        self.assertIn("store robberies", memory.hint())
        self.assertIn("radio communication", memory.hint())
        for _ in range(7):
            memory.observe("はい")
        self.assertIn("GTA", memory.hint())
        memory.reset()
        self.assertEqual(memory.hint(), "")

    def test_stale_scene_expires_and_weak_terms_do_not_create_one(self):
        memory = SceneMemory(max_idle_lines=8)
        memory.observe("ワイド画面で新聞を見る")
        self.assertEqual(memory.hint(), "")
        memory.observe("パドックを見よう")
        self.assertEqual(memory.current_topic(), "horse_racing")
        for _ in range(9):
            memory.observe("うん")
        self.assertEqual(memory.hint(), "")

    def test_cached_utterances_reconstruct_scene_and_seek_clears_it(self):
        class SceneFakeTranslator(FakeTranslator):
            def __init__(self):
                super().__init__()
                self.scene = SceneMemory()
                self.seen = []
                self.resets = 0

            def reset_scene_context(self):
                self.scene.reset()
                self.resets += 1

            def observe_scene(self, ja):
                self.scene.observe(ja)

            def translate(self, ja, history):
                self.observe_scene(ja)
                self.seen.append(self.scene.hint())
                return super().translate(ja, history)

        fake = SceneFakeTranslator()
        incoming, outgoing = queue.Queue(), queue.Queue()
        # Cached subtitle bypasses the translator but must still seed context.
        incoming.put(Subtitle(1, 2, "競馬の馬券を買う", "Buy a bet."))
        incoming.put(Transcript(2, 3, "そうだよ"))
        incoming.put(AudioProgress(sample=5 * 16000, reset=True, generation=1))
        incoming.put(Transcript(6, 7, "そうだよ", generation=1))
        incoming.put(Transcript(7, 8, "オッズを見る", generation=1))
        incoming.put(None)
        run_translate(fake, 5, incoming, outgoing)
        self.assertEqual(len(fake.seen), 3)
        self.assertIn("horse racing", fake.seen[0])
        self.assertEqual(fake.seen[1], "")
        self.assertIn("horse racing", fake.seen[2])
        self.assertEqual(fake.resets, 1)

    def test_pipeline_memory_extends_beyond_dialogue_context(self):
        translator = FakeTranslator()
        incoming, outgoing = queue.Queue(), queue.Queue()
        for i in range(15):
            incoming.put(Transcript(i, i + 1, f"話{i}"))
        incoming.put(None)
        run_translate(translator, 8, incoming, outgoing)
        self.assertEqual(len(translator.calls[-1][1]), 14)


if __name__ == "__main__":
    unittest.main()
