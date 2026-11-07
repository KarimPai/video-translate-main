"""Short-term memory should help repeats without overwriting unrelated ASR."""

import queue
import unittest

from auto_translate.context_memory import build_recent_context, repair_near_repeated_word, reuse_recent_short_term
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
