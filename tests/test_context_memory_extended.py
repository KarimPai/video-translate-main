"""Deterministic bounded-memory and source-fidelity regressions; no models."""

import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from auto_translate.config import TranslateConfig
from auto_translate.context_memory import (
    build_recent_context, recurring_term_hints, select_dialogue_history,
)
from auto_translate.glossary import Glossary
from auto_translate.translate.plamo import PlamoTranslator


class ExtendedContextMemoryTests(unittest.TestCase):
    def translator(self):
        obj = object.__new__(PlamoTranslator)
        obj.cfg = TranslateConfig()
        obj.memory_lines = 16
        obj.glossary = Glossary.load(Path(__file__).resolve().parents[1] / "glossary.toml")
        obj.vod_title = "ホロライブのゲーム配信"
        obj.url = "http://127.0.0.1:8000/v1/chat/completions"
        obj.client = MagicMock()
        obj.client.post.return_value.json.return_value = {
            "choices": [{"message": {"content": "Done."}}],
        }
        return obj

    def body(self, translator, current, history):
        translator._chat(current, history, timeout=1)
        return translator.client.post.call_args.kwargs["json"]

    def test_history_retrieves_relevant_older_turn_without_reordering(self):
        history = [
            ("赤いシャッターを開けて", "Open the red shutter."),
            ("関係ない話", "Another topic."),
            ("水を飲む", "Drink some water."),
            ("次に行こう", "Move along."),
        ]
        selected = select_dialogue_history(history, "赤いシャッターはどこ", context_lines=3)
        self.assertEqual(selected, [history[0], history[2], history[3]])

    def test_history_overflow_is_bounded_and_never_uses_evicted_turns(self):
        history = [(f"古い秘密{i}", f"old-{i}") for i in range(100)]
        history.extend(("新しい会話", "Recent conversation.") for _ in range(16))
        selected = select_dialogue_history(history, "古い秘密", context_lines=3, memory_lines=16)
        self.assertEqual(len(selected), 3)
        self.assertFalse(any("秘密" in source for source, _ in selected))

    def test_oversized_examples_are_omitted_whole(self):
        history = [("長" * 5000, "Long" * 5000), ("了解", "Roger.")]
        selected = select_dialogue_history(history, "長い", max_chars=30)
        self.assertEqual(selected, [history[-1]])
        self.assertLessEqual(sum(len(a) + len(b) for a, b in selected), 30)

    def test_memory_only_includes_current_source_glossary_terms(self):
        glossary = Glossary({"ラプラス": "Laplus", "シャッター": "Shutter"})
        history = [("ラプラス", "Laplus"), ("シャッター", "Shutter")]
        memory = build_recent_context(history, glossary, memory_lines=16,
                                      current_ja="シャッターを開けて")
        self.assertIn("シャッター = Shutter", memory)
        self.assertNotIn("Laplus", memory)

    def test_one_unverified_translation_cannot_become_a_learned_term(self):
        glossary = Glossary({})
        self.assertEqual(recurring_term_hints("シャッター", [("シャッター", "Shudder")], glossary), [])

    def test_conflicting_repeated_spellings_abstain(self):
        glossary = Glossary({})
        history = [("シャッター", "Shutter"), ("シャッター", "Shutter"), ("シャッター", "Shudder")]
        self.assertEqual(recurring_term_hints("シャッター", history, glossary), [])

    def test_consistent_isolated_term_is_reused_as_a_weak_hint(self):
        history = [("シャッター", "Shutter"), ("シャッター", "Shutter")]
        self.assertEqual(recurring_term_hints("シャッターを開けて", history, Glossary({})),
                         [("シャッター", "Shutter")])
        self.assertEqual(recurring_term_hints("シャッターチャンス", history, Glossary({})), [])

    def test_reviewed_glossary_wins_over_prior_translations(self):
        history = [("シャッター", "Shudder"), ("シャッター", "Shudder")]
        glossary = Glossary({"シャッター": "Shutter"})
        self.assertEqual(recurring_term_hints("シャッター", history, glossary), [])
        memory = build_recent_context(history, glossary, memory_lines=16, current_ja="シャッター")
        self.assertIn("シャッター = Shutter", memory)
        self.assertNotIn("Shudder", memory)

    def test_memory_budget_cannot_be_exceeded_by_large_glossary(self):
        glossary = Glossary({"シャッター": "Shutter", "アーマー": "Armor"})
        history = [("シャッター", "Shutter"), ("アーマー", "Armor")]
        memory = build_recent_context(history, glossary, memory_lines=16,
                                      current_ja="シャッターとアーマー", max_chars=80)
        self.assertLessEqual(len(memory), 80)

    def test_extended_toggle_keeps_only_short_adjacent_context(self):
        obj = self.translator()
        obj.cfg.extended_memory = False
        history = [("赤いシャッターを開けて", "Open the red shutter.")]
        history.extend(("了解", "Roger.") for _ in range(5))
        body = self.body(obj, "赤いシャッターはどこ", history)
        self.assertNotIn("Open the red shutter.", json.dumps(body, ensure_ascii=False))
        self.assertNotIn("Earlier Japanese", body["chat_template_kwargs"]["context"])
        self.assertIn("Roger.", str(body["messages"]))

    def test_prompt_retrieval_never_adds_gendered_prior_translation(self):
        obj = self.translator()
        body = self.body(obj, "赤いシャッターはどこ", [
            ("赤いシャッターを開けて", "He opened the red shutter."),
            ("了解", "Roger."), ("行くよ", "Moving."), ("待って", "Wait."),
        ])
        self.assertNotIn("He opened", json.dumps(body, ensure_ascii=False))
        self.assertEqual(body["messages"][-1]["content"], "赤いシャッターはどこ")

    def test_spoken_nicknames_win_over_conflicting_full_name_history(self):
        for source, expected in [("ラプ", "Lap"), ("ラプ様", "Lap-sama"),
                                 ("ラプちゃん", "Lap-chan"), ("ラプラス", "Laplus"),
                                 ("みこ", "Miko"), ("みこち", "Mikochi"),
                                 ("わため", "Watame"), ("まつり", "Matsuri")]:
            with self.subTest(source=source):
                obj = self.translator()
                prior = ({"みこ": ("さくらみこ", "Sakura Miko"),
                          "まつり": ("夏色まつり", "Natsuiro Matsuri")}
                         .get(source, ("ラプラス", "Laplus")))
                body = self.body(obj, source, [prior])
                self.assertIn({"source": source, "target": expected},
                              body["chat_template_kwargs"].get("glossary", []))

    def test_unverified_cached_word_does_not_bypass_translation(self):
        obj = self.translator()
        with patch.object(obj, "_chat", return_value="Shutter") as request:
            self.assertEqual(obj._translate_one("シャッター", [("シャッター", "Shudder")]), "Shutter")
        request.assert_called_once()


if __name__ == "__main__":
    unittest.main()
