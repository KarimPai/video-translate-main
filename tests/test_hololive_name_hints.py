"""Offline tests for evidence-limited Hololive name normalization."""
import unittest

from auto_translate.asr_correction import correct_known_asr_terms
from auto_translate.glossary import Glossary
from auto_translate.name_hints import whisper_name_prompt, canonicalize_english_names
from auto_translate.translate.plamo import _current_glossary_matches
from auto_translate.translation_guard import suspicious_translation
from pathlib import Path


class HololiveNameHintsTests(unittest.TestCase):
    def test_glossary_covers_spoken_nicknames(self):
        glossary = Glossary.load(Path("glossary.toml"))
        for jp, en in [("はーちゃまって初犯なんですか", "Haachama"),
                       ("はやちゃまは捕まってない", "Haachama"),
                       ("ラプラスラプラス", "Laplus")]:
            self.assertIn(en, [target for _, target in glossary.find(jp)])
        self.assertIn(("ラプラス", "Laplus"),
                      _current_glossary_matches("ラプラスが来た", "ホロGTA", glossary))
        self.assertIn(("ラプラス・ダークネス", "Laplus"),
                      _current_glossary_matches("ラプラス・ダークネス。", "ホロGTA", glossary))
        self.assertIn(("ラプラスダークネス", "Laplus"),
                      _current_glossary_matches("ラプラスダークネス。", "ホロGTA", glossary))

    def test_title_gated_name_correction_preserves_caption_conflict(self):
        ja = "確かはやちゃまは捕まってない"
        self.assertEqual(
            correct_known_asr_terms(ja, video_context="ホロGTA"),
            "確かはあちゃまは捕まってない",
        )
        self.assertEqual(correct_known_asr_terms(ja, video_context="料理配信"), ja)
        self.assertEqual(
            correct_known_asr_terms(ja, video_context="ホロGTA", caption=ja), ja,
        )

    def test_whisper_prompt_not_global(self):
        # No static name seed may bias Whisper for unrelated dialogue.
        self.assertIsNone(whisper_name_prompt("ホロGTA 3日目"))
        self.assertIsNone(whisper_name_prompt("ホロGTA 3日目", enabled=True))
        self.assertIsNone(whisper_name_prompt("普通のニュース放送"))
        self.assertIsNone(whisper_name_prompt("ホロGTA", enabled=False))

    def test_english_name_spelling_only_when_evidenced(self):
        self.assertEqual(
            canonicalize_english_names("はあちゃまが捕まった", "Hasa-chama got caught.", "ホロGTA"),
            "Haachama got caught.",
        )
        self.assertEqual(
            canonicalize_english_names("はーちゃまって初犯?", "Ha-chama's first offense?", "ホロGTA"),
            "Haachama's first offense?",
        )
        self.assertEqual(
            canonicalize_english_names("ラプラスラプラス", "Laplace, Laplace!", "ホロGTA"),
            "Laplus, Laplus!",
        )
        for original in (
            "La+ Darknesss.", "La+ Darkness.", "Laplus Darknesss.",
            "Laplace Darknesss.", "La Plus Darknesss.", "Laplus.",
        ):
            with self.subTest(original=original):
                self.assertEqual(
                    canonicalize_english_names("ラプラス・ダークネス。", original, "ホロGTA"),
                    "Laplus.",
                )
        self.assertEqual(
            canonicalize_english_names("ラプラスが来た", "La+ Darknesss arrived!", "ホロGTA"),
            "Laplus arrived!",
        )
        self.assertEqual(
            canonicalize_english_names("ラプラスだ", "Laplace.", "math lecture"), "Laplace.",
        )
        self.assertEqual(
            canonicalize_english_names("今日は晴れ", "La+ Darknesss.", "ホロGTA"),
            "La+ Darknesss.",
        )
        self.assertEqual(
            canonicalize_english_names("今日は晴れ", "Laplace.", "ホロGTA"), "Laplace.",
        )

    def test_guard_protects_matched_names(self):
        self.assertEqual(
            suspicious_translation("はあちゃまが来た", "Someone arrived",
                                   names=[("はあちゃま", "Haachama")]),
            "proper_name_mismatch",
        )
        self.assertIsNone(
            suspicious_translation("ラプラスが来た", "Laplus arrived",
                                   names=[("ラプラス", "Laplus")])
        )


if __name__ == "__main__":
    unittest.main()
