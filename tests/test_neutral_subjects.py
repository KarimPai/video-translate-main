"""Speech-subject neutrality checks for PLaMo without a running model."""
import unittest

from auto_translate.config import TranslateConfig
from auto_translate.glossary import Glossary
from auto_translate.subject_guard import (
    unsupported_person_reference,
    neutralize_unwarranted_gender,
    neutralize_unwarranted_first_person,
    neutralize_unwarranted_possessives,
)
from auto_translate.translate.plamo import PlamoTranslator, _translation_context


class NeutralSubjectTests(unittest.TestCase):
    def test_source_unstated_person_is_flagged(self):
        self.assertEqual(
            unsupported_person_reference("どこ行った？", "Where did he go?"),
            "unsupported_male_reference",
        )
        self.assertEqual(
            unsupported_person_reference("もう終わった", "I finished already"),
            "unsupported_first_person_reference",
        )
        self.assertEqual(
            unsupported_person_reference("ありがとう", "She's great"),
            "unsupported_female_reference",
        )

    def test_explicit_or_strongly_implied_person_preserved(self):
        self.assertIsNone(unsupported_person_reference("俺がやった", "I did it."))
        self.assertIsNone(unsupported_person_reference("私が行く", "I'm going."))
        self.assertIsNone(unsupported_person_reference("こうしたい", "I want to do it this way."))
        self.assertIsNone(unsupported_person_reference("そう思う", "I think so."))
        self.assertIsNone(unsupported_person_reference("彼は走った", "He ran."))
        self.assertIsNone(unsupported_person_reference("彼女が来た", "She arrived."))

    def test_gender_neutral_rewrite_keeps_verb_grammar(self):
        self.assertEqual(
            neutralize_unwarranted_gender("向こうに行った", "He walked away."),
            "That person walked away.",
        )
        self.assertEqual(
            neutralize_unwarranted_gender("上手いな", "She is really good."),
            "That person is really good.",
        )
        self.assertEqual(
            neutralize_unwarranted_gender("何で？", "I told him to wait."),
            "I told them to wait.",
        )
        self.assertEqual(
            neutralize_unwarranted_gender("彼の車", "He drove his car."),
            "He drove his car.",
        )
        self.assertEqual(
            neutralize_unwarranted_gender("彼女の車", "She drove her car."),
            "She drove her car.",
        )
        self.assertEqual(
            neutralize_unwarranted_gender("あの人の車", "Her car was here."),
            "Their car was here.",
        )
        self.assertEqual(
            neutralize_unwarranted_gender("なんだ", "She won it herself."),
            "That person won it themselves.",
        )
        self.assertEqual(
            neutralize_unwarranted_gender("分からない", "She's gone."),
            "She's gone.",
        )
        self.assertEqual(
            neutralize_unwarranted_gender("ドア", "It's his."),
            "It's theirs.",
        )
        self.assertEqual(
            neutralize_unwarranted_gender("彼女が来た", "He came."),
            "He came.",  # unresolved, rather than silently erase explicit feminine evidence
        )

    def test_unstated_possessives_prefer_noun_phrase_not_their(self):
        self.assertEqual(
            neutralize_unwarranted_possessives("初めてのお客さん！", "Their first customer!"),
            "First customer!",
        )
        self.assertEqual(
            neutralize_unwarranted_possessives("最初のお客さん", "Your first customer."),
            "First customer.",
        )
        self.assertEqual(
            neutralize_unwarranted_possessives("初めてのお客さんが来た", "Their first customer arrived."),
            "The first customer arrived.",
        )
        self.assertEqual(
            neutralize_unwarranted_possessives("最初のお客さんだ", "It's their first customer."),
            "It's the first customer.",
        )
        self.assertEqual(
            neutralize_unwarranted_possessives("お客さんが来た", "Her customer arrived."),
            "The customer arrived.",
        )
        self.assertEqual(
            neutralize_unwarranted_possessives("店が開いた", "Their shop is open."),
            "The shop is open.",
        )

    def test_explicit_ownership_is_preserved_without_assuming_addressee(self):
        explicit = (
            ("あなたの最初のお客さん", "Your first customer."),
            ("お前の初めてのお客さん", "Your first customer."),
            ("俺の最初のお客さん", "My first customer."),
            ("彼女の初めてのお客さん", "Her first customer."),
            ("あの人の初めてのお客さん", "Their first customer."),
            ("ポルカさんの初めてのお客さん", "Their first customer."),
        )
        for ja, en in explicit:
            with self.subTest(japanese=ja):
                self.assertEqual(neutralize_unwarranted_possessives(ja, en), en)
                self.assertIsNone(unsupported_person_reference(ja, en))
        self.assertEqual(
            unsupported_person_reference("最初のお客さん", "Their first customer."),
            "unsupported_possessive_reference",
        )
        self.assertEqual(
            unsupported_person_reference("最初のお客さん", "Your first customer."),
            "unsupported_possessive_reference",
        )

    def test_safe_first_person_fallback_preserves_meaning(self):
        self.assertEqual(
            neutralize_unwarranted_first_person("分からない", "I don't know."),
            "Not sure.",
        )
        self.assertEqual(
            neutralize_unwarranted_first_person("すごいね", "I think that's incredible."),
            "That's incredible.",
        )
        self.assertEqual(
            neutralize_unwarranted_first_person("俺は分からない", "I don't know."),
            "I don't know.",
        )
        self.assertEqual(
            neutralize_unwarranted_first_person("そう思う", "I think so."),
            "I think so.",
        )
        self.assertEqual(
            neutralize_unwarranted_first_person("明日行く", "I will go tomorrow."),
            "I will go tomorrow.",
        )
        self.assertEqual(
            neutralize_unwarranted_first_person("もう終わった", "I finished it."),
            "Finished.",
        )
        self.assertEqual(
            neutralize_unwarranted_first_person("見つけた", "I found it!"),
            "Found it.",
        )
        self.assertEqual(
            neutralize_unwarranted_first_person("分かった", "I got it."),
            "Got it.",
        )

    @staticmethod
    def fake_translator(outputs):
        obj = object.__new__(PlamoTranslator)
        obj.cfg = TranslateConfig(target_lang="en", neutral_subjects=True)
        obj.glossary = Glossary({})
        obj.vod_title = "ホロGTA"
        obj.memory_lines = 0
        responses = iter(outputs)
        calls = []

        def chat(ja, history, timeout, *, neutral_retry=False):
            calls.append({"ja": ja, "history": list(history), "neutral_retry": neutral_retry})
            return next(responses)

        obj._chat = chat
        return obj, calls

    def test_retry_replaces_unsupported_gender_with_neutral_result(self):
        translator, calls = self.fake_translator(["He left.", "Someone left."])
        result = translator.translate("もう出た", [("さっき", "he ran earlier")])
        self.assertEqual(result, "Someone left.")
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[-1]["neutral_retry"])
        self.assertEqual(calls[-1]["history"], [])

    def test_failed_retry_falls_back_to_gender_neutral_phrase(self):
        translator, calls = self.fake_translator(["He left.", "He went away."])
        self.assertEqual(translator.translate("もう出た", []), "That person left.")
        self.assertEqual(len(calls), 2)

    def test_translation_retry_cleans_guessed_possessive(self):
        translator, calls = self.fake_translator([
            "Their first customer!", "First customer!",
        ])
        self.assertEqual(translator.translate("最初のお客さん！", []), "First customer!")
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[-1]["neutral_retry"])

    def test_translation_fallback_cleans_guessed_possessive(self):
        translator, calls = self.fake_translator([
            "Their first customer!", "Their first customer!",
        ])
        self.assertEqual(translator.translate("最初のお客さん！", []), "First customer!")
        self.assertEqual(len(calls), 2)
        translator2, calls2 = self.fake_translator(["Your first customer!"])
        self.assertEqual(
            translator2.translate("お前の最初のお客さん！", []),
            "Your first customer!",
        )
        self.assertEqual(len(calls2), 1)

    def test_first_person_implicit_retries_without_changing_explicit_i(self):
        translator, calls = self.fake_translator(["I finished already.", "Already finished."])
        self.assertEqual(translator.translate("終わった", []), "Already finished.")
        self.assertTrue(calls[-1]["neutral_retry"])
        translator2, calls2 = self.fake_translator(["I want to go."])
        self.assertEqual(translator2.translate("行きたい", []), "I want to go.")
        self.assertEqual(len(calls2), 1)

    def test_disable_subject_checks_without_disabling_existing_gaming_guard(self):
        translator, calls = self.fake_translator(["He left."])
        translator.cfg.neutral_subjects = False
        self.assertEqual(translator.translate("もう出た", []), "He left.")
        self.assertEqual(len(calls), 1)

    def test_prompt_clearly_preserves_explicit_person_and_neutrality(self):
        context = _translation_context("ホロGTA", neutral_retry=True)
        self.assertIn("'they/them'", context)
        self.assertIn("Do not insert 'their'", context)
        self.assertIn("First customer", context)
        self.assertIn("Preserve explicit or strongly implied first-person", context)
        self.assertIn("previous translation", context.lower())
        self.assertIn("video title", context.lower())


if __name__ == "__main__":
    unittest.main()
