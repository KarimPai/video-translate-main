"""Regression examples from real Japanese gaming / Hololive subtitle output."""

import unittest

from auto_translate.subject_guard import neutralize_unsupported_subjects as fix


class SubjectGuardTests(unittest.TestCase):
    def test_stream_examples_do_not_invent_driver_or_crash_agent(self):
        self.assertEqual(
            fix(
                "なんか運転がまぁ上手くないから当たっちゃったんだよなぁ",
                "My driving's just not that great, so I ended up hitting it.",
            ),
            "The driving's just not that great, so there was a collision.",
        )
        self.assertEqual(
            fix(
                "いやよくないそれぶつかっちゃったんだよ",
                "No, that's not good—you just crashed into it.",
            ),
            "No, that's not good—there was a collision.",
        )
        self.assertEqual(
            fix("運転が下手だね", "Your driving is bad."),
            "The driving is bad.",
        )

    def test_explicit_pronouns_preserved(self):
        self.assertEqual(
            fix(
                "俺の運転が下手だからぶつかった",
                "My driving is bad, so I crashed into it.",
            ),
            "My driving is bad, so I crashed into it.",
        )
        self.assertEqual(
            fix(
                "お前の運転が悪いからぶつかった",
                "Your driving is bad, so you crashed into it.",
            ),
            "Your driving is bad, so you crashed into it.",
        )
        self.assertEqual(
            fix("彼の運転が下手だ", "His driving is bad."),
            "His driving is bad.",
        )

    def test_unrelated_or_uncertain_phrases_not_changed(self):
        examples = [
            ("大丈夫っすか?", "Are you okay?"),
            ("ごめんなさい", "My bad!"),
            ("ゲーム始めましょう", "Let's play!"),
            ("君は運転が好き", "Your driving is nice."),
            ("当たっちゃった", "I hit it."),  # Could mean winning, not collision
            ("ぶつかった", "I hit the bus."),
            ("ぶつかった", "Ayame crashed into it."),
            ("今こっちは揉めてます!助けてください!", "There's a fight going on here! Please help!"),
        ]
        for japanese, english in examples:
            with self.subTest(japanese=japanese):
                self.assertEqual(fix(japanese, english), english)

    def test_other_common_driving_phrasings(self):
        self.assertEqual(
            fix("運転が下手", "I'm bad at driving."),
            "The driving isn't very good.",
        )
        self.assertEqual(
            fix("運転が上手くない", "You're not very good at driving."),
            "The driving isn't very good.",
        )
        self.assertEqual(
            fix("運転が下手", "I can't drive well."),
            "The driving isn't very good.",
        )
        # Ability to drive is not the same as poor driving skill.
        self.assertEqual(
            fix("運転できない", "I can't drive well."),
            "I can't drive well.",
        )
        self.assertEqual(
            fix("俺は運転が下手", "I'm bad at driving."),
            "I'm bad at driving.",
        )
        self.assertEqual(
            fix("お前は運転が下手", "You're not very good at driving."),
            "You're not very good at driving.",
        )

    def test_other_gameplay_possessives(self):
        self.assertEqual(
            fix("エイムが悪い", "My aim is bad."),
            "The aim is bad.",
        )
        self.assertEqual(
            fix("そっちのエイムが悪い", "Your aim is bad."),
            "Your aim is bad.",
        )
        self.assertEqual(
            fix("ぶつかった", "I hit the bus with my car."),
            "I hit the bus with the car.",
        )


if __name__ == "__main__":
    unittest.main()
