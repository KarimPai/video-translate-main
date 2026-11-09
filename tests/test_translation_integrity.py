"""Deterministic safeguards for ReazonSpeech -> PLaMo JP/EN subtitles.

No model downloads or llama.cpp server needed.
"""
import io
import json
import queue
import unittest
from unittest.mock import patch

from auto_translate.config import TranslateConfig
from auto_translate.glossary import Glossary
from auto_translate.pipeline import run_translate
from auto_translate.translation_integrity import (
    enforce_known_name_spellings,
    enforce_numeric_fidelity,
    japanese_number_facts,
    nearby_japanese_fragments,
)
from auto_translate.translate.plamo import PlamoTranslator
from auto_translate.types import AudioProgress, Subtitle, Transcript


class TranslationIntegrityTests(unittest.TestCase):
    def test_reazon_number_8028_not_828(self):
        self.assertEqual(
            enforce_numeric_fidelity(
                "八千二十八に行ったらそこの上にたどりつけると思う",
                "If you go to 828, you can reach the top.",
            ),
            "If you go to 8028, you can reach the top.",
        )

    def test_number_7240_and_original_punctuation(self):
        self.assertEqual(
            enforce_numeric_fidelity(
                "七千二百四十とかに出るのかな",
                "I wonder if it leads to 7,240, though.",
            ),
            "I wonder if it leads to 7,240, though.",
        )
        self.assertEqual(
            enforce_numeric_fidelity(
                "七千二百四十とかに出るのかな",
                "Does it lead to 724?",
            ),
            "Does it lead to 7240?",
        )

    def test_japanese_numeric_facts_explicit_only(self):
        self.assertEqual(japanese_number_facts("八千二十八"), [("八千二十八", 8028)])
        self.assertEqual(japanese_number_facts("七千二百四十"), [("七千二百四十", 7240)])
        self.assertEqual(japanese_number_facts("二十八"), [("二十八", 28)])
        self.assertEqual(japanese_number_facts("二〇二六"), [("二〇二六", 2026)])
        self.assertEqual(japanese_number_facts("GTA8028"), [("8028", 8028)])
        self.assertEqual(japanese_number_facts("万一そうなったら"), [])

    def test_do_not_replace_unrelated_or_ambiguous_numbers(self):
        self.assertEqual(enforce_numeric_fidelity("さっき", "GTA 5"), "GTA 5")
        self.assertEqual(
            enforce_numeric_fidelity("八千二十八", "Try 828 or 90."),
            "Try 828 or 90.",
        )
        self.assertEqual(
            enforce_numeric_fidelity("八千二十八", "Try 8:28."),
            "Try 8:28.",
        )
        self.assertEqual(
            enforce_numeric_fidelity("八千二十八", "Try eight hundred."),
            "Try eight hundred.",
        )

    def test_two_japanese_number_occurrences(self):
        self.assertEqual(
            enforce_numeric_fidelity(
                "八千二十八の八千二十八かな",
                "Is it 828 or 828?",
            ),
            "Is it 8028 or 8028?",
        )

    def test_no_invented_names(self):
        glossary = Glossary({"マリン": "Marine", "ベイ": "Bae"})
        self.assertEqual(
            enforce_known_name_spellings(
                "マリンが着いた", "Marin has arrived.", glossary,
            ),
            "Marine has arrived.",
        )
        self.assertEqual(
            enforce_known_name_spellings(
                "ベイの通訳もよろしく", "Please translate for Bay too.", glossary,
            ),
            "Please translate for Bae too.",
        )
        self.assertEqual(
            enforce_known_name_spellings(
                "こっちの湾", "The bay is here.", glossary,
            ),
            "The bay is here.",
        )
        self.assertEqual(
            enforce_known_name_spellings(
                "アイヌ先輩の車", "Ainu-senpai's car.", glossary,
            ),
            "Ainu-senpai's car.",
        )

    def test_adjacent_fragment_gives_only_past_japanese(self):
        recent = [(10.0, 10.5, "今正面の道")]
        self.assertEqual(
            nearby_japanese_fragments(
                recent, 10.8, "ところに行ったんですけど"
            ),
            ("今正面の道",),
        )

    def test_distant_fragment_not_used(self):
        self.assertEqual(
            nearby_japanese_fragments(
                [(10.0, 10.5, "今正面の道")], 15.0, "ところに行ったんですけど"
            ), (),
        )

    def test_reaction_breaks_fragment_context(self):
        recent = [
            (10.0, 10.5, "今正面の道"),
            (10.6, 10.75, "はい"),
        ]
        self.assertEqual(
            nearby_japanese_fragments(recent, 10.8, "ところに行ったんですけど"), ()
        )

    def test_fragment_context_disabled(self):
        self.assertEqual(
            nearby_japanese_fragments(
                [(10.0, 10.5, "今正面の道")],
                10.8, "ところに行ったんですけど", max_fragments=0
            ),
            (),
        )

    def test_live_pipeline_preserves_each_subtitle_timestamps(self):
        class DummyTranslator:
            cfg = TranslateConfig()
            memory_lines = 0

            def __init__(self):
                self.calls = []

            def translate(self, ja, history, *, fragment_context=()):
                self.calls.append((ja, tuple(fragment_context), list(history)))
                return f"translated {ja}"

        model = DummyTranslator()
        inp, out = queue.Queue(), queue.Queue()
        inp.put(Transcript(10.0, 10.5, "今正面の道"))
        inp.put(Transcript(10.8, 11.5, "ところに行ったんですけど"))
        inp.put(Transcript(20.0, 20.2, "別の話"))
        inp.put(None)
        run_translate(model, 8, inp, out)
        lines = []
        while True:
            sub = out.get_nowait()
            if sub is None:
                break
            lines.append(sub)
        self.assertEqual(len(lines), 3)
        self.assertEqual(
            [(sub.start_s, sub.end_s) for sub in lines],
            [(10.0, 10.5), (10.8, 11.5), (20.0, 20.2)],
        )
        self.assertEqual(model.calls[1][1], ("今正面の道",))
        self.assertEqual(model.calls[2][1], ())
        self.assertEqual(lines[1].ja, "ところに行ったんですけど")

    def test_seek_clears_fragment_context(self):
        class DummyTranslator:
            cfg = TranslateConfig()
            memory_lines = 0

            def __init__(self):
                self.contexts = []

            def translate(self, ja, history, *, fragment_context=()):
                self.contexts.append(tuple(fragment_context))
                return "okay"

        model = DummyTranslator()
        inp, out = queue.Queue(), queue.Queue()
        inp.put(Transcript(10.0, 10.5, "今正面の道", generation=0))
        inp.put(AudioProgress(sample=0, reset=True, generation=1))
        inp.put(Transcript(10.8, 11.5, "ところに行ったんですけど", generation=1))
        inp.put(None)
        run_translate(model, 8, inp, out)
        self.assertEqual(model.contexts, [(), ()])

    def test_plamo_template_carries_fragment_and_glossary_but_not_fused_text(self):
        model = PlamoTranslator.__new__(PlamoTranslator)
        model.cfg = TranslateConfig()
        model.vod_title = "Hololive GTA"
        model.glossary = Glossary({"マリン": "Marine"})
        model.url = "http://127.0.0.1:8000/v1/chat/completions"
        seen = []

        def fake_urlopen(req, timeout=None):
            seen.append(json.loads(req.data.decode("utf-8")))
            return io.BytesIO(b'{"choices":[{"message":{"content":"Marin goes to 828."}}]}')

        with patch("auto_translate.translate.plamo.urllib.request.urlopen",
                   side_effect=fake_urlopen):
            ja = "マリンは八千二十八に行った"
            output = model._chat(
                ja, [], timeout=1,
                fragment_context=("今正面の道",),
            )
            self.assertEqual(output, "Marin goes to 828.")
            result = model.translate(
                ja, [], fragment_context=("今正面の道",)
            )

        self.assertEqual(result, "Marine goes to 8028.")
        body = seen[-1]
        self.assertEqual(body["messages"][-1]["content"], ja)
        ctx = body["chat_template_kwargs"]["context"]
        self.assertIn("今正面の道", ctx)
        self.assertIn("八千二十八 = 8028", ctx)
        self.assertIn({"source": "マリン", "target": "Marine"},
                      body["chat_template_kwargs"]["glossary"])


if __name__ == "__main__":
    unittest.main()
