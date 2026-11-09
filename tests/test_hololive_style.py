"""Hololive metadata naming and conservative concise subtitle regressions.

The tests are offline and use no Whisper, YouTube, or GPU model downloads.
"""
import queue
import unittest
from dataclasses import replace

import numpy as np

from auto_translate.config import WhisperConfig
from auto_translate.context import RecentSpeechContext
from auto_translate.hololive_names import TALENTS, select_hololive_hints
from auto_translate.pipeline import run_whisper
from auto_translate.subtitle_style import concise_english
from auto_translate.types import Utterance
from auto_translate.whisper_v3 import WhisperDecode


class HololiveNameTests(unittest.TestCase):
    def test_official_reference_has_jp_en_and_recent_devis_members(self):
        names = {talent.official for talent in TALENTS}
        self.assertGreaterEqual(len(TALENTS), 75)
        for name in [
            "Tsunomaki Watame", "Nakiri Ayame", "Nekomata Okayu",
            "Shishiro Botan", "Oozora Subaru", "Otonose Kanade",
            "Juufuutei Raden", "Kikirara Vivi", "Isaki Riona",
            "Gawr Gura", "Murasaki Shion",
        ]:
            self.assertIn(name, names)

    def test_channel_identity_is_selected_before_title_collaborators(self):
        found = select_hololive_hints({
            "channel": "猫又おかゆ Nekomata Okayu",
            "title": "GTA with Watame, Botan and Towa",
        })
        self.assertEqual(found, ["Okayu", "Watame", "Towa", "Botan"])

    def test_title_with_japanese_names(self):
        found = select_hololive_hints({
            "channel": "Unknown producer",
            "title": "常闇トワ と 角巻わため GTA配信",
        })
        self.assertEqual(found, ["Watame", "Towa"])

    def test_ambiguous_unrelated_channel_receives_no_hints(self):
        self.assertEqual(select_hololive_hints({
            "title": "New GTA VOD day 4", "channel": "Gaming channel",
        }), [])
        self.assertEqual(select_hololive_hints(None), [])
        self.assertEqual(select_hololive_hints({
            "channel": "Nekomata Okayu", "title": "GTA",
        }, enabled=False), [])
        self.assertEqual(select_hololive_hints({
            "channel": "Nekomata Okayu",
        }, limit=0), [])

    def test_limit_and_manual_names_are_bounded(self):
        selected = select_hololive_hints({
            "channel": "Nekomata Okayu",
            "title": "Watame Towa Botan Korone Subaru Pekora",
        }, limit=3)
        self.assertEqual(len(selected), 3)
        self.assertEqual(selected[0], "Okayu")

        cfg = WhisperConfig(
            hotwords=["Watame", "Botan"],
            hololive_detected_names=["Okayu", "Watame", "Towa"],
        )
        actual = RecentSpeechContext(cfg).hotwords()
        self.assertEqual(actual, "Watame, Botan, Okayu, Towa")

    def test_manual_hotwords_can_still_be_used_for_unknown_stream(self):
        cfg = WhisperConfig(hotwords=["Watame"], hololive_detected_names=[])
        self.assertEqual(RecentSpeechContext(cfg).hotwords(), "Watame")


class ConciseStyleTests(unittest.TestCase):
    def test_remove_only_interjections_not_speech(self):
        self.assertEqual(concise_english("Um, well, I need armor."), "I need armor.")
        self.assertEqual(concise_english("Hahaha, you're fast."), "You're fast.")
        self.assertEqual(concise_english("well, well, well, I need a car."),
                         "I need a car.")
        self.assertEqual(concise_english("Yes, yes, yes, yes, I'll go."),
                         "Yes, I'll go.")
        self.assertEqual(concise_english("hahaha"), "")

    def test_preserve_substantive_meaning_and_profanity(self):
        for original in [
            "I don't want to go.",
            "No, no, no!",
            "Watame isn't Watanabe.",
            "That's fucking dangerous.",
            "Well done, Botan.",
            "No one can leave.",
        ]:
            self.assertEqual(concise_english(original), original)

    def test_cleanup_switches_can_turn_off_filtering(self):
        raw = "Um, um, um, I'll go."
        self.assertEqual(concise_english(raw, remove_fillers=False,
                                         collapse_repetitions=False), raw)

    def test_whisper_pipeline_passes_style_prompt_and_selected_names(self):
        class Fake:
            def __init__(self):
                self.cfg = WhisperConfig(hololive_detected_names=["Watame"])
                self.calls = []
            def decode_audio(self, audio, *, prompt=None, hotwords=None,
                             temperature=None):
                self.calls.append((prompt, hotwords))
                return WhisperDecode("Um, I need armor.", avg_logprob=-.2)
        engine = Fake()
        audio = np.ones(16000, dtype=np.float32) * .1
        incoming, outgoing = queue.Queue(), queue.Queue()
        incoming.put(Utterance(0, 1, audio))
        incoming.put(None)
        run_whisper(engine, incoming, outgoing)
        self.assertEqual(outgoing.get_nowait().en, "I need armor.")
        self.assertIsNone(outgoing.get_nowait())
        self.assertIsNone(engine.calls[0][0])  # no instruction-style Whisper priming
        self.assertEqual(engine.calls[0][1], "Watame")

    def test_explicit_style_prompt_can_still_be_opted_in(self):
        class Fake:
            def __init__(self):
                self.cfg = WhisperConfig(style_prompt="GTA gaming subtitles.")
                self.last_prompt = None
            def decode_audio(self, audio, *, prompt=None, hotwords=None,
                             temperature=None):
                self.last_prompt = prompt
                return WhisperDecode("Get in the car.", avg_logprob=-.2)
        engine = Fake()
        incoming, outgoing = queue.Queue(), queue.Queue()
        incoming.put(Utterance(0, 1, np.ones(16000, dtype=np.float32)))
        incoming.put(None)
        run_whisper(engine, incoming, outgoing)
        self.assertEqual(outgoing.get_nowait().en, "Get in the car.")
        self.assertEqual(engine.last_prompt, "GTA gaming subtitles.")

    def test_style_prompt_can_be_disabled_without_changing_raw_translation(self):
        class Fake:
            def __init__(self):
                self.cfg = WhisperConfig(concise_english=False)
                self.prompt = "marker"
            def decode_audio(self, audio, *, prompt=None, hotwords=None,
                             temperature=None):
                self.prompt = prompt
                return WhisperDecode("Um, I need armor.", avg_logprob=-.2)
        model = Fake()
        incoming, outgoing = queue.Queue(), queue.Queue()
        incoming.put(Utterance(0, 1, np.ones(16000, dtype=np.float32)))
        incoming.put(None)
        run_whisper(model, incoming, outgoing)
        self.assertEqual(outgoing.get_nowait().en, "Um, I need armor.")
        self.assertIsNone(model.prompt)


if __name__ == "__main__":
    unittest.main()
