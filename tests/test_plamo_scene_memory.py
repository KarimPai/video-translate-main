"""Regression tests for evidence-based PLaMo scene memory and seek handling."""
import io
import json
import queue
import unittest
from unittest.mock import patch

from auto_translate.config import TranslateConfig
from auto_translate.context_memory import SceneMemory
from auto_translate.glossary import Glossary
from auto_translate.pipeline import run_translate
from auto_translate.translate.plamo import PlamoTranslator, build_plamo_prompt
from auto_translate.types import AudioProgress, Subtitle, Transcript


class PlamoSceneMemoryTests(unittest.TestCase):
    @staticmethod
    def translator(*, max_idle_lines=8):
        # Skip model download and CUDA process; exercise actual PLaMo methods.
        obj = object.__new__(PlamoTranslator)
        obj.cfg = TranslateConfig()
        obj.memory_lines = 0
        obj.glossary = Glossary({"わため": "Watame"})
        obj.vod_title = "Hololive GTA"
        obj.base_url = obj.cfg.plamo_base_url
        obj.url = obj.base_url + "/completions"
        obj._reference_context = None
        obj._scene_memory = SceneMemory(max_idle_lines=max_idle_lines)
        return obj

    @staticmethod
    def completion_spy(obj):
        prompts = []

        def fake_open(request, timeout):
            payload = json.loads(request.data)
            prompts.append(payload["prompt"])
            return io.BytesIO(json.dumps({
                "choices": [{"text": "Okay."}],
            }).encode("utf-8"))

        return prompts, fake_open

    def test_unrelated_or_weak_lines_do_not_receive_invented_game_context(self):
        obj = self.translator()
        prompts, fake_open = self.completion_spy(obj)
        with patch("auto_translate.translate.plamo.urllib.request.urlopen",
                   side_effect=fake_open):
            obj.translate("車に乗りたい", [])
            obj.translate("今日は服を買う", [])
            obj.translate("新聞を買った", [])
        self.assertEqual(len(prompts), 3)
        self.assertTrue(all("<|plamo:tag|>context" not in prompt for prompt in prompts))
        self.assertEqual(obj._scene_memory.current_topic(), None)

    def test_verified_gta_scene_reaches_pfml3_and_ages_out(self):
        obj = self.translator(max_idle_lines=8)
        prompts, fake_open = self.completion_spy(obj)
        with patch("auto_translate.translate.plamo.urllib.request.urlopen",
                   side_effect=fake_open):
            obj.translate("カージャックされた", [])
            obj.translate("カジノの経費で払って", [])
            for _ in range(9):
                obj.translate("そうだね", [])
        self.assertIn("GTA driving/crime gameplay", prompts[0])
        self.assertIn("casino roleplay", prompts[1])
        self.assertIn("in-game expenses", prompts[1])
        self.assertIn("<|plamo:tag|>context", prompts[1])
        self.assertNotIn("GTA driving/crime gameplay", prompts[-1])
        self.assertEqual(obj._scene_memory.current_topic(), None)

    def test_switch_suppresses_stale_scene_without_inventing_new_one(self):
        obj = self.translator()
        prompts, fake_open = self.completion_spy(obj)
        with patch("auto_translate.translate.plamo.urllib.request.urlopen",
                   side_effect=fake_open):
            obj.translate("競馬の馬券を買おう", [])
            obj.translate("恐竜がいる", [])
            obj.translate("ギガノトをテイムした", [])
        self.assertIn("horse racing", prompts[0])
        self.assertNotIn("horse racing", prompts[1])
        self.assertNotIn("ARK survival", prompts[1])
        self.assertIn("ARK survival", prompts[2])
        self.assertNotIn("horse racing", prompts[2])

    def test_scene_context_and_same_speaker_prior_japanese_share_one_block(self):
        prompt = build_plamo_prompt(
            "今度ドライブしよう", [("わため", "Watame")],
            scene_context="GTA driving/crime gameplay",
            reference_context="わためはどこ?",
        )
        self.assertEqual(prompt.count("<|plamo:tag|>context"), 1)
        self.assertIn("Scene background (not spoken): GTA driving/crime gameplay", prompt)
        self.assertIn("Previous Japanese (verified same speaker, not current): わためはどこ?", prompt)
        self.assertNotIn("previous English", prompt)
        self.assertLess(prompt.index("GTA driving"), prompt.index("わためはどこ"))
        self.assertLess(prompt.index("わためはどこ"), prompt.index("<|plamo:tag|>input"))

    def test_reset_clears_scene_and_unconsumed_speaker_reference(self):
        obj = self.translator()
        obj.observe_scene("カージャックされた")
        obj.set_reference_context("さっきの話")
        self.assertEqual(obj._scene_memory.current_topic(), "gta")
        obj.reset_scene_context()
        self.assertEqual(obj._scene_memory.current_topic(), None)
        self.assertIsNone(obj._reference_context)

    def test_cached_subtitles_restore_scene_and_seek_clears_it(self):
        obj = self.translator()
        seen = []
        def inference(ja, *, timeout):
            seen.append((ja, obj._scene_memory.hint()))
            return "Okay."
        incoming, outgoing = queue.Queue(), queue.Queue()
        incoming.put(Subtitle(1, 2, "競馬の馬券を買う", "Bought tickets."))
        incoming.put(Transcript(2.1, 3.0, "そうだね"))
        incoming.put(AudioProgress(sample=5 * 16000, reset=True, generation=1))
        incoming.put(Transcript(6, 7, "そうだね", generation=1))
        incoming.put(None)
        with patch.object(obj, "_complete", side_effect=inference):
            run_translate(obj, 0, incoming, outgoing)
        self.assertEqual(len(seen), 2)
        self.assertIn("horse racing", seen[0][1])
        self.assertEqual(seen[1][1], "")

    def test_long_audio_gap_forgets_earlier_scene(self):
        obj = self.translator()
        seen = []
        def inference(ja, *, timeout):
            seen.append(obj._scene_memory.hint())
            return "Okay."
        incoming, outgoing = queue.Queue(), queue.Queue()
        incoming.put(Transcript(0, 2, "カージャックされた"))
        incoming.put(Transcript(45, 46, "そうだね"))
        incoming.put(None)
        with patch.object(obj, "_complete", side_effect=inference):
            run_translate(obj, 0, incoming, outgoing)
        self.assertIn("GTA driving", seen[0])
        self.assertEqual(seen[1], "")

    def test_scene_survives_handoff_but_not_prior_other_speaker_words(self):
        obj = self.translator()
        captured = []
        def fake_open(request, timeout):
            captured.append(json.loads(request.data)["prompt"])
            return io.BytesIO(json.dumps({"choices": [{"text": "Okay."}]}).encode())
        incoming, outgoing = queue.Queue(), queue.Queue()
        incoming.put(Transcript(0, 2, "カージャックされた", speaker_context_key=1))
        incoming.put(Transcript(
            2.05, 3.9, "わためはどうする?", speaker_handoff=True,
            speaker_context_key=2,
        ))
        incoming.put(None)
        with patch("auto_translate.translate.plamo.urllib.request.urlopen",
                   side_effect=fake_open):
            run_translate(obj, 0, incoming, outgoing)
        self.assertEqual(len(captured), 2)
        self.assertIn("GTA driving", captured[1])
        self.assertNotIn("Previous Japanese (verified same speaker", captured[1])


if __name__ == "__main__":
    unittest.main()
