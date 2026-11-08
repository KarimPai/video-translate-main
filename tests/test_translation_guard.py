"""No-model accuracy safeguards: avoid hallucination amplification."""
import queue
import unittest
from unittest.mock import patch

from auto_translate.config import TranslateConfig
from auto_translate.context_memory import SceneMemory
from auto_translate.glossary import Glossary
from auto_translate.pipeline import run_translate
from auto_translate.translate.plamo import PlamoTranslator
from auto_translate.translation_guard import (
    uncertain_asr_text, translation_quality_flags, trusted_merge_candidate,
)
from auto_translate.types import AudioProgress, Transcript


class GuardTests(unittest.TestCase):
    def test_suspect_asr_is_preserved_but_flagged(self):
        self.assertTrue(uncertain_asr_text("マイルファイル"))
        self.assertFalse(uncertain_asr_text("カージャック"))
        self.assertFalse(uncertain_asr_text("わため"))
        self.assertFalse(uncertain_asr_text("警察が来た"))
        self.assertFalse(uncertain_asr_text("おはようございます"))
        self.assertTrue(uncertain_asr_text(
            "今日はよくわかりませんでした",
            caption="警察官の方が今から来ます",
            caption_coverage=.94, caption_overlap_s=1.7,
        ))
        self.assertFalse(uncertain_asr_text(
            "今日はよくわかりませんでした",
            caption="警察官の方が今から来ます",
            caption_coverage=.35, caption_overlap_s=.2,
        ))

    def test_grounded_quality_flags_ignore_normal_conversation(self):
        self.assertEqual(translation_quality_flags(
            "おーい撃つなよ", "Hey, don't shoot!"), frozenset())
        self.assertEqual(translation_quality_flags(
            "帰れなかった", "Couldn't go home"), frozenset())
        self.assertEqual(translation_quality_flags(
            "40人です", "There are 40 people"), frozenset())
        self.assertEqual(translation_quality_flags(
            "何も悪くないよ", "There's nothing wrong"), frozenset())
        self.assertIn("numbers", translation_quality_flags(
            "40人です", "There are 50 people"))
        self.assertIn("negation", translation_quality_flags(
            "帰れなかった", "I went home"))
        self.assertIn("carry_location", translation_quality_flags(
            "だっこしていけばいい", "Carry them on your shoulders"))
        self.assertIn("nonpayment", translation_quality_flags(
            "請求を踏み倒した", "Devastated the bill"))

    def test_number_units_and_apologies_do_not_trigger_spurious_retries(self):
        self.assertEqual(translation_quality_flags(
            "1000万円です", "It's 10 million yen"), frozenset())
        self.assertEqual(translation_quality_flags(
            "すみません", "Sorry"), frozenset())
        self.assertEqual(translation_quality_flags(
            "申し訳ありません", "I'm sorry"), frozenset())

    def test_asr_stage_preserves_suspect_katakana_but_does_not_trust_it(self):
        import numpy as np
        from auto_translate.pipeline import run_asr
        from auto_translate.types import Utterance

        class ASR:
            def __init__(self, text):
                self.text = text
            def transcribe(self, audio, probe=False):
                return self.text

        for source, expected in (("マイルファイル", True),
                                 ("カージャック", False),
                                 ("警察が来た", False)):
            with self.subTest(source=source):
                incoming, outgoing = queue.Queue(), queue.Queue()
                incoming.put(Utterance(0, 2, np.zeros(32000, dtype=np.float32)))
                incoming.put(None)
                run_asr(ASR(source), incoming, outgoing)
                result = outgoing.get_nowait()
                self.assertEqual(result.ja, source)
                self.assertEqual(result.asr_uncertain, expected)

    def test_merge_requires_verified_continuity(self):
        kwargs = dict(
            first_text="あとはさけど",
            next_text="警察が来た",
            first_key=2,
            next_key=2,
            first_end_s=2.0,
            next_start_s=2.12,
            combined_duration_s=3.5,
            next_handoff=False,
            max_gap_s=.35,
        )
        self.assertTrue(trusted_merge_candidate(**kwargs))
        for overrides in (
            {"next_key": 3}, {"first_key": None}, {"next_handoff": True},
            {"next_start_s": 2.8}, {"combined_duration_s": 8.0},
            {"first_text": "もう帰るよ"},
        ):
            with self.subTest(overrides=overrides):
                self.assertFalse(trusted_merge_candidate(**(kwargs | overrides)))


class SelectivePlamoRetryTests(unittest.TestCase):
    @staticmethod
    def plamo():
        obj = object.__new__(PlamoTranslator)
        obj.cfg = TranslateConfig(
            selective_retry_enabled=True,
            selective_retry_timeout_s=3.0,
        )
        obj.glossary = Glossary({})
        obj.memory_lines = 0
        obj.vod_title = "GTA"
        obj._reference_context = None
        obj._source_uncertain = False
        obj._scene_memory = SceneMemory()
        return obj

    def test_normal_translation_uses_one_pass(self):
        tr = self.plamo()
        with patch.object(tr, "_complete", return_value="Hey, don't shoot!") as primary:
            with patch.object(tr, "_complete_retry") as retry:
                self.assertEqual(tr.translate("おーい撃つなよ", []), "Hey, don't shoot!")
        primary.assert_called_once()
        retry.assert_not_called()

    def test_retry_replaces_only_detectable_contradiction(self):
        tr = self.plamo()
        with patch.object(tr, "_complete", return_value="I went home") as primary:
            with patch.object(tr, "_complete_retry", return_value="I couldn't go home") as retry:
                result = tr.translate("帰れなかった", [])
        self.assertEqual(result, "I couldn't go home")
        primary.assert_called_once()
        retry.assert_called_once()

    def test_retry_keeps_original_when_no_improvement(self):
        tr = self.plamo()
        with patch.object(tr, "_complete", return_value="I went home"):
            with patch.object(tr, "_complete_retry", return_value="I went there"):
                self.assertEqual(tr.translate("帰れなかった", []), "I went home")

    def test_single_retry_can_recover_invalid_completion(self):
        tr = self.plamo()
        with patch.object(tr, "_complete", return_value="I don't see the Japanese text."):
            with patch.object(tr, "_complete_retry", return_value="I'm here"):
                self.assertEqual(tr.translate("ここにいる", []), "I'm here")

    def test_uncertain_haas_skips_context_and_retry(self):
        tr = self.plamo()
        tr.observe_scene("カージャックされた")
        tr.set_source_reliability(True)
        with patch.object(tr, "_complete", return_value="I went home") as primary:
            with patch.object(tr, "_complete_retry") as retry:
                self.assertEqual(tr.translate("帰れなかった", []), "I went home")
        retry.assert_not_called()
        self.assertFalse(tr._source_uncertain)
        self.assertEqual(tr._scene_memory.current_topic(), "gta")
        primary.assert_called_once()

    def test_source_uncertainty_drops_scene_and_speaker_prompt_hints(self):
        import io
        import json
        tr = self.plamo()
        tr.observe_scene("カージャックされた")
        tr.set_reference_context("警察が来た")
        tr.set_source_reliability(True)
        requests = []
        def responder(request, timeout):
            requests.append(json.loads(request.data)["prompt"])
            return io.BytesIO(json.dumps(
                {"choices": [{"text": "Milefile"}]}
            ).encode("utf-8"))
        tr.url = "http://127.0.0.1:8005/v1/completions"
        with patch("auto_translate.translate.plamo.urllib.request.urlopen",
                   side_effect=responder):
            self.assertEqual(tr.translate("マイルファイル", []), "Milefile")
        self.assertNotIn("<|plamo:tag|>context", requests[0])
        self.assertEqual(tr._scene_memory.current_topic(), "gta")

    def test_retry_failure_keeps_primary_and_does_not_recurse(self):
        tr = self.plamo()
        with patch.object(tr, "_complete", return_value="I went home"):
            with patch.object(tr, "_complete_retry", side_effect=TimeoutError("slow")) as retry:
                self.assertEqual(tr.translate("帰れなかった", []), "I went home")
        retry.assert_called_once()


class VerifiedFragmentPipelineTests(unittest.TestCase):
    class Spy:
        memory_lines = 0
        cfg = TranslateConfig(fragment_join_enabled=True, fragment_join_wait_s=.5)

        def __init__(self):
            self.japanese = []
            self.reliability = []

        def set_source_reliability(self, suspect):
            self.reliability.append(suspect)

        def translate(self, ja, history):
            self.japanese.append(ja)
            return "translated"

    @staticmethod
    def trans(start, end, ja, *, key=None, handoff=False, suspect=False, generation=0):
        return Transcript(
            start, end, ja, speaker_context_key=key,
            speaker_handoff=handoff, asr_uncertain=suspect,
            generation=generation,
        )

    def run_sequence(self, *items):
        tr = self.Spy()
        incoming, outgoing = queue.Queue(), queue.Queue()
        for it in items:
            incoming.put(it)
        incoming.put(None)
        run_translate(tr, 0, incoming, outgoing)
        outputs = []
        while (item := outgoing.get_nowait()) is not None:
            outputs.append(item)
        return tr, outputs

    def test_join_verified_same_speaker_only(self):
        spy, subs = self.run_sequence(
            self.trans(0, 2, "行きたいけど", key=4),
            self.trans(2.1, 3.4, "今は無理だよ", key=4),
        )
        self.assertEqual(spy.japanese, ["行きたいけど今は無理だよ"])
        self.assertEqual((subs[0].start_s, subs[0].end_s), (0, 3.4))

    def test_never_join_other_speaker_or_uncertain_asr(self):
        spy, _ = self.run_sequence(
            self.trans(0, 2, "行きたいけど", key=4),
            self.trans(2.1, 3.4, "今は無理だよ", key=5),
            self.trans(3.5, 5, "帰りたいけど", key=5, suspect=True),
            self.trans(5.1, 6.2, "また今度", key=5),
        )
        self.assertEqual(len(spy.japanese), 4)
        self.assertEqual(spy.reliability, [False, False, True, False])

    def test_seek_marker_is_not_swallowed_by_lookahead(self):
        spy, outs = self.run_sequence(
            self.trans(0, 2, "行きたいけど", key=1),
            AudioProgress(sample=10 * 16000, reset=True, generation=1),
            self.trans(10, 12, "無理です", generation=1, key=1),
        )
        self.assertEqual(spy.japanese, ["行きたいけど", "無理です"])
        self.assertEqual(sum(isinstance(x, AudioProgress) for x in outs), 1)

    def test_eos_flushes_incomplete_phrase(self):
        spy, subs = self.run_sequence(self.trans(0, 2, "行きたいけど", key=4))
        self.assertEqual(spy.japanese, ["行きたいけど"])
        self.assertEqual(len(subs), 1)


if __name__ == "__main__":
    unittest.main()
