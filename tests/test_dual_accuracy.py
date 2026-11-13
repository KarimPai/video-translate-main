import queue
import unittest
from unittest.mock import Mock, patch

import numpy as np

from auto_translate.asr.dual import DualASR
from auto_translate.asr.selection import clause_differences, compare, scene_repair
from auto_translate.config import Config, VadConfig, vad_for_source
from auto_translate.glossary import Glossary
from auto_translate.language_filter import LanguageDecision
from auto_translate.pipeline import run_asr, run_translate
from auto_translate.text_filter import rejection_reason
from auto_translate.translate.plamo import _current_glossary_matches
from auto_translate.types import Utterance
from auto_translate.vad import FRAME
from test_vad_longform import LongformBoundaryTests as _BoundaryHelper

# Import just the fixture function; unittest discovers module-level TestCase types.
process_pcm = _BoundaryHelper._process
del _BoundaryHelper


class AccuracyTests(unittest.TestCase):
    def test_peer_clause_and_topic_required(self):
        for before, after, anchor in [("子宮", "支給", "制服"), ("研究隊", "救急隊", "搬送"),
                                      ("無銭", "無線", "周波数")]:
            a = f"{before}じゃないね。あと2個。"
            b = f"{after}じゃないね。あと3個。"
            wanted = f"{after}じゃないね。あと2個。"
            repaired, edits = scene_repair(a, b, "ホロGTA", [anchor])
            self.assertEqual(repaired, wanted)
            self.assertEqual(edits[0]["peer_clause"], b.split("。")[0] + "。")
            for peer, title, history in [(a, "ホロGTA", [anchor]), (b, "健康相談", [anchor]),
                                          (b, "ホロGTA", []), (b.replace("じゃない", "だ"), "GTA", [anchor])]:
                self.assertEqual(scene_repair(a, peer, title, history), (a, []))

    def test_clause_alignment_and_protected_details(self):
        self.assertEqual(clause_differences("待って。戻る。", "待って。無線で。戻る。"),
                         [{"haas": "", "whisper": "無線で。"}])
        for a, b, label in [("私が", "私を", "particle"), ("2台", "3台", "number"),
                            ("行く", "行かない", "negation"), ("わため", "わたなべ", "glossary")]:
            self.assertIn(label, compare(a, b, [("わため", "Watame")])[1])

    def test_short_reactions_names_and_spoken_laughter(self):
        for spoken in ("えっ", "うん", "あはは", "フワモコ", "オパンパンカオン"):
            self.assertIsNone(rejection_reason(spoken))
        self.assertIsNotNone(rejection_reason("[laughter]"))
        self.assertIsNotNone(rejection_reason("ダイジョブ" * 8))

    def test_equipment_glossary_requires_current_source_and_title(self):
        glossary = Glossary({})
        self.assertIn(("無線機", "radio"), _current_glossary_matches("無線機だ", "ホロGTA", glossary))
        for ja, title in [("無線機だ", "雑談"), ("無銭だ", "ホロGTA")]:
            self.assertEqual(_current_glossary_matches(ja, title, glossary), [])

    def test_live_and_vod_turn_pause_preserves_pcm_and_media_time(self):
        for live, gap, frames in [(True, 450, 680), (False, 700, 1100)]:
            with self.subTest(live=live):
                pcm = np.full(frames * FRAME, .2, dtype=np.float32)
                probs = np.full(frames, .95)
                pcm[gap*FRAME:(gap+20)*FRAME] = 0
                pcm[(gap+20)*FRAME:] = -.15
                probs[gap:gap+20] = .03
                cues = process_pcm(pcm, probs, vad_for_source(VadConfig(), live=live))
                self.assertEqual(len(cues), 2)
                self.assertAlmostEqual(cues[0].end_s, cues[1].start_s)
                self.assertAlmostEqual(cues[-1].end_s, frames / 100)
                np.testing.assert_array_equal(np.concatenate([c.audio for c in cues]), pcm)

    def test_en_id_excluded_before_asr(self):
        for language in ("en", "id"):
            iq, oq, primary = queue.Queue(), queue.Queue(), Mock()
            iq.put(Utterance(0, 4, np.ones(64000)))
            iq.put(None)
            lid = Mock()
            lid.classify.return_value = LanguageDecision(language, .98, .01, True, "confident")
            run_asr(primary, iq, oq, language_filter=lid)
            primary.transcribe.assert_not_called()
            self.assertEqual(list(oq.queue), [None])

    def test_stage_provenance_and_presentation_timestamps(self):
        cfg = Config()
        cfg.dual_asr.enabled = False
        d = DualASR(cfg, Mock(pressure_level="normal"), Glossary({}), False)
        d.worker = Mock(last_stop="complete")
        d.selector = Mock(last_choice="haas")
        a = "制服は子宮じゃないね。あと2個。"
        b = "制服は支給じゃないね。あと3個。"
        d.worker.transcribe.return_value = b
        d.selector.choose.return_value = a
        iq, mq, oq = queue.Queue(), queue.Queue(), queue.Queue()
        iq.put(Utterance(1, 5, np.full(64000, .2), vad_confidence=.95))
        iq.put(None)
        primary, translator = Mock(last_stop="complete"), Mock(memory_lines=0)
        primary.transcribe.return_value = a
        translator.translate.return_value = "Uniforms aren't issued. Two more."
        with patch("auto_translate.asr.dual.append_comparison") as comparison, patch(
                "auto_translate.pipeline.append_comparison") as stages:
            run_asr(primary, iq, mq, vod_title="ホロGTA", dual=d)
            run_translate(translator, 4, mq, oq)
        decision = comparison.call_args.args[0]
        translated = stages.call_args.args[0]
        cue = oq.get_nowait()
        self.assertEqual(decision["lfm_selected"], a)
        self.assertEqual(decision["corrected_ja"], a.replace("子宮", "支給"))
        self.assertEqual(translated["final_ja"], decision["corrected_ja"])
        self.assertEqual((translated["final_en"], cue.start_s, cue.end_s),
                         (translator.translate.return_value, 1, 5))

    def test_model_limited_prefix_and_credit_fallback(self):
        cfg = Config()
        cfg.dual_asr.enabled = False
        dual = DualASR(cfg, Mock(pressure_level="normal"), Glossary({}), False)
        dual.worker = Mock(last_stop="complete")
        dual.selector = Mock()
        utt = Utterance(0, 4, np.full(64000, .1), vad_confidence=.9)
        for haas, whisper, stop, expected in [
                ("無線機を", "無線機を持ってきて", "model_limit", "無線機を持ってきて"),
                ("by H.", "無線機を持ってきて", "complete", "無線機を持ってきて"),
                ("by H.", "by H.", "complete", "by H.")]:
            dual.worker.transcribe.return_value = whisper
            with patch("auto_translate.asr.dual.append_comparison"):
                self.assertEqual(dual.choose(utt, haas, "", lambda: False, None, stop), expected)
        dual.selector.choose.assert_not_called()
