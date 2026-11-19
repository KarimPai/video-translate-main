import queue
import unittest
from pathlib import Path
from unittest.mock import Mock

import httpx
import numpy as np

from test_qwen3_asr import fake_asr, response
from auto_translate.asr import asr_context
from auto_translate.config import AsrConfig
from auto_translate.glossary import Glossary
from auto_translate.pipeline import run_asr, run_translate
from auto_translate.types import Utterance


# Verbatim source and original (pre-shortlist) context for RKLF4I3f_gA.
WORDS = "店舗強盗、銀行強盗、宝石店強盗、コンビニ強盗、指名手配、現着、応援、無線、救急隊、犯人、パトカー"
CONTEXT = "人名：みこち、ラプ\n用語：" + WORDS
SPILL = WORDS + "。"
PCM = (.02 * np.sin(np.arange(85920) * .09)).astype(np.float32)


def ja(text):
    return response("language Japanese<asr_text>" + text)


def recognize(asr, *segments, **options):
    inputs, outputs = queue.Queue(), queue.Queue()
    for segment in segments:
        inputs.put(segment)
    inputs.put(None)
    run_asr(asr, inputs, outputs, **options)
    return outputs


class ContextLeakTests(unittest.TestCase):
    def test_reported_cues_recover_once_and_only_recovery_reaches_translation(self):
        asr = fake_asr(ja(SPILL), ja("ちょっと待って。"), ja(SPILL), ja("無線聞こえる？"), context=CONTEXT)
        segments = [Utterance(0, 5.37, PCM, speech_start_s=.01, vad_confidence=.8),
                    Utterance(127.74, 142.41, np.resize(PCM, 234720),
                              speech_start_s=127.74, vad_confidence=.7)]
        with self.assertLogs("auto_translate.pipeline", "INFO") as logs:
            transcripts = recognize(asr, *segments)
        self.assertEqual(asr._request.call_count, 4)
        for first, retry in zip(asr._request.call_args_list[::2], asr._request.call_args_list[1::2]):
            self.assertEqual(first.args, retry.args)  # Identical WAV and duration.
            self.assertIn(WORDS, first.kwargs["system"])
            self.assertEqual(retry.kwargs["system"], "")
            self.assertLessEqual(retry.kwargs["timeout"], first.kwargs["timeout"])
        self.assertIn("action=retry_without_context", logs.output[0])
        self.assertIn("outcome=accepted", logs.output[1])
        self.assertNotIn(WORDS, " ".join(logs.output))
        translator = Mock(cfg=None)
        translator.translate.side_effect = ["Wait a second.", "Can you hear the radio?"]
        out = queue.Queue()
        run_translate(translator, 0, transcripts, out)
        self.assertEqual([c.args[0] for c in translator.translate.call_args_list],
                         ["ちょっと待って。", "無線聞こえる？"])
        for sub, original in zip(out.queue, segments):
            self.assertEqual((sub.start_s, sub.end_s, sub.speech_start_s, sub.generation),
                             (original.start_s, original.end_s, original.speech_start_s, original.generation))

    def test_partial_reordered_and_punctuationless_lists(self):
        for text in ("銀行強盗、宝石店強盗、コンビニ強盗。", "犯人, 無線, 現着。", WORDS.replace("、", "")):
            with self.subTest(text=text):
                asr = fake_asr(ja(text), response("language None<asr_text>"), context=CONTEXT)
                self.assertEqual(list(recognize(asr, Utterance(0, 5.37, PCM)).queue), [None])
                self.assertEqual(asr._request.call_count, 2)

    def test_full_original_list_echo_is_caught_with_current_bounded_context(self):
        title = "GTA"
        glossary = Glossary.load(Path("glossary.toml")).for_stream(title)
        context = asr_context(AsrConfig(), glossary, title)
        hints = context.split("用語：")[1].split("、")
        self.assertEqual(len(hints), 4)
        self.assertGreaterEqual(sum(word in WORDS for word in hints), 3)
        asr = fake_asr(ja(SPILL), ja("ちょっと待って。"), context=context)
        out = recognize(asr, Utterance(0, 5.37, PCM, vad_confidence=.8))
        self.assertEqual(asr._request.call_count, 2)
        self.assertEqual(out.get_nowait().ja, "ちょっと待って。")

    def test_conversation_and_nicknames_use_one_decode(self):
        for text in ("銀行強盗だ！", "無線で救急隊に応援を頼んで。犯人はまだ店の中！",
                     "やばい", "えぐい", "ラプ", "みこち", "あはは！"):
            with self.subTest(text=text):
                asr = fake_asr(ja(text), context=CONTEXT)
                self.assertEqual(asr.transcribe(PCM), text)
                self.assertEqual(asr._request.call_count, 1)

    def test_spoken_list_corroborated_on_speech_can_pass(self):
        asr = fake_asr(ja(SPILL), ja(WORDS + "!"), context=CONTEXT)
        out = recognize(asr, Utterance(0, 5.37, PCM, vad_confidence=.78))
        self.assertEqual(out.get_nowait().ja, WORDS + "!")

    def test_weak_or_unconfirmed_recovery_never_publishes(self):
        for pcm, second, confidence in ((PCM * 0, "はい", .8), (PCM * .001, SPILL, .7),
                                        (PCM, SPILL, .24), (PCM, "救急隊、現着、犯人。", .9)):
            with self.subTest(second=second, confidence=confidence):
                asr = fake_asr(ja(SPILL), ja(second), context=CONTEXT)
                self.assertEqual(list(recognize(asr, Utterance(0, 5.37, pcm, vad_confidence=confidence)).queue),
                                 [None])

    def test_weak_repeat_after_unrelated_audio_with_history_prompt_disabled(self):
        text = "パトカーがもうすぐ到着するよ。"
        asr = fake_asr(ja(text), ja("了解"), ja(text), ja(text), cfg=AsrConfig(recent_context_lines=0))
        out = recognize(asr, Utterance(0, 5.37, PCM), Utterance(8, 13.37, PCM),
                        Utterance(20, 25.37, PCM * 0))
        self.assertEqual([t.ja for t in out.queue if t], [text, "了解"])
        self.assertEqual(asr._request.call_count, 4)

    def test_recovery_error_or_disabled_retry_cannot_restart_transport_loop(self):
        for limit in (0, 1):
            asr = fake_asr(ja(SPILL), httpx.ReadTimeout("timeout"), context=CONTEXT,
                           cfg=AsrConfig(context_retry_limit=limit))
            self.assertEqual(list(recognize(asr, Utterance(0, 5.37, PCM)).queue), [None])
            self.assertEqual(asr._request.call_count, 1 + limit)
        with self.assertRaises(ValueError):
            AsrConfig(context_retry_limit=2)

    def test_live_expiry_before_retry(self):
        asr = fake_asr(ja(SPILL), context=CONTEXT)
        clock = Mock(side_effect=[1, 1, 20, 20])
        out = recognize(asr, Utterance(0, 5.37, PCM), live=True, playback_position=clock)
        self.assertEqual(list(out.queue), [None])
        self.assertEqual(asr._request.call_count, 1)
