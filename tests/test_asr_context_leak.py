import queue
import unittest
from pathlib import Path
from unittest.mock import Mock

import httpx
import numpy as np

from test_haas_asr import fake_asr
from auto_translate.asr import asr_context
from auto_translate.config import AsrConfig
from auto_translate.glossary import Glossary
from auto_translate.pipeline import run_asr, run_translate
from auto_translate.types import Utterance


# Verbatim source and original (pre-shortlist) context for RKLF4I3f_gA.
WORDS = "店舗強盗、銀行強盗、宝石店強盗、コンビニ強盗、指名手配、現着、応援、無線、救急隊、犯人、パトカー"
CONTEXT = "人名：みこち、ラプ\n用語：" + WORDS
SPILL = WORDS + "!"
PCM = (.02 * np.sin(np.arange(85920) * .09)).astype(np.float32)


def recognize(asr, *segments, **options):
    inputs, outputs = queue.Queue(), queue.Queue()
    for segment in segments:
        inputs.put(segment)
    inputs.put(None)
    run_asr(asr, inputs, outputs, **options)
    return outputs


class ContextLeakTests(unittest.TestCase):
    def test_conversation_and_nicknames_use_one_decode(self):
        for text in ("銀行強盗だ！", "無線で救急隊に応援を頼んで。犯人はまだ店の中！",
                     "やばい", "えぐい", "ラプ", "みこち", "あはは！"):
            with self.subTest(text=text):
                asr = fake_asr(text, context=CONTEXT)
                self.assertEqual(asr.transcribe(PCM), text)

    def test_spoken_list_corroborated_on_speech_can_pass(self):
        asr = fake_asr(SPILL, WORDS + "!", context=CONTEXT)
        out = recognize(asr, Utterance(0, 5.37, PCM, vad_confidence=.78))
        self.assertEqual(out.get_nowait().ja, WORDS + "!")

    def test_weak_repeat_after_unrelated_audio_with_history_prompt_disabled(self):
        text = "パトカーがもうすぐ到着するよ。"
        asr = fake_asr(text, "了解", "", cfg=AsrConfig(recent_context_lines=0))
        out = recognize(asr, Utterance(0, 5.37, PCM), Utterance(8, 13.37, PCM),
                        Utterance(20, 25.37, PCM * 0))
        self.assertEqual([t.ja for t in out.queue if t], [text, "了解"])

    def test_context_built_from_glossary(self):
        title = "GTA"
        glossary = Glossary.load(Path("glossary.toml")).for_stream(title)
        context = asr_context(AsrConfig(), glossary, title)
        self.assertIn("用語", context)
        hints = context.split("用語：")[1].split("、")
        self.assertEqual(len(hints), 4)
        self.assertGreaterEqual(sum(word in WORDS for word in hints), 3)
