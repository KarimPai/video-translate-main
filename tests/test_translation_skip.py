"""Translation sentinel must never appear as an MPV subtitle or history."""

import queue
import unittest

from auto_translate.pipeline import run_translate
from auto_translate.types import Transcript


class SkipTranslationPipelineTests(unittest.TestCase):
    def test_model_skip_drops_only_that_subtitle(self):
        class StubTranslator:
            memory_lines = 0

            def __init__(self):
                self.calls = []

            def translate(self, ja, history):
                self.calls.append((ja, list(history)))
                return "[SKIP]" if ja == "タタタタ" else "The police are here!"

        incoming, outgoing = queue.Queue(), queue.Queue()
        incoming.put(Transcript(0.0, 0.5, "タタタタ"))
        incoming.put(Transcript(0.5, 1.5, "警察が来た"))
        incoming.put(None)
        translator = StubTranslator()
        run_translate(translator, 8, incoming, outgoing)
        result = outgoing.get_nowait()
        self.assertEqual(result.ja, "警察が来た")
        self.assertEqual(result.en, "The police are here!")
        self.assertIsNone(outgoing.get_nowait())
        self.assertEqual(translator.calls[1][1], [])


if __name__ == "__main__":
    unittest.main()
