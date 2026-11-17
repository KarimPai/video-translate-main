"""Per-cue name evidence must survive queues, seeks and optional failures."""
import json
from pathlib import Path
import queue
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from auto_translate.asr.diagnostics import append_comparison, flush_comparisons
from auto_translate.pipeline import run_asr, run_translate
from auto_translate.proper_names import resolve_names
from auto_translate.session_cache import SessionSubtitleCache
from auto_translate.types import AudioProgress, Subtitle, Transcript, Utterance


class NamePipelineTests(unittest.TestCase):
    def run_source(self, texts, reset=False):
        incoming, outgoing = queue.Queue(), queue.Queue()
        for index, _ in enumerate(texts):
            if reset and index:
                incoming.put(AudioProgress(32000, reset=True, generation=1))
            incoming.put(Utterance(index * 2, index * 2 + 1,
                                  np.full(16000, .1, np.float32),
                                  generation=1 if reset and index else 0))
        incoming.put(None)
        asr = SimpleNamespace(transcribe=Mock(side_effect=texts), last_keyword_diagnostics={})
        run_asr(asr, incoming, outgoing, vod_title="ホロライブ まつり みこ")
        return [item for item in outgoing.queue if isinstance(item, Transcript)]

    def test_confirmed_dialogue_survives_across_cues_without_timestamps_changes(self):
        first, second = self.run_source(["夏色まつりちゃんがいる", "まつりが来た"])
        self.assertEqual((second.start_s, second.end_s), (2, 3))
        self.assertEqual(second.ja, "まつりが来た")
        self.assertEqual(second.name_context["mentions"][0]["classification"], "PERSON")
        self.assertTrue(first.name_context["mentions"])

    def test_seek_clears_person_context_and_title_cannot_reestablish_it(self):
        _, second = self.run_source(["夏色まつりちゃんがいる", "まつりが来た"], reset=True)
        self.assertEqual(second.name_context["mentions"][0]["classification"], "UNKNOWN")

    def test_optional_resolver_failure_does_not_lose_later_cues(self):
        with patch("auto_translate.pipeline.resolve_names",
                   side_effect=[RuntimeError("optional failure"), resolve_names("みこちが来た")]):
            first, second = self.run_source(["まつりちゃんが来た", "みこちが来た"])
        self.assertEqual(first.name_context["resolver_error"], "RuntimeError")
        self.assertEqual(second.name_context["mentions"][0]["target"], "Mikochi")

    def test_translation_receives_exact_cue_evidence(self):
        incoming, outgoing = queue.Queue(), queue.Queue()
        source = "祭りちゃんはお祭りに行かない"
        mentions = [m.as_dict() for m in resolve_names(source)]
        incoming.put(Transcript(2, 3, source, name_context={"japanese": source, "mentions": mentions}))
        incoming.put(None)
        translator = SimpleNamespace(memory_lines=0, set_name_context=Mock(),
                                     translate=Mock(return_value="Matsuri-chan isn't going to the festival."))
        run_translate(translator, 3, incoming, outgoing)
        translator.set_name_context.assert_called_once_with(source, mentions)
        result = outgoing.get_nowait()
        self.assertEqual((result.start_s, result.end_s, result.ja), (2, 3, source))
        self.assertIn("isn't", result.en)

    def test_cached_japanese_reestablishes_dialogue_after_seek(self):
        incoming, outgoing = queue.Queue(), queue.Queue()
        cache = SessionSubtitleCache()
        cache.add(Subtitle(0, 1, "夏色まつりがいる", "Natsuiro Matsuri is here."))
        incoming.put(AudioProgress(0, reset=True, generation=1))
        for start in (0, 2):
            incoming.put(Utterance(start, start + 1, np.full(16000, .1, np.float32), generation=1))
        incoming.put(None)
        asr = SimpleNamespace(transcribe=Mock(return_value="まつりが来た"))
        run_asr(asr, incoming, outgoing, session_cache=cache)
        result = [item for item in outgoing.queue if isinstance(item, Transcript)][0]
        self.assertEqual(result.name_context["mentions"][0]["classification"], "PERSON")
        asr.transcribe.assert_called_once()

    def test_failed_name_translation_is_diagnosed_without_losing_next_subtitle(self):
        incoming, outgoing = queue.Queue(), queue.Queue()
        incoming.put(Transcript(0, 1, "まつりちゃんが来た"))
        incoming.put(Transcript(2, 3, "こんにちは"))
        incoming.put(None)
        translator = SimpleNamespace(memory_lines=0, last_name_diagnostics={})
        def translate(source, history):
            if source.startswith("まつり"):
                translator.last_name_diagnostics = {"unresolved_name_issue": "proper_name_mismatch"}
                return ""
            translator.last_name_diagnostics = {}
            return "Hello!"
        translator.translate = translate
        run_translate(translator, 3, incoming, outgoing)
        result = outgoing.get_nowait()
        self.assertEqual((result.start_s, result.en), (2, "Hello!"))
        self.assertIsNone(outgoing.get_nowait())

    def test_raw_audio_and_full_history_cannot_enter_comparison_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.jsonl"
            append_comparison({"stage": "name", "audio": np.ones(100000),
                               "history": ["private prior dialogue"],
                               "mentions": [{"source": "祭り", "classification": "COMMON_NOUN"}],
                               "raw_english": "a" * 100000}, path)
            self.assertTrue(flush_comparisons())
            record = json.loads(path.read_text(encoding="utf8"))
            self.assertNotIn("audio", record)
            self.assertNotIn("history", record)
            self.assertEqual(record["mentions"][0]["classification"], "COMMON_NOUN")
            self.assertLess(path.stat().st_size, 65536)


if __name__ == "__main__":
    unittest.main()
