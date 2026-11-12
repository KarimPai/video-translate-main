"""CPU-only contract tests for Nemotron language routing and subtitle passthrough."""
import io
import json
import queue
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import Mock

import numpy as np

from auto_translate.asr.dual import DualASR
from auto_translate.asr.language_router import classify_language
from auto_translate.asr.nemotron import LanguageTranscript, NemotronASR
from auto_translate.config import Config
from auto_translate.glossary import Glossary
from auto_translate.pipeline import run_asr, run_translate
from auto_translate.types import Subtitle, Transcript, Utterance


class LanguageDecisionTests(unittest.TestCase):
    def test_known_locales(self):
        self.assertEqual(classify_language("en-US", "Where are we going?", 2), "english")
        self.assertEqual(classify_language("en", "Thank you", 0.9), "english")
        self.assertEqual(classify_language("ja-JP", "次に行こう", 2), "japanese")

    def test_uncertain_short_speech_and_mixed_script(self):
        for language, text, duration in (
            (None, "Hello there", 3),
            ("und", "Hello there", 3),
            ("auto", "Hello there", 3),
            ("en", "Eh", 0.3),
            ("en", "Hi", 0.4),
            ("en", "まだ帰らない Hello", 3),
            ("en", "[Music]", 2),
            ("fr-FR", "Bonjour ici", 2),
        ):
            with self.subTest(language=language, text=text):
                self.assertEqual(classify_language(language, text, duration), "uncertain")

    def test_english_followup_can_be_short_but_never_japanese(self):
        self.assertEqual(classify_language("en", "Okay", 0.4, previous_english=True),
                         "english")
        self.assertEqual(classify_language("en", "Okay", 0.4), "uncertain")
        self.assertEqual(classify_language("ja", "OK", 2, previous_english=True),
                         "japanese")


class NemotronLanguageHTTPTests(unittest.TestCase):
    def test_auto_uses_verbose_json_and_returns_language_metadata(self):
        uploads = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                uploads.append(self.rfile.read(int(self.headers["Content-Length"])))
                answer = json.dumps({"text": "Are you okay?", "language": "en-US",
                                     "words": [], "duration": 2.1}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(answer)

            def log_message(self, *args):
                pass

        with HTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.handle_request, daemon=True)
            thread.start()
            worker = object.__new__(NemotronASR)
            worker.url = f"http://127.0.0.1:{server.server_port}"
            worker.process = Mock()
            worker.process.poll.return_value = None
            result = worker.transcribe_with_language(np.ones(32000, dtype=np.float32),
                                                     timeout=2)
            thread.join(2)
        self.assertEqual(result, LanguageTranscript("Are you okay?", "en-US"))
        self.assertIn(b'auto', uploads[0])
        self.assertIn(b'verbose_json', uploads[0])

    def test_missing_response_language_stays_unknown(self):
        # No guessing English from Latin script when a server omits language.
        self.assertEqual(classify_language(None, "Where did you go?", 2), "uncertain")


class PipelineRoutingTests(unittest.TestCase):
    def setup_dual(self, language_text, language):
        cfg = Config()
        cfg.dual_asr.enabled = False  # suppress model startup in unit tests
        dual = DualASR(cfg, Mock(pressure_level="normal"), Glossary({}), False)
        cfg.dual_asr.enabled = True
        worker = Mock(last_stop="unknown")
        worker.transcribe_with_language.return_value = LanguageTranscript(
            language_text, language
        )
        worker.transcribe.return_value = "待って"
        dual.worker = worker
        dual.selector = None
        return dual

    @staticmethod
    def utterance(start=2, end=4):
        return Utterance(start, end,
                         np.ones(int((end-start)*16000), dtype=np.float32) * .1,
                         vad_confidence=.95)

    def test_clear_english_skips_haas_lfm_plamo_and_keeps_timestamps(self):
        dual = self.setup_dual("Where are you going today?", "en-US")
        primary = Mock()
        uq, tq = queue.Queue(), queue.Queue()
        uq.put(self.utterance())
        uq.put(None)
        run_asr(primary, uq, tq, dual=dual)
        source = tq.get_nowait()
        self.assertIsInstance(source, Subtitle)
        self.assertEqual((source.start_s, source.end_s), (2, 4))
        self.assertEqual(source.en, "Where are you going today?")
        self.assertEqual(source.ja, "")
        self.assertIsNone(tq.get_nowait())
        primary.transcribe.assert_not_called()
        dual.worker.transcribe.assert_not_called()

        translator = Mock(memory_lines=3)
        output = queue.Queue()
        tq.put(source)
        tq.put(None)
        run_translate(translator, 4, tq, output)
        self.assertEqual(output.get_nowait().en, source.en)
        self.assertIsNone(output.get_nowait())
        translator.translate.assert_not_called()

    def test_japanese_auto_output_reused_without_second_nemotron_decode(self):
        dual = self.setup_dual("待って", "ja-JP")
        primary = Mock(last_stop="complete")
        primary.transcribe.return_value = "待って"
        uq, tq = queue.Queue(), queue.Queue()
        uq.put(self.utterance())
        uq.put(None)
        run_asr(primary, uq, tq, dual=dual)
        selected = tq.get_nowait()
        self.assertIsInstance(selected, Transcript)
        self.assertEqual(selected.ja, "待って")
        self.assertIsNone(tq.get_nowait())
        primary.transcribe.assert_called_once()
        dual.worker.transcribe_with_language.assert_called_once()
        dual.worker.transcribe.assert_not_called()

    def test_uncertain_english_keeps_forced_japanese_fallback(self):
        dual = self.setup_dual("Eh", "en-US")
        primary = Mock(last_stop="complete")
        primary.transcribe.return_value = "うん"
        uq, tq = queue.Queue(), queue.Queue()
        uq.put(self.utterance(2, 2.4))
        uq.put(None)
        run_asr(primary, uq, tq, dual=dual)
        selected = tq.get_nowait()
        self.assertIsInstance(selected, Transcript)
        self.assertEqual(selected.ja, "うん")
        primary.transcribe.assert_called_once()
        dual.worker.transcribe.assert_called_once()

    def test_language_routing_disabled_keeps_existing_japanese_pipeline(self):
        dual = self.setup_dual("Are you okay?", "en")
        dual.cfg.dual_asr.language_routing = False
        primary = Mock(last_stop="complete")
        primary.transcribe.return_value = "何ですか"
        uq, tq = queue.Queue(), queue.Queue()
        uq.put(self.utterance())
        uq.put(None)
        run_asr(primary, uq, tq, dual=dual)
        self.assertIsInstance(tq.get_nowait(), Transcript)
        dual.worker.transcribe_with_language.assert_not_called()
        dual.worker.transcribe.assert_called_once()

    def test_unknown_auto_metadata_does_not_passthrough(self):
        dual = self.setup_dual("Where are you going?", None)
        primary = Mock(last_stop="complete")
        primary.transcribe.return_value = "どこ行く"
        uq, tq = queue.Queue(), queue.Queue()
        uq.put(self.utterance())
        uq.put(None)
        run_asr(primary, uq, tq, dual=dual)
        self.assertIsInstance(tq.get_nowait(), Transcript)
        primary.transcribe.assert_called_once()
        dual.worker.transcribe.assert_called_once()


if __name__ == "__main__":
    unittest.main()
