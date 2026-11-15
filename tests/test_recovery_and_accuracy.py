import os
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from auto_translate.asr.quality_guard import weak_outro
from auto_translate.captions import CaptionMatch
from auto_translate.glossary import Glossary
from auto_translate.ingest import _prune_abandoned_sessions
from auto_translate.output.mpv import MpvOutput
from auto_translate.pipeline import run_asr
from auto_translate.session_cache import SessionSubtitleCache
from auto_translate.text_filter import trim_repetition_tail
from auto_translate.translation_guard import suspicious_translation
from auto_translate.types import Utterance


class AccuracyTests(unittest.TestCase):
    def test_repetition_suffix_and_real_emphasis(self):
        for tail in ("あ" * 20, "石" * 20, "猫です" * 9):
            self.assertEqual(trim_repetition_tail("そこにいる。" + tail), "そこにいる。")
        for text in ("待って待って待って", "まだまだまだ", "そうです。" + "はい" * 10):
            self.assertEqual(trim_repetition_tail(text), text)

    def test_outro_evidence(self):
        phrase = "ご視聴ありがとうございました"
        audio = np.full(16000, .001)
        self.assertTrue(weak_outro(phrase, "", 1., .45, audio))
        self.assertFalse(weak_outro(phrase, phrase, 1., .45, audio))
        self.assertFalse(weak_outro(phrase, "", 3., .9, audio))
        self.assertFalse(weak_outro(phrase, "", 1., .45, audio * 100))

    def test_short_glossary_term_preserves_compounds(self):
        glossary = Glossary({"敵": "enemy", "石": "gem", "草": "LOL"})
        self.assertEqual(glossary.find("素敵な草原の化石"), [])
        self.assertEqual(glossary.find("敵がいる"), [("敵", "enemy")])

    def test_new_negation_signals_are_log_only(self):
        for text in ("今日は行かなかった。", "何も食べずに帰る", "今日は行きません"):
            with self.assertLogs("auto_translate.translation_guard", "INFO"):
                self.assertIsNone(suspicious_translation(text, "I went home."))
        with self.assertNoLogs("auto_translate.translation_guard"):
            self.assertIsNone(suspicious_translation("必ず行きます", "I will go."))

    def test_empty_audio_transcript_requires_creator_caption(self):
        for manual in (True, False):
            with self.subTest(manual=manual):
                audio = queue.Queue()
                output = queue.Queue()
                audio.put(Utterance(0, 2, np.ones(32000), vad_confidence=.9))
                audio.put(None)
                asr, captions = Mock(), Mock()
                asr.transcribe.return_value = ""
                captions.match.return_value = CaptionMatch("まだいるよ", 2., 1., True, manual)
                run_asr(asr, audio, output, captions=captions)
                result = output.get_nowait()
                self.assertEqual(result.ja if result else None, "まだいるよ" if manual else None)


class RecoveryTests(unittest.TestCase):
    def test_restart_from_first_uncovered_point(self):
        p = object.__new__(MpvOutput)
        p.ipc, p.ingest = Mock(), Mock()
        p._lock = threading.Lock()
        p._lines = []
        p._uses_original_timeline = True
        p._subtitle_cache = SessionSubtitleCache()
        p._subtitle_cache.record_processed(10, 12)
        p.reset_buffer = Mock()
        p._render = Mock()
        for seek in (10., 11.5, 11.6):
            p._restart_audio_at(seek)
            p.ingest.request_restart.assert_called_with(12)
            self.assertEqual(p._session_start, seek)

    def test_prune_requires_identifiable_dead_owner(self):
        with tempfile.TemporaryDirectory() as root:
            parent = Path(root)
            stale = parent / "auto-translate-fmp4-pid123-aaaaaaaa"
            live = parent / f"auto-translate-fmp4-pid{os.getpid()}-bbbbbbbb"
            unknown = parent / "auto-translate-fmp4-unknown"
            for folder in (stale, live, unknown):
                folder.mkdir()
            with patch("auto_translate.ingest._owner_gone", side_effect=lambda pid: pid != os.getpid()):
                _prune_abandoned_sessions(parent)
            self.assertFalse(stale.exists())
            self.assertTrue(live.exists())
            self.assertTrue(unknown.exists())
