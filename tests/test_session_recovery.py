"""Session recovery regressions with simulated process and transport failures."""
import io
import queue
import threading
import unittest
from unittest.mock import Mock, patch

import httpx

from auto_translate.config import IngestConfig, TranslateConfig
from auto_translate.glossary import Glossary
from auto_translate.ingest import Ingest, Source
from auto_translate.output.mpv import MpvError, MpvOutput
from auto_translate.pipeline import run_translate
from auto_translate.translate.base import TranslatorUnavailable
from auto_translate.translate.plamo import PlamoTranslator
from auto_translate.types import AudioProgress, SAMPLE_RATE, Transcript


class SessionRecoveryTests(unittest.TestCase):
    def translator(self, process):
        with patch.object(PlamoTranslator, "_start_server"), patch.object(PlamoTranslator, "_warmup"):
            translator = PlamoTranslator(TranslateConfig(), Glossary({}))
        translator.process = process
        self.addCleanup(translator.close)
        return translator

    def test_owned_server_restarts_once_and_replays_same_cue(self):
        translator = self.translator(Mock())
        translator.process.poll.return_value = 1
        def restart():
            translator.process = Mock()
            translator.process.poll.return_value = 1
        translator._start_server = Mock(side_effect=restart)
        translator._translate_cue = Mock(side_effect=[httpx.ReadError("closed"), "Hello", httpx.ConnectError("dead")])
        self.assertEqual(translator.translate("こんにちは", []), "Hello")
        first, retry = translator._translate_cue.call_args_list
        self.assertEqual(first, retry)
        with self.assertRaises(TranslatorUnavailable):
            translator.translate("次", [])
        translator._start_server.assert_called_once()

    def test_external_server_outage_surfaces_despite_audio_progress(self):
        translator = self.translator(None)
        translator._start_server = Mock()
        translator._translate_cue = Mock(side_effect=httpx.ConnectError("down"))
        inputs = queue.Queue()
        for i in range(3):
            inputs.put(Transcript(i, i + 1, "テスト"))
            inputs.put(AudioProgress(i * SAMPLE_RATE))
        inputs.put(None)
        with self.assertRaisesRegex(TranslatorUnavailable, "three consecutive"):
            run_translate(translator, 0, inputs, queue.Queue())
        translator._start_server.assert_not_called()

    def test_vod_midstream_error_refreshes_and_keeps_pcm_contiguous(self):
        ingest = Ingest(Source("https://example.test/video", False, start_s=3), IngestConfig())
        commands = []
        def spawn(cmd, **kwargs):
            commands.append(cmd)
            proc = Mock()
            proc.returncode = 1 if len(commands) == 1 else 0
            proc.stdout = io.BytesIO(bytes(1600))
            return proc
        outputs = queue.Queue()
        with patch.object(ingest, "_resolve_audio_url", side_effect=["url1", "url2"]), patch(
                "auto_translate.ingest.owned_popen", side_effect=spawn), patch("auto_translate.ingest.time.sleep"):
            ingest.run(outputs)
        self.assertEqual([cmd[cmd.index("-i")+1] for cmd in commands], ["url1", "url2"])
        self.assertIn("-rw_timeout", commands[0])
        start = 3 * SAMPLE_RATE
        self.assertEqual([outputs.get_nowait().start_sample for _ in commands], [start, start + 800])
        self.assertAlmostEqual(float(commands[1][commands[1].index("-ss")+1]), 3 + 800 / SAMPLE_RATE)
        self.assertIsNone(outputs.get_nowait())
        self.assertEqual(ingest.source.start_s, 3)

    def test_vod_blocked_read_is_killed_and_clean_eof_ends_retry(self):
        ingest = Ingest(Source("https://example.test/video", False), IngestConfig())
        blocked, eof = Mock(), Mock()
        killed = threading.Event()
        def read(_size):
            if not killed.wait(3):
                self.fail("watchdog did not interrupt the read")
            return b""
        blocked.stdout.read.side_effect = read
        blocked.kill.side_effect = killed.set
        blocked.returncode = 1
        eof.stdout = io.BytesIO()
        eof.returncode = 0
        with patch.object(ingest, "_resolve_audio_url", return_value="audio"), patch(
                "auto_translate.ingest.owned_popen", side_effect=[blocked, eof]) as spawn, patch(
                "auto_translate.ingest.PCM_READ_TIMEOUT_S", 0.01):
            ingest.run(queue.Queue())
        blocked.kill.assert_called_once()
        self.assertEqual(spawn.call_count, 2)
        self.assertFalse(ingest.video_failed.is_set())

    def test_transient_pause_rejection_preserves_intent_and_disconnect_propagates(self):
        player = object.__new__(MpvOutput)
        player.stop = threading.Event()
        player.ipc = Mock()
        player.ipc.get.return_value = False
        player._user_paused = False
        player.ipc.command.side_effect = [MpvError("rejected"), None, ConnectionError("closed")]
        player._toggle_user_pause()
        self.assertFalse(player._user_paused)
        self.assertFalse(player.stop.is_set())
        player._toggle_user_pause()
        self.assertTrue(player._user_paused)
        with self.assertRaises(ConnectionError):
            player._toggle_user_pause()
