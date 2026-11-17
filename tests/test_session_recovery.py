"""Session recovery regressions with simulated process and transport failures."""
import io
import queue
import threading
import unittest
from unittest.mock import Mock, patch

from auto_translate.config import IngestConfig
from auto_translate.ingest import Ingest, Source
from auto_translate.llama_server import LlamaServerError
from auto_translate.output.mpv import MpvError, MpvOutput
from auto_translate.pipeline import run_translate
from auto_translate.translate.base import TranslatorUnavailable
from auto_translate.types import AudioProgress, SAMPLE_RATE, SourceEOF, Transcript
from test_english_reliability import fake_translator


class SessionRecoveryTests(unittest.TestCase):
    def test_owned_server_restarts_once_and_replays_same_cue(self):
        translator = fake_translator(LlamaServerError("closed"), "Hello")
        self.addCleanup(translator.close)
        translator.server.alive.side_effect = [True, False, False]
        translator.reload = Mock(side_effect=[None, LlamaServerError("dead")])
        self.assertEqual(translator.translate("こんにちは", []), "Hello")
        first, retry = translator._chat.call_args_list
        self.assertEqual(first, retry)
        translator.reload.assert_called_once_with(translator.gpu_layers - 8, "closed")
        with self.assertRaisesRegex(TranslatorUnavailable, "could not restart"):
            translator.translate("次", [])
        self.assertEqual(translator.reload.call_count, 2)
        self.assertEqual(translator._chat.call_count, 2)

    def test_server_outage_surfaces_despite_audio_progress(self):
        translator = fake_translator()
        self.addCleanup(translator.close)
        translator._chat = Mock(side_effect=LlamaServerError("down"))
        translator.reload = Mock()
        inputs = queue.Queue()
        for i in range(3):
            inputs.put(Transcript(i, i + 1, "テスト"))
            inputs.put(AudioProgress(i * SAMPLE_RATE))
        inputs.put(None)
        outputs = queue.Queue()
        with self.assertRaisesRegex(TranslatorUnavailable, "three consecutive"):
            run_translate(translator, 0, inputs, outputs)
        self.assertEqual(translator._chat.call_count, 3)
        self.assertEqual(list(outputs.queue), [AudioProgress(0), AudioProgress(SAMPLE_RATE)])
        translator.reload.assert_not_called()

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

    def test_stopping_wakes_ingest_parked_after_vod_eof(self):
        ingest = Ingest(Source("synthetic.wav", False), IngestConfig())
        ingest.enable_vod_restart()
        outputs = queue.Queue()
        worker = threading.Thread(target=ingest.run, args=(outputs,), daemon=True)
        self.addCleanup(ingest.stop)
        with patch.object(ingest, "_read_session", return_value=False) as read:
            worker.start()
            try:
                self.assertEqual(outputs.get(timeout=3), SourceEOF(0))
                self.assertTrue(worker.is_alive())
            finally:
                ingest.stop()
                worker.join(timeout=3)
        self.assertFalse(worker.is_alive())
        self.assertIsNone(outputs.get_nowait())
        self.assertTrue(outputs.empty())
        read.assert_called_once()

    def test_no_player_and_live_eof_remain_terminal(self):
        for live in (False, True):
            with self.subTest(live=live):
                ingest = Ingest(Source("synthetic.wav", live), IngestConfig())
                if live:
                    ingest.enable_vod_restart()
                self.addCleanup(ingest.stop)
                outputs = queue.Queue()
                with patch.object(ingest, "_read_session", return_value=False):
                    ingest.run(outputs)
                self.assertEqual(list(outputs.queue), [None])

    def test_stale_eof_cannot_mark_new_seek_translation_ready(self):
        player = object.__new__(MpvOutput)
        player.ingest = Mock(generation=1)
        player._buffer_condition = threading.Condition()
        player._source_finished = False
        player._now = 20.0
        player._translated_until = 20.0
        player._subtitle_cache = None
        player.finish(0)
        self.assertFalse(player._translation_ready())
        player.finish(1)
        self.assertTrue(player._translation_ready())
        player.ingest.generation = 2
        player.reset_buffer(40.0)
        player._now = 40.0
        player.finish(1)
        self.assertFalse(player._translation_ready())
        player.finish(2)
        self.assertTrue(player._translation_ready())

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
