"""Offline regressions for live backlog shedding and stalled Haas pipes."""
import queue
import subprocess
import sys
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from auto_translate.asr.haas import HaasASR
from auto_translate.config import Config
from auto_translate.pipeline import run_asr, run_translate
from auto_translate.types import Transcript, Utterance


class ExpiredLiveWorkTests(unittest.TestCase):
    def test_asr_skips_only_cues_already_past_player(self):
        primary = Mock()
        primary.transcribe.return_value = "待って"
        optional = Mock()
        optional.playback_position = lambda: 20.0
        input_q, output_q = queue.Queue(), queue.Queue()
        input_q.put(Utterance(2, 3, np.ones(16000, dtype=np.float32)))
        input_q.put(None)
        run_asr(primary, input_q, output_q, live=True, dual=optional,
                live_late_window_s=7.0)
        primary.transcribe.assert_not_called()
        self.assertEqual(list(output_q.queue), [None])

    def test_mismatched_mpv_clock_never_discards_all_live_speech(self):
        primary = Mock()
        primary.transcribe.return_value = "待って"
        optional = Mock()
        optional.playback_position = lambda: 15260.0
        optional.select_segment.return_value = None
        input_q, output_q = queue.Queue(), queue.Queue()
        input_q.put(Utterance(2, 3, np.ones(16000, dtype=np.float32)))
        input_q.put(None)
        run_asr(primary, input_q, output_q, live=True, dual=optional,
                live_late_window_s=7.0)
        primary.transcribe.assert_called_once()
        self.assertEqual(len(output_q.queue), 2)

    def test_vod_keeps_older_cues(self):
        primary = Mock()
        primary.transcribe.return_value = "待って"
        input_q, output_q = queue.Queue(), queue.Queue()
        input_q.put(Utterance(2, 3, np.ones(16000, dtype=np.float32)))
        input_q.put(None)
        run_asr(primary, input_q, output_q, live=False)
        primary.transcribe.assert_called_once()
        self.assertEqual(len(output_q.queue), 2)

    def test_plamo_skips_expired_live_transcript(self):
        translator = Mock()
        translator.memory_lines = 0
        input_q, output_q = queue.Queue(), queue.Queue()
        input_q.put(Transcript(2, 3, "待って"))
        input_q.put(None)
        run_translate(translator, 0, input_q, output_q, live=True,
                      playback_position=lambda: 20.0, live_late_window_s=7.0)
        translator.translate.assert_not_called()
        self.assertEqual(list(output_q.queue), [None])

    def test_uninitialized_player_preserves_cues(self):
        translator = Mock()
        translator.memory_lines = 0
        translator.translate.return_value = "Wait."
        input_q, output_q = queue.Queue(), queue.Queue()
        input_q.put(Transcript(2, 3, "待って"))
        input_q.put(None)
        run_translate(translator, 0, input_q, output_q, live=True,
                      playback_position=lambda: None, live_late_window_s=7.0)
        translator.translate.assert_called_once()
        self.assertEqual(len(output_q.queue), 2)


class StalledPipeTests(unittest.TestCase):
    def test_stdin_deadline_applies_when_child_does_not_read(self):
        # Large PCM saturates the OS pipe while the child remains alive.
        script = "import json, time; print(json.dumps({'type': 'ready'}), flush=True); time.sleep(5)"
        real_spawn = subprocess.Popen

        def fake_spawn(cmd, **kwargs):
            return real_spawn([sys.executable, "-u", "-c", script], **kwargs)

        cfg = Config().asr
        cfg.response_timeout_s = 0.15
        cfg.startup_timeout_s = 3.0
        with patch("auto_translate.asr.haas.shutil.which", return_value="speech"), patch(
                "auto_translate.asr.haas.subprocess.Popen", side_effect=fake_spawn):
            worker = HaasASR(cfg)
            try:
                start = time.monotonic()
                with self.assertRaises(TimeoutError):
                    worker.transcribe(np.ones(40000, dtype=np.float32))
                self.assertLess(time.monotonic() - start, 1.0)
            finally:
                worker.close()


class SelectorPressureTests(unittest.TestCase):
    def test_soft_pressure_releases_only_gpu_selector(self):
        import threading
        from auto_translate.asr.dual import DualASR
        dual = object.__new__(DualASR)
        dual.lock = threading.Lock()
        selector = Mock(gpu_layers=0)
        dual.selector = selector
        dual.release_selector()
        selector.close.assert_not_called()
        self.assertIs(dual.selector, selector)

        selector.gpu_layers = 99
        dual.release_selector()
        selector.close.assert_called_once()
        self.assertIsNone(dual.selector)


class BackpressureTests(unittest.TestCase):
    def test_vod_bypasses_optional_when_primary_usable_and_backlogged(self):
        from auto_translate.asr.dual import DualASR
        from auto_translate.glossary import Glossary
        cfg = Config()
        cfg.dual_asr.enabled = False
        dual = DualASR(cfg, Mock(used=3., cfg=cfg.gpu_budget), Glossary({}), False)
        dual.worker = Mock()
        utt = Utterance(2, 3, np.ones(16000, dtype=np.float32), vad_confidence=0.9)
        result = dual.select_segment(utt, "待って", "", lambda: True, lambda: 0)
        self.assertIsNone(result)
        dual.worker.transcribe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
