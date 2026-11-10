"""Long-VOD indexed subtitle cache and diagnostic bound tests.

Offline: MPV and CUDA models are not launched.
"""
from __future__ import annotations

import queue
import threading
import unittest
from collections import deque
from types import SimpleNamespace

from auto_translate.config import MpvConfig
from auto_translate.output.mpv import (
    BOUNDED_DEMUX_CACHE_OPTIONS, MpvOutput,
)
from auto_translate.session_cache import SessionSubtitleCache
from auto_translate.types import Subtitle


class BombList(list):
    def __iter__(self):
        raise AssertionError("Cached VOD renderer must never scan _lines")


class LongVodMpvTests(unittest.TestCase):
    def make_output(self):
        out = MpvOutput.__new__(MpvOutput)
        out.cfg = MpvConfig()
        out.stop = threading.Event()
        out._subtitle_cache = SessionSubtitleCache()
        out._session_start = 0.0
        out._now = 25190.0
        out._lines = BombList()
        out._lock = threading.Lock()
        out._shown = ""
        out._logged_subtitle_generations = set()
        out._logged_displayed_subtitles = set()
        out._lags = deque(maxlen=4096)
        out._subtitle_ready = threading.Event()
        out._live_overlay_confirmed = False
        out.ingest = SimpleNamespace(
            source=SimpleNamespace(live=False),
        )
        commands = []
        out.ipc = SimpleNamespace(command=lambda *args, **kwargs: commands.append((args, kwargs)))
        return out, commands

    def test_cache_profile_keeps_seekable_bounds(self):
        self.assertIn("--cache=yes", BOUNDED_DEMUX_CACHE_OPTIONS)
        self.assertIn("--demuxer-max-bytes=256MiB", BOUNDED_DEMUX_CACHE_OPTIONS)
        self.assertIn("--demuxer-max-back-bytes=64MiB", BOUNDED_DEMUX_CACHE_OPTIONS)
        self.assertNotIn("--demuxer-max-bytes=1GiB", BOUNDED_DEMUX_CACHE_OPTIONS)

    def test_seven_hour_vod_never_scans_future_subtitle_list(self):
        out, commands = self.make_output()
        # A playback moment 10 seconds before a 7-hour video ends.
        cue = Subtitle(25189.0, 25191.0, "ありがとう", "Thank you")
        self.assertTrue(out._subtitle_cache.add(cue))
        out.show(cue)
        self.assertEqual(len(out._lines), 0)
        out._render(25190.0)
        self.assertTrue(commands)
        self.assertIn("Thank you", commands[-1][1]["data"])
        # Seeking back and forth also uses the time index, not _lines.
        out._render(18285.0)
        out._render(25190.0)
        self.assertIn("Thank you", commands[-1][1]["data"])

    def test_diagnostic_state_is_bounded(self):
        out, _ = self.make_output()
        for _ in range(10000):
            out._lags.append(.2)
        self.assertEqual(len(out._lags), 4096)
        out._logged_displayed_subtitles = set(range(8192))
        cue = Subtitle(25189.0, 25191.0, "ありがとう", "Thank you")
        out._subtitle_cache.add(cue)
        out._render(25190.0)
        self.assertLess(len(out._logged_displayed_subtitles), 8192)


if __name__ == "__main__":
    unittest.main()
