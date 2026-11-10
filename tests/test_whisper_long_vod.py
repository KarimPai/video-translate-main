"""Long-VOD rendering and transcription fidelity regression tests.

Offline: MPV, CUDA and llama-server are not launched.
"""
from __future__ import annotations

import queue
import threading
import unittest
from collections import deque
from types import SimpleNamespace

from auto_translate.asr_quality import (
    is_standalone_laughter, formulaic_japanese_outro_key,
)
from auto_translate.config import MpvConfig, TranslateConfig
from auto_translate.output.mpv import (
    BOUNDED_DEMUX_CACHE_OPTIONS, MpvOutput,
)
from auto_translate.session_cache import SessionSubtitleCache
from auto_translate.translation_integrity import contextual_translation_hints
from auto_translate.two_stage import run_plamo_translate, run_whisper_asr
from auto_translate.types import AudioProgress, Subtitle, Transcript, Utterance
from auto_translate.whisper_v3 import WhisperDecode
import numpy as np


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


class ModelFidelityTests(unittest.TestCase):
    def test_standalone_laughter_only(self):
        self.assertTrue(is_standalone_laughter("あははは…"))
        self.assertTrue(is_standalone_laughter("ふふ…"))
        self.assertFalse(is_standalone_laughter("そうだね、あはは"))
        self.assertFalse(is_standalone_laughter("え、すごい！"))

    def test_outro_key_is_punctuation_tolerant(self):
        self.assertEqual(
            formulaic_japanese_outro_key("ご視聴ありがとうございました。"),
            "ご視聴ありがとうございました",
        )
        self.assertIsNone(
            formulaic_japanese_outro_key("ご視聴ありがとうございました、またね"),
        )

    def test_homophone_hints_require_context(self):
        self.assertTrue(contextual_translation_hints(
            "無線マウスに慣れると遊星に戻れない"))
        self.assertFalse(contextual_translation_hints("遊星の風景"))
        self.assertTrue(contextual_translation_hints("車が弾付き事故になった"))
        self.assertFalse(contextual_translation_hints("弾付き事故"))
        self.assertTrue(contextual_translation_hints(
            "強姦", recent_fragments=("交換屋で渡した",)))
        self.assertFalse(contextual_translation_hints("強姦"))

    def test_translator_preserves_raw_katakana_and_timestamps(self):
        class FakeTranslator:
            cfg = TranslateConfig()
            def translate(self, ja, history, *, fragment_context=()):
                return "Translation: " + ja

        q_in, q_out = queue.Queue(), queue.Queue()
        q_in.put(Transcript(1.0, 2.0, "シャッター"))
        q_in.put(Transcript(2.1, 2.5, "シャッタ"))
        q_in.put(None)
        run_plamo_translate(FakeTranslator(), q_in, q_out)
        first, second = q_out.get_nowait(), q_out.get_nowait()
        self.assertEqual(first.ja, "シャッター")
        self.assertEqual(second.ja, "シャッタ")
        self.assertEqual(second.start_s, 2.1)
        self.assertEqual(second.en, "Translation: シャッタ")

    def test_laughter_filter_resets_on_seek(self):
        class FakeWhisper:
            def __init__(self):
                from auto_translate.config import WhisperConfig
                self.cfg = WhisperConfig(retry_uncertain=False)
                self.answers = iter([
                    "あははは", "大丈夫、あはは",
                    "ご視聴ありがとうございました",
                    "ご視聴ありがとうございました。",
                    "ご視聴ありがとうございました",
                ])
            def decode_audio(self, audio, *, prompt=None, hotwords=None, temperature=None):
                return WhisperDecode(next(self.answers), avg_logprob=-.2)

        q_in, q_out = queue.Queue(), queue.Queue()
        def utt(t, generation=0):
            return Utterance(t, t+.5, np.full(8000, .2, dtype=np.float32),
                            generation=generation)
        q_in.put(utt(1))
        q_in.put(utt(2))
        q_in.put(utt(3))
        q_in.put(utt(4))
        q_in.put(AudioProgress(5*3600*16000, reset=True, generation=1))
        q_in.put(utt(5*3600, generation=1))
        q_in.put(None)
        run_whisper_asr(FakeWhisper(), q_in, q_out)
        results = []
        while True:
            item = q_out.get_nowait()
            if item is None:
                break
            if isinstance(item, Transcript):
                results.append(item.ja)
        self.assertEqual(results, [
            "大丈夫、あはは",
            "ご視聴ありがとうございました",
            "ご視聴ありがとうございました",
        ])


if __name__ == "__main__":
    unittest.main()
