"""Standalone outro hallucination safeguards.

Tests cover live, unknown-duration, VOD middle/end, cache replay, seek reset,
and legitimate non-outro thanks. No model download.
"""
import queue
import unittest

import numpy as np

from auto_translate.config import WhisperConfig
from auto_translate.pipeline import _suppress_generic_outro, run_whisper
from auto_translate.session_cache import SessionSubtitleCache
from auto_translate.types import Subtitle, Utterance, AudioProgress
from auto_translate.whisper_v3 import WhisperDecode


def utt(start, generation=0):
    return Utterance(start, start + 1, np.ones(16000, dtype=np.float32),
                     generation=generation)


class TestOutroDetection(unittest.TestCase):
    def test_live_rejects_all_standalone_outros(self):
        cfg = WhisperConfig()
        for s in (
            "Thank you for watching.",
            "Thank you for watching!",
            "Thank you for watching until the end.",
            "Thanks for watching!",
        ):
            self.assertTrue(_suppress_generic_outro(
                s, 100, cfg=cfg, source_info={"live": True},
            ))

    def test_unknown_duration_rejects_even_first_occurrence(self):
        self.assertTrue(_suppress_generic_outro(
            "Thank you for watching.", 1,
            cfg=WhisperConfig(), source_info=None,
        ))

    def test_vod_allows_real_end_but_not_midstream(self):
        cfg = WhisperConfig()
        info = {"live": False, "duration_s": 3000.0}
        self.assertTrue(_suppress_generic_outro(
            "Thank you for watching.", 120.0, cfg=cfg, source_info=info,
        ))
        self.assertFalse(_suppress_generic_outro(
            "Thank you for watching!", 2970.0, cfg=cfg, source_info=info,
        ))

    def test_non_outros_never_filtered(self):
        for s in (
            "Thank you for your hard work.",
            "Thank you!",
            "Thanks for watching the car.",
            "I said thank you for watching.",
            "I'm not thanking you for watching.",
        ):
            self.assertFalse(_suppress_generic_outro(
                s, 50, cfg=WhisperConfig(), source_info={"live": True},
            ))

    def test_user_override_disables_policy(self):
        self.assertFalse(_suppress_generic_outro(
            "Thank you for watching.", 50,
            cfg=WhisperConfig(suppress_generic_outros=False),
            source_info={"live": True},
        ))

    def test_audio_progress_reset_does_not_restore_live_outros(self):
        class Fake:
            def __init__(self):
                self.cfg = WhisperConfig(retry_uncertain=False)
            def decode_audio(self, *args, **kwargs):
                return WhisperDecode("Thank you for watching.", avg_logprob=-.1)

        incoming, outgoing = queue.Queue(), queue.Queue()
        incoming.put(utt(1))
        incoming.put(AudioProgress(0, reset=True, generation=1))
        incoming.put(utt(1, generation=1))
        incoming.put(None)
        run_whisper(Fake(), incoming, outgoing,
                    source_info={"live": True})
        self.assertTrue(outgoing.get_nowait().reset)
        self.assertIsNone(outgoing.get_nowait())

    def test_cached_outro_is_not_re_emitted(self):
        class Fake:
            def __init__(self):
                self.cfg = WhisperConfig(retry_uncertain=False)
                self.calls = 0
            def decode_audio(self, *args, **kwargs):
                self.calls += 1
                return WhisperDecode("Thank you for watching.", avg_logprob=-.1)

        cache = SessionSubtitleCache()
        cache.add(Subtitle(1, 2, "", "Thank you for watching."))
        incoming, outgoing = queue.Queue(), queue.Queue()
        incoming.put(utt(1))
        incoming.put(None)
        decoder = Fake()
        run_whisper(decoder, incoming, outgoing, session_cache=cache,
                    source_info={"live": False, "duration_s": 1000})
        self.assertIsNone(outgoing.get_nowait())
        self.assertEqual(decoder.calls, 0)


if __name__ == "__main__":
    unittest.main()
