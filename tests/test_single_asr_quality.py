import queue
import unittest
from unittest.mock import Mock

import numpy as np

from auto_translate.config import VadConfig, vad_for_source
from auto_translate.language_filter import LanguageDecision
from auto_translate.pipeline import run_asr
from auto_translate.text_filter import rejection_reason
from auto_translate.types import Utterance
from auto_translate.vad import FRAME
from test_vad_longform import LongformBoundaryTests as _BoundaryHelper

# Import just the fixture function; unittest discovers module-level TestCase types.
process_pcm = _BoundaryHelper._process
del _BoundaryHelper


class AccuracyTests(unittest.TestCase):


    def test_short_reactions_names_and_spoken_laughter(self):
        for spoken in ("えっ", "うん", "フワモコ", "オパンパンカオン"):
            self.assertIsNone(rejection_reason(spoken))
        # Pure laughter is now filtered at the ASR quality guard level
        self.assertEqual(rejection_reason("あはは"), "laughter_only")
        self.assertIsNotNone(rejection_reason("[laughter]"))
        self.assertIsNotNone(rejection_reason("ダイジョブ" * 8))

    def test_live_and_vod_turn_pause_preserves_pcm_and_media_time(self):
        for live, gap, frames in [(True, 450, 680), (False, 700, 1100)]:
            with self.subTest(live=live):
                pcm = np.full(frames * FRAME, .2, dtype=np.float32)
                probs = np.full(frames, .95)
                # Smoothing trims pause edges; retain 180 ms below the negative threshold.
                pcm[gap*FRAME:(gap+24)*FRAME] = 0
                pcm[(gap+24)*FRAME:] = -.15
                probs[gap:gap+24] = .03
                cues = process_pcm(pcm, probs, vad_for_source(VadConfig(), live=live))
                self.assertEqual(len(cues), 2)
                self.assertAlmostEqual(cues[0].end_s, cues[1].start_s)
                self.assertAlmostEqual(cues[-1].end_s, frames / 100)
                np.testing.assert_array_equal(np.concatenate([c.audio for c in cues]), pcm)

    def test_en_id_excluded_before_asr(self):
        for language in ("en", "id"):
            iq, oq, primary = queue.Queue(), queue.Queue(), Mock()
            iq.put(Utterance(0, 4, np.ones(64000)))
            iq.put(None)
            lid = Mock()
            lid.classify.return_value = LanguageDecision(language, .98, .01, True, "confident")
            run_asr(primary, iq, oq, language_filter=lid)
            primary.transcribe.assert_not_called()
            self.assertEqual(list(oq.queue), [None])


