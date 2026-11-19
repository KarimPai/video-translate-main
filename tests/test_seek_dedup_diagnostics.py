"""Regression coverage for seek handling, timestamp-aware deduplication and
diagnostic events introduced to prevent stale subtitles and duplicate
publication across seeks and overlapping chunks."""
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import auto_translate.pipeline as pipeline_mod
from auto_translate.config import Config
from auto_translate.pipeline import run
from auto_translate.types import SAMPLE_RATE, Subtitle, Timings
from tests.test_safe_playback_regressions import RecordingTranslator

PCM = np.ones(SAMPLE_RATE, dtype=np.float32) * 0.02


def _sub(start_s, end_s, en, generation=0, ja="日本語"):
    return Subtitle(start_s, end_s, ja, en, Timings(), generation=generation,
                    speech_start_s=start_s)


class _Harness:
    """Drive pipeline.run() with the subtitle queue injected directly."""

    def run(self, subs, generation=0):
        # Pre-create the four queues run() will build (audio, utt, text,
        # subtitle) and inject subtitles directly into the subtitle queue.
        queues = [queue.Queue(maxsize=0) for _ in range(4)]
        sub_q = queues[3]
        for sub in subs:
            sub_q.put(sub)
        sub_q.put(None)

        output = Mock()
        ingest = SimpleNamespace(
            source=SimpleNamespace(live=False, start_s=0), generation=generation,
            run=lambda out: out.put(None), stop=lambda: None,
        )

        def segmenter(model, incoming, outgoing, generation, *args):
            pass

        cfg = Config()
        real_queue = queue.Queue
        order = iter(queues)

        def make_queue(maxsize=0):
            try:
                return next(order)
            except StopIteration:
                return real_queue(maxsize=maxsize)

        with patch.object(pipeline_mod.queue, "Queue", side_effect=make_queue), \
                patch("auto_translate.pipeline.run_segmenter", side_effect=segmenter), \
                patch("auto_translate.pipeline.run_asr", side_effect=lambda *a, **k: None), \
                patch("auto_translate.pipeline.run_translate", side_effect=lambda *a, **k: None):
            stop = threading.Event()
            run(ingest, cfg, Mock(), RecordingTranslator(), [output], stop,
                segmenter=object())
        return output


class TimestampDedupTests(unittest.TestCase):
    def test_overlapping_same_text_suppressed_but_genuine_repetition_survives(self):
        harness = _Harness()
        subs = [
            _sub(0, 2, "go go go"),      # published (count 1)
            _sub(1, 3, "go go go"),      # overlaps sub 1; same text (count 2)
            _sub(2, 4, "go go go"),      # overlaps sub 2; same text (count 3 -> dropped)
            _sub(4, 6, "go go go"),      # non-overlapping genuine repetition (count reset 1)
        ]
        output = harness.run(subs)
        shown = sorted(call.args[0].start_s for call in output.show.call_args_list)
        self.assertEqual(shown, [0.0, 1.0, 4.0])

    def test_different_text_overlapping_chunks_both_survive(self):
        harness = _Harness()
        subs = [
            _sub(0, 2, "wait here."),     # published
            _sub(1, 3, "wait a moment."), # overlaps but different text -> allowed
        ]
        output = harness.run(subs)
        shown = sorted(call.args[0].start_s for call in output.show.call_args_list)
        self.assertEqual(shown, [0.0, 1.0])

    def test_dedup_state_resets_across_generations(self):
        harness = _Harness()
        # Same cue published under generation 0 ...
        output0 = harness.run([_sub(0, 2, "wait here.")], generation=0)
        # ... and re-published under a new generation (seek), because seek
        # handling resets the dedup window instead of carrying it over.
        output1 = harness.run([_sub(0, 2, "wait here.", generation=1)], generation=1)
        self.assertEqual(
            sorted(c.args[0].start_s for c in output0.show.call_args_list),
            [0.0],
        )
        self.assertEqual(
            sorted(c.args[0].start_s for c in output1.show.call_args_list),
            [0.0],
        )

    def test_subtitle_published_diagnostic_recorded(self):
        harness = _Harness()
        events = []
        def capture(event, **data):
            events.append((event, data))
        with patch("auto_translate.pipeline.record_event", side_effect=capture):
            harness.run([_sub(0, 2, "wait here.")])
        self.assertTrue(any(e == "subtitle_published" for e, _ in events))
        pub = [d for e, d in events if e == "subtitle_published"][0]
        self.assertEqual(pub["pcm_start_s"], 0.0)
        self.assertEqual(pub["speech_end_s"], 2.0)


class SeekHandlingTests(unittest.TestCase):
    def test_stale_generation_subtitle_dropped_in_main_loop(self):
        harness = _Harness()
        output = harness.run([_sub(0, 2, "stale caption.", generation=0)], generation=1)
        output.show.assert_not_called()


if __name__ == "__main__":
    unittest.main()
