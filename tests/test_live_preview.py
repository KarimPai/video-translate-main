"""Preview isolation, local agreement, seek invalidation and bounded work."""
import queue
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from auto_translate.config import VadConfig
from auto_translate.live_preview import LivePreview, PreviewSnapshot, agreed_prefix
from auto_translate.types import AudioChunk, SAMPLE_RATE, Utterance
from auto_translate.vad import FRAME, Segmenter, run_segmenter


def snapshot(value=1, *, generation=0, final=False, start=0):
    return PreviewSnapshot(float(start), float(start + value),
                           np.full(160, value, dtype=np.float32),
                           generation, final=final)


class LivePreviewTests(unittest.TestCase):
    def test_local_agreement_on_japanese_characters_without_word_spaces(self):
        self.assertEqual(agreed_prefix(["今日はいい天気", "今日はいいです"]), "今日はいい")
        events = queue.Queue()
        decoder = Mock(side_effect=["今日はいい天気", "今日はいいです"])
        preview = LivePreview(decoder, events.put_nowait, min_agreement=2)
        self.addCleanup(preview.close)
        preview.submit(snapshot(1))
        self.assertEqual(events.get(timeout=1)["stable_ja"], "")
        preview.submit(snapshot(2))
        event = events.get(timeout=1)
        self.assertEqual(event["stable_ja"], "今日はいい")
        self.assertEqual(event["ja"], "今日はいいです")
        self.assertTrue(event["provisional"])

    def test_only_latest_pending_snapshot_is_retained(self):
        started, release = threading.Event(), threading.Event()
        seen = []

        def decode(audio, generation, **kwargs):
            value = int(audio[0])
            seen.append(value)
            if len(seen) == 1:
                started.set()
                release.wait(timeout=2)
            return str(value)

        events = queue.Queue()
        preview = LivePreview(decode, events.put_nowait)
        self.addCleanup(preview.close)
        preview.submit(snapshot(1))
        self.assertTrue(started.wait(1))
        preview.submit(snapshot(2))
        preview.submit(snapshot(3))
        release.set()
        self.assertEqual(events.get(timeout=1)["ja"], "1")
        self.assertEqual(events.get(timeout=1)["ja"], "3")
        self.assertEqual(seen, [1, 3])

    def test_final_boundary_cancels_and_rejects_late_result(self):
        started, returned = threading.Event(), threading.Event()

        def decode(audio, generation, cancel_event=None, **kwargs):
            started.set()
            cancel_event.wait(timeout=2)
            returned.set()
            return "obsolete draft"

        events = queue.Queue()
        preview = LivePreview(decode, events.put_nowait)
        preview.submit(snapshot())
        self.assertTrue(started.wait(1))
        preview.submit(snapshot(final=True))
        event = events.get(timeout=1)
        self.assertTrue(event["clear"])
        self.assertTrue(returned.wait(1))
        preview.close()
        self.assertTrue(events.empty())
        self.assertFalse(preview._thread.is_alive())

    def test_seek_invalidates_active_result_before_new_snapshot(self):
        started, release = threading.Event(), threading.Event()

        def decode(audio, generation, **kwargs):
            if generation == 0:
                started.set()
                release.wait(timeout=2)
                return "old generation"
            return "new generation"

        events = queue.Queue()
        preview = LivePreview(decode, events.put_nowait)
        self.addCleanup(preview.close)
        preview.submit(snapshot())
        self.assertTrue(started.wait(1))
        preview.submit(PreviewSnapshot(20, 20, np.empty(0, dtype=np.float32),
                                       generation=1, reset=True))
        self.assertEqual(events.get(timeout=1)["generation"], 1)
        preview.submit(snapshot(generation=1, start=20))
        release.set()
        self.assertEqual(events.get(timeout=1)["ja"], "new generation")
        preview.close()
        self.assertTrue(events.empty())

    def test_busy_secondary_skips_without_committing_preview(self):
        called = threading.Event()

        def decode(*args, **kwargs):
            called.set()
            return None

        events = queue.Queue()
        preview = LivePreview(decode, events.put_nowait)
        preview.submit(snapshot())
        self.assertTrue(called.wait(1))
        preview.close()
        self.assertTrue(events.empty())


class SegmenterPreviewTests(unittest.TestCase):
    def test_preview_pcm_is_bounded_and_final_utterance_unchanged(self):
        class FakeVAD:
            def reset(self):
                pass

            def __call__(self, audio):
                return np.full(len(audio) // FRAME, .95)

        pcm = np.full(3 * SAMPLE_RATE, .1, dtype=np.float32)
        with patch("auto_translate.vad.StreamingFireRedVAD", return_value=FakeVAD()):
            baseline = Segmenter(VadConfig())
            expected = baseline.feed(AudioChunk(0, pcm, 0)) + baseline.flush()
            seg = Segmenter(VadConfig())
            previews = []
            seg.set_preview(previews.append, interval_s=.5, window_s=1)
            actual = []
            for offset in range(0, len(pcm), SAMPLE_RATE // 2):
                actual += seg.feed(AudioChunk(offset, pcm[offset:offset + SAMPLE_RATE // 2], 0))
            actual += seg.flush()
            self.assertEqual(len(actual), len(expected))
            self.assertTrue(np.array_equal(actual[0].audio, expected[0].audio))
            self.assertEqual(actual[0].start_s, expected[0].start_s)
            self.assertEqual(actual[0].end_s, expected[0].end_s)
            drafts = [item for item in previews if not item.final]
            self.assertGreaterEqual(len(drafts), 3)
            self.assertTrue(all(len(item.audio) <= SAMPLE_RATE for item in drafts))
            self.assertTrue(previews[-1].final)
            seg.feed(AudioChunk(20 * SAMPLE_RATE, pcm[:0], 0, reset=True, generation=1))
            self.assertTrue(previews[-1].reset)
            self.assertEqual(previews[-1].generation, 1)

    def test_cancelled_segmenter_does_not_wait_for_input(self):
        stop = threading.Event()
        stop.set()
        seg = Mock()
        thread = threading.Thread(target=run_segmenter,
                                  args=(seg, queue.Queue(), queue.Queue(), None, stop))
        thread.start()
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        seg.feed.assert_not_called()

    def test_full_dead_consumer_queue_unblocks_on_stop(self):
        stop, fed = threading.Event(), threading.Event()
        seg = Mock()
        utterance = Utterance(0, 1, np.ones(SAMPLE_RATE, dtype=np.float32))

        def feed(_chunk):
            fed.set()
            return [utterance]

        seg.feed.side_effect = feed
        input_q, output_q = queue.Queue(), queue.Queue(maxsize=1)
        input_q.put(AudioChunk(0, np.ones(SAMPLE_RATE, dtype=np.float32), 0))
        output_q.put("dead consumer")
        thread = threading.Thread(target=run_segmenter,
                                  args=(seg, input_q, output_q, None, stop))
        thread.start()
        self.assertTrue(fed.wait(1))
        stop.set()
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
