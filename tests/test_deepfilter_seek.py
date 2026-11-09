"""Regress DeepFilterNet seek cancellation and translated audio progress.

No model weights or GPU are needed; a fake 48 kHz DeepFilterNet implementation
preserves timestamps while verifying the worker's queue and cancellation rules.
"""
import queue
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

import numpy as np

from auto_translate.deepfilter import DeepFilterWorker, MAX_QUEUED_MESSAGES
from auto_translate.types import SAMPLE_RATE, AudioChunk, AudioProgress


class PassThroughDeepFilter:
    sample_rate = 48000

    def process_chunk(self, audio):
        return np.asarray(audio, dtype=np.float32)

    def finalize(self):
        return np.empty(0, dtype=np.float32)


class DeepFilterSeekTests(unittest.TestCase):
    def fake_module(self, cls=PassThroughDeepFilter):
        return patch.dict(
            sys.modules, {"deepfilternet_rs": types.SimpleNamespace(
                DeepFilterNetRealtime=cls
            )},
        )

    def test_48k_audio_is_emitted_as_16k_with_absolute_seek_timestamp(self):
        out = queue.Queue()
        first = 450 * SAMPLE_RATE
        with self.fake_module():
            worker = DeepFilterWorker(out, first_sample=first, generation=3)
            self.assertEqual(worker.input_q.maxsize, MAX_QUEUED_MESSAGES)
            self.assertTrue(worker.put(np.ones(4800, dtype=np.float32)))
            self.assertTrue(worker.put_progress())
            worker.close()
        packets = []
        while not out.empty():
            packets.append(out.get_nowait())
        chunks = [p for p in packets if isinstance(p, AudioChunk)]
        progress = [p for p in packets if isinstance(p, AudioProgress)]
        self.assertEqual(sum(len(p.pcm) for p in chunks), 1600)
        self.assertEqual(chunks[0].start_sample, first)
        self.assertTrue(all(p.generation == 3 for p in packets))
        self.assertEqual(progress[-1].sample, first + 1600)

    def test_seek_cancels_queued_old_pcm_without_waiting_for_drain(self):
        entered, release = threading.Event(), threading.Event()
        processed = []

        class SlowFilter(PassThroughDeepFilter):
            def process_chunk(self, audio):
                processed.append(len(audio))
                entered.set()
                release.wait(timeout=2)
                return np.asarray(audio, dtype=np.float32)

        out = queue.Queue(maxsize=1)  # even blocked output must be cancellable
        generation = [1]
        with self.fake_module(SlowFilter):
            worker = DeepFilterWorker(
                out, 0, 1, is_current=lambda: generation[0] == 1
            )
            worker.put(np.ones(4800, dtype=np.float32))
            self.assertTrue(entered.wait(timeout=1))
            for _ in range(6):
                self.assertTrue(worker.put(np.ones(4800, dtype=np.float32)))
            generation[0] = 2
            self.assertFalse(worker.put(np.ones(4800, dtype=np.float32)))
            release.set()
            started = time.monotonic()
            worker.close(cancel=True)
            self.assertLess(time.monotonic() - started, 1.0)
            worker.worker.join(timeout=1)
            self.assertFalse(worker.worker.is_alive())
        self.assertEqual(len(processed), 1)

    def test_blocked_put_returns_when_seek_changes_generation(self):
        entered, release = threading.Event(), threading.Event()

        class WaitingFilter(PassThroughDeepFilter):
            def process_chunk(self, audio):
                entered.set()
                release.wait(timeout=2)
                return np.asarray(audio, dtype=np.float32)

        generation = [7]
        result = []
        with self.fake_module(WaitingFilter):
            worker = DeepFilterWorker(
                queue.Queue(), 0, 7,
                is_current=lambda: generation[0] == 7,
            )
            self.assertTrue(worker.put(np.ones(4800, dtype=np.float32)))
            self.assertTrue(entered.wait(timeout=1))
            for _ in range(MAX_QUEUED_MESSAGES):
                self.assertTrue(worker.put(np.ones(4800, dtype=np.float32)))
            blocked = threading.Thread(
                target=lambda: result.append(worker.put(np.ones(4800, dtype=np.float32))),
                daemon=True,
            )
            blocked.start()
            time.sleep(0.1)
            self.assertTrue(blocked.is_alive())
            generation[0] = 8
            blocked.join(timeout=1)
            self.assertEqual(result, [False])
            release.set()
            worker.close(cancel=True)
            worker.worker.join(timeout=1)
            self.assertFalse(worker.worker.is_alive())

    def test_normal_end_of_stream_drains_in_order(self):
        out = queue.Queue()
        with self.fake_module():
            worker = DeepFilterWorker(out, 0, generation=0)
            for _ in range(3):
                self.assertTrue(worker.put(np.ones(4800, dtype=np.float32)))
                self.assertTrue(worker.put_progress())
            worker.close()
        packets = []
        while not out.empty():
            packets.append(out.get_nowait())
        self.assertEqual(
            sum(len(p.pcm) for p in packets if isinstance(p, AudioChunk)),
            4800,
        )
        positions = [p.sample for p in packets if isinstance(p, AudioProgress)]
        self.assertEqual(positions, [1600, 3200, 4800])


if __name__ == "__main__":
    unittest.main()
