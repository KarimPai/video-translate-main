"""Streaming GTCRN regression tests using a fake Sherpa ONNX runtime.

Tests verify absolute timestamps, progress, queue backpressure, and seek
cancellation without downloading weights or requiring GPU libraries.
"""
import queue
import sys
import threading
import time
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

from auto_translate.gtcrn import GtcrnWorker, MAX_QUEUED_MESSAGES
from auto_translate.types import SAMPLE_RATE, AudioChunk, AudioProgress


class _Config:
    def validate(self):
        return True


class _PassThrough:
    sample_rate = SAMPLE_RATE
    frame_shift_in_samples = 256

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, audio, sample_rate):
        assert sample_rate == SAMPLE_RATE
        return types.SimpleNamespace(
            samples=np.asarray(audio, dtype=np.float32),
            sample_rate=SAMPLE_RATE,
        )

    def flush(self):
        return types.SimpleNamespace(
            samples=np.empty(0, dtype=np.float32), sample_rate=SAMPLE_RATE,
        )


@contextmanager
def fake_sherpa(denoiser_cls=_PassThrough):
    fake_module = types.SimpleNamespace(
        OnlineSpeechDenoiserConfig=lambda **kw: _Config(),
        OfflineSpeechDenoiserModelConfig=lambda **kw: _Config(),
        OfflineSpeechDenoiserGtcrnModelConfig=lambda **kw: _Config(),
        OnlineSpeechDenoiser=denoiser_cls,
    )
    with patch.dict(sys.modules, {"sherpa_onnx": fake_module}), patch(
        "auto_translate.gtcrn._model_file", return_value=Path(__file__)
    ):
        yield


class GtcrnSeekTests(unittest.TestCase):
    def test_audio_is_16k_with_absolute_seek_timestamp_and_progress(self):
        out = queue.Queue()
        first = 450 * SAMPLE_RATE
        with fake_sherpa():
            worker = GtcrnWorker(out, first_sample=first, generation=3)
            self.assertEqual(worker.input_q.maxsize, MAX_QUEUED_MESSAGES)
            self.assertTrue(worker.put(np.ones(4096, dtype=np.float32)))
            self.assertTrue(worker.put_progress())
            worker.close()
        packets = []
        while not out.empty():
            packets.append(out.get_nowait())
        chunks = [p for p in packets if isinstance(p, AudioChunk)]
        progress = [p for p in packets if isinstance(p, AudioProgress)]
        self.assertEqual(sum(len(p.pcm) for p in chunks), 4096)
        self.assertEqual(chunks[0].start_sample, first)
        self.assertTrue(all(p.generation == 3 for p in packets))
        self.assertEqual(progress[-1].sample, first + 4096)

    def test_last_partial_frame_is_flushed_at_end_of_stream(self):
        out = queue.Queue()
        with fake_sherpa():
            worker = GtcrnWorker(out, 0, generation=0)
            worker.put(np.ones(1600, dtype=np.float32))
            worker.close()
        chunks = []
        while not out.empty():
            packet = out.get_nowait()
            if isinstance(packet, AudioChunk):
                chunks.append(packet)
        self.assertEqual(sum(len(p.pcm) for p in chunks), 1600)
        self.assertEqual(chunks[0].start_sample, 0)

    def test_seek_cancels_queued_old_audio(self):
        entered, release = threading.Event(), threading.Event()
        processed = []

        class SlowDenoiser(_PassThrough):
            def __call__(self, audio, sample_rate):
                processed.append(len(audio))
                entered.set()
                release.wait(timeout=2)
                return super().__call__(audio, sample_rate)

        out = queue.Queue(maxsize=1)
        generation = [1]
        with fake_sherpa(SlowDenoiser):
            worker = GtcrnWorker(
                out, 0, 1, is_current=lambda: generation[0] == 1,
            )
            self.assertTrue(worker.put(np.ones(4096, dtype=np.float32)))
            self.assertTrue(entered.wait(timeout=1))
            for _ in range(6):
                self.assertTrue(worker.put(np.ones(4096, dtype=np.float32)))
            generation[0] = 2
            self.assertFalse(worker.put(np.ones(4096, dtype=np.float32)))
            release.set()
            started = time.monotonic()
            worker.close(cancel=True)
            self.assertLess(time.monotonic() - started, 1.0)
            worker.worker.join(timeout=1)
            self.assertFalse(worker.worker.is_alive())
        self.assertEqual(len(processed), 1)

    def test_blocked_put_is_released_when_generation_changes(self):
        entered, release = threading.Event(), threading.Event()

        class WaitingDenoiser(_PassThrough):
            def __call__(self, audio, sample_rate):
                entered.set()
                release.wait(timeout=2)
                return super().__call__(audio, sample_rate)

        generation = [7]
        result = []
        with fake_sherpa(WaitingDenoiser):
            worker = GtcrnWorker(
                queue.Queue(), 0, 7,
                is_current=lambda: generation[0] == 7,
            )
            self.assertTrue(worker.put(np.ones(4096, dtype=np.float32)))
            self.assertTrue(entered.wait(timeout=1))
            for _ in range(MAX_QUEUED_MESSAGES):
                self.assertTrue(worker.put(np.ones(4096, dtype=np.float32)))
            blocked = threading.Thread(
                target=lambda: result.append(
                    worker.put(np.ones(4096, dtype=np.float32))
                ),
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

    def test_end_of_stream_drains_in_order(self):
        out = queue.Queue()
        with fake_sherpa():
            worker = GtcrnWorker(out, 0, generation=0)
            for _ in range(3):
                self.assertTrue(worker.put(np.ones(4096, dtype=np.float32)))
                self.assertTrue(worker.put_progress())
            worker.close()
        packets = []
        while not out.empty():
            packets.append(out.get_nowait())
        self.assertEqual(
            sum(len(p.pcm) for p in packets if isinstance(p, AudioChunk)),
            12288,
        )
        positions = [p.sample for p in packets if isinstance(p, AudioProgress)]
        self.assertEqual(positions, [4096, 8192, 12288])


if __name__ == "__main__":
    unittest.main()
