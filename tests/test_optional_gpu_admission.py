"""GPU VAD/separation admission and seek/shutdown queue safety without models."""
from concurrent.futures import CancelledError
import queue
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from auto_translate.config import SafePlaybackConfig
from auto_translate.safe_playback import SafeInferenceGate
from auto_translate.separation import SeparationWorker
from auto_translate.types import AudioChunk, AudioProgress, Utterance
from auto_translate.vad import run_segmenter
from auto_translate.windows_subprocess import stopping


class ObservedSemaphore:
    def __init__(self, semaphore):
        self.semaphore = semaphore
        self.waiting = threading.Event()

    def acquire(self, **kwargs):
        self.waiting.set()
        return self.semaphore.acquire(**kwargs)

    def release(self):
        self.semaphore.release()


class ObservedQueue(queue.Queue):
    def __init__(self, maxsize):
        super().__init__(maxsize=maxsize)
        self.blocked = threading.Event()

    def put(self, item, *args, **kwargs):
        if self.full():
            self.blocked.set()
        return super().put(item, *args, **kwargs)


class SegmenterFixture:
    def __init__(self, on_feed=None):
        self._generation = 0
        self._next_frame_sample = 0
        self.calls = []
        self.on_feed = on_feed

    def feed(self, item):
        if item.reset:
            self._generation = item.generation
            return []
        self.calls.append(item.generation)
        if self.on_feed:
            self.on_feed()
        return [Utterance(0, 1, np.ones(160), generation=item.generation)]

    def flush(self):
        return []

    def advance_progress(self, *args, **kwargs):
        return None


class OptionalGpuAdmissionTests(unittest.TestCase):
    def setUp(self):
        stopping.clear()
        self.addCleanup(stopping.clear)

    def gate(self, enabled=True):
        gate = SafeInferenceGate(SafePlaybackConfig(enabled=enabled))
        gate._slots = ObservedSemaphore(gate._slots)
        return gate

    def spawn(self, function):
        errors = []

        def run():
            try:
                function()
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 3)
        return thread, errors

    def assert_finished(self, thread, errors):
        thread.join(timeout=3)
        self.assertFalse(thread.is_alive(), "optional GPU stage blocked shutdown/seek")
        self.assertEqual(errors, [])

    @staticmethod
    def incoming(*items):
        result = queue.Queue()
        for item in items:
            result.put(item)
        result.put(None)
        return result

    def test_vad_waiting_on_gpu_discards_seek_stale_chunk(self):
        gate, generation = self.gate(), [0]
        seg, output = SegmenterFixture(), queue.Queue()
        incoming = self.incoming(
            AudioChunk(0, np.ones(160), 0, generation=0),
            AudioChunk(16000, np.empty(0), 0, reset=True, generation=1),
            AudioChunk(16000, np.ones(160), 0, generation=1),
        )
        with gate.operation("asr"):
            gate._slots.waiting.clear()
            thread, errors = self.spawn(lambda: run_segmenter(seg, incoming, output,
                                                              lambda: generation[0], gate))
            self.assertTrue(gate._slots.waiting.wait(timeout=1))
            generation[0] = 1
        self.assert_finished(thread, errors)
        self.assertEqual(seg.calls, [1])
        utterances = [item for item in output.queue if isinstance(item, Utterance)]
        self.assertEqual([item.generation for item in utterances], [1])

    def test_active_vad_result_is_discarded_after_seek(self):
        generation = [0]
        seg = SegmenterFixture(lambda: generation.__setitem__(0, 1))
        output = queue.Queue()
        run_segmenter(seg, self.incoming(AudioChunk(0, np.ones(160), 0)), output,
                      lambda: generation[0], self.gate())
        self.assertEqual(list(output.queue), [None])

    def test_vad_reset_and_flush_do_not_wait_for_inference_admission(self):
        gate, seg, output = self.gate(), SegmenterFixture(), queue.Queue()
        with gate.operation("mt"):
            thread, errors = self.spawn(lambda: run_segmenter(
                seg, self.incoming(AudioChunk(16000, np.empty(0), 0, reset=True, generation=1)),
                output, lambda: 1, gate))
            self.assert_finished(thread, errors)
        self.assertEqual(seg.calls, [])
        self.assertIsInstance(output.get_nowait(), AudioProgress)

    def test_separation_load_and_inference_both_obey_shared_gate(self):
        gate, output = self.gate(), queue.Queue()
        checked = []

        def ensure_admitted(stage):
            acquired = gate._slots.semaphore.acquire(blocking=False)
            if acquired:
                gate._slots.semaphore.release()
            self.assertFalse(acquired, f"{stage} bypassed GPU admission")
            checked.append(stage)

        model = Mock()
        model.separate.side_effect = lambda audio, index: ensure_admitted("inference") or audio.mean(axis=1)

        def load(path):
            ensure_admitted("load")
            return model

        with patch("auto_translate.separation.VocalSeparator", side_effect=load), \
                patch.object(SeparationWorker, "_emit") as emit:
            worker = SeparationWorker(output, 0, "fixture", 0, gate, lambda: 0)
            worker.put(np.ones((64, 2), dtype=np.float32))
            worker.close()
        self.assertEqual(checked, ["load", "inference"])
        self.assertEqual(worker.input_q.maxsize, 4)
        self.assertFalse(worker.worker.is_alive())
        self.assertIsNone(worker.error)
        emit.assert_called_once()
        model.close.assert_called_once()

    def test_separation_backpressure_put_cancels_after_seek(self):
        gate, generation = self.gate(), [0]
        with patch("auto_translate.separation.VocalSeparator") as model:
            with gate.operation("asr"):
                gate._slots.waiting.clear()
                worker = SeparationWorker(queue.Queue(), 0, "fixture", 0, gate, lambda: generation[0])
                self.assertTrue(gate._slots.waiting.wait(timeout=1))
                worker.input_q = ObservedQueue(4)
                for _ in range(4):
                    worker.put(np.ones((16, 2), dtype=np.float32))
                producer, errors = self.spawn(lambda: worker.put(np.ones((16, 2), dtype=np.float32)))
                self.assertTrue(worker.input_q.blocked.wait(timeout=1))
                generation[0] = 1
                producer.join(timeout=2)
                self.assertFalse(producer.is_alive())
                self.assertEqual(len(errors), 1)
                self.assertIsInstance(errors[0], CancelledError)
            worker.close()
            model.assert_not_called()
        self.assertIsNone(worker.error)
        self.assertFalse(worker.worker.is_alive())

    def test_separation_failed_worker_with_full_input_queue_does_not_deadlock_close(self):
        gate = self.gate()
        with patch("auto_translate.separation.VocalSeparator", side_effect=RuntimeError("fixture model failure")):
            with gate.operation("asr"):
                gate._slots.waiting.clear()
                worker = SeparationWorker(queue.Queue(), 0, "fixture", 0, gate, lambda: 0)
                self.assertTrue(gate._slots.waiting.wait(timeout=1))
                for _ in range(4):
                    worker.put(np.ones((16, 2), dtype=np.float32))
            worker.worker.join(timeout=2)
            self.assertFalse(worker.worker.is_alive())
            with self.assertRaisesRegex(RuntimeError, "fixture model failure"):
                worker.close()

    def test_separation_active_result_cannot_emit_after_seek(self):
        gate, generation = self.gate(), [0]
        model = Mock()

        def separate(audio, index):
            generation[0] = 1
            return audio.mean(axis=1)

        model.separate.side_effect = separate
        with patch("auto_translate.separation.VocalSeparator", return_value=model), \
                patch.object(SeparationWorker, "_emit") as emit:
            worker = SeparationWorker(queue.Queue(), 0, "fixture", 0, gate, lambda: generation[0])
            worker.put(np.ones((64, 2), dtype=np.float32))
            worker.close()
            emit.assert_not_called()
        self.assertIsNone(worker.error)
        model.close.assert_called_once()

    def test_separation_blocked_output_releases_on_seek(self):
        gate, generation = self.gate(), [0]
        output = ObservedQueue(1)
        output.put("existing")
        model = Mock()
        model.separate.side_effect = lambda audio, index: audio.mean(axis=1)
        signal = SimpleNamespace(resample_poly=lambda audio, *args: audio)
        with patch("auto_translate.separation.VocalSeparator", return_value=model), \
                patch.dict("sys.modules", {"scipy": SimpleNamespace(), "scipy.signal": signal}):
            worker = SeparationWorker(output, 0, "fixture", 0, gate, lambda: generation[0])
            worker.put(np.ones((64, 2), dtype=np.float32))
            closer, errors = self.spawn(worker.close)
            self.assertTrue(output.blocked.wait(timeout=1))
            generation[0] = 1
            self.assert_finished(closer, errors)
        self.assertEqual(list(output.queue), ["existing"])
        self.assertIsNone(worker.error)
        self.assertFalse(worker.worker.is_alive())

    def test_separation_shutdown_unblocks_producer_and_simultaneous_close(self):
        gate = self.gate()
        with patch("auto_translate.separation.VocalSeparator") as model:
            with gate.operation("asr"):
                gate._slots.waiting.clear()
                worker = SeparationWorker(queue.Queue(), 0, "fixture", 0, gate, lambda: 0)
                self.assertTrue(gate._slots.waiting.wait(timeout=1))
                worker.input_q = ObservedQueue(4)
                for _ in range(4):
                    worker.put(np.ones((16, 2), dtype=np.float32))
                producer, errors = self.spawn(lambda: worker.put(np.ones((16, 2), dtype=np.float32)))
                self.assertTrue(worker.input_q.blocked.wait(timeout=1))
                stopping.set()
                producer.join(timeout=2)
                self.assertFalse(producer.is_alive())
                self.assertEqual(len(errors), 1)
                self.assertIsInstance(errors[0], CancelledError)
            closers = [self.spawn(worker.close) for _ in range(2)]
            for closer, failures in closers:
                self.assert_finished(closer, failures)
            model.assert_not_called()
        self.assertIsNone(worker.error)
        self.assertFalse(worker.worker.is_alive())

    def test_disabled_safe_mode_preserves_separation_queue_policy(self):
        with patch("auto_translate.separation.VocalSeparator", return_value=Mock()):
            worker = SeparationWorker(queue.Queue(), 0, "fixture", 0, self.gate(enabled=False), lambda: 0)
            self.assertEqual(worker.input_q.maxsize, 0)
            worker.close()
        self.assertIsNone(worker.error)


if __name__ == "__main__":
    unittest.main()
