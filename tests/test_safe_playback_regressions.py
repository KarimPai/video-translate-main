"""CPU-only concurrency, cancellation and playback-generation regressions."""

from concurrent.futures import CancelledError
import queue
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from auto_translate import pipeline
from auto_translate.config import Config, SafePlaybackConfig, TranslateConfig
from auto_translate.safe_playback import SafeInferenceGate
from auto_translate.translate.base import TranslatorUnavailable
from auto_translate.types import AudioProgress, Subtitle, Transcript, Utterance
from auto_translate.windows_subprocess import stopping


class ObservedSemaphore:
    """Expose admission attempts so tests need no timing sleeps."""
    def __init__(self, semaphore):
        self.semaphore = semaphore
        self.attempted = threading.Event()

    def acquire(self, **kwargs):
        self.attempted.set()
        return self.semaphore.acquire(**kwargs)

    def release(self):
        return self.semaphore.release()


class RecordingTranslator:
    def __init__(self, callback=None):
        self.cfg = TranslateConfig()
        self.calls = []
        self.callback = callback

    def translate(self, japanese, history):
        self.calls.append((japanese, list(history)))
        if self.callback:
            return self.callback(len(self.calls))
        return "Wait here."


class SafePlaybackRegressions(unittest.TestCase):
    def setUp(self):
        stopping.clear()
        self.addCleanup(stopping.clear)

    def gate(self, *, enabled=True, stop=None, concurrency=1):
        gate = SafeInferenceGate(SafePlaybackConfig(
            enabled=enabled, max_inference_concurrency=concurrency), stop)
        gate._slots = ObservedSemaphore(gate._slots)
        return gate

    def spawn(self, function):
        errors = []

        def run():
            try:
                function()
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        self.addCleanup(worker.join, 2)
        return worker, errors

    def join(self, worker, errors):
        worker.join(2)
        self.assertFalse(worker.is_alive(), "worker failed to leave admission/model stage")
        self.assertEqual(errors, [])

    @staticmethod
    def queues(*items):
        incoming, outgoing = queue.Queue(), queue.Queue()
        for item in items:
            incoming.put(item)
        incoming.put(None)
        return incoming, outgoing

    def test_concurrent_inference_never_exceeds_configured_slots(self):
        gate = self.gate(concurrency=2)
        release = threading.Event()
        self.addCleanup(release.set)
        full = threading.Event()
        lock = threading.Lock()
        active = [0, 0]

        def model():
            with gate.operation("mock-model"):
                with lock:
                    active[0] += 1
                    active[1] = max(active)
                    if active[0] == 2:
                        full.set()
                self.assertTrue(release.wait(2))
                with lock:
                    active[0] -= 1

        workers = [self.spawn(model) for _ in range(4)]
        self.assertTrue(full.wait(2))
        self.assertEqual(active[1], 2)
        release.set()
        for worker, errors in workers:
            self.join(worker, errors)
        self.assertEqual(active, [0, 2])

    def test_disabled_gate_preserves_normal_parallelism(self):
        gate = self.gate(enabled=False)
        with gate.operation("asr"):
            with gate.operation("mt", deadline=float("inf")):
                pass

    def test_cancelled_admission_never_calls_model_and_does_not_leak_slot(self):
        gate = self.gate()
        cancel = threading.Event()
        called = []
        cancelled = []

        def waiting():
            try:
                with gate.operation("asr", cancel_event=cancel):
                    called.append(True)
            except CancelledError:
                cancelled.append(True)

        with gate.operation("asr"):
            gate._slots.attempted.clear()
            worker, errors = self.spawn(waiting)
            self.assertTrue(gate._slots.attempted.wait(1))
            cancel.set()
            self.join(worker, errors)
        self.assertEqual(called, [])
        self.assertEqual(cancelled, [True])
        with gate.operation("recovered"):
            pass

    def test_shutdown_cancels_waiting_inference(self):
        stop = threading.Event()
        gate = self.gate(stop=stop)
        cancelled = []

        def waiting():
            try:
                with gate.operation("mt"):
                    self.fail("model entered after shutdown")
            except CancelledError:
                cancelled.append(True)

        with gate.operation("asr"):
            gate._slots.attempted.clear()
            worker, errors = self.spawn(waiting)
            self.assertTrue(gate._slots.attempted.wait(1))
            stop.set()
            self.join(worker, errors)
        self.assertEqual(cancelled, [True])

    def test_expired_deadline_after_admission_wait_abstains(self):
        gate = self.gate()
        clock = [10.0]
        cancelled = []

        def waiting():
            try:
                with gate.operation("mt", deadline=11.0):
                    self.fail("expired selector entered model")
            except CancelledError:
                cancelled.append(True)

        with patch("auto_translate.safe_playback.time.monotonic", side_effect=lambda: clock[0]):
            with gate.operation("asr"):
                gate._slots.attempted.clear()
                worker, errors = self.spawn(waiting)
                self.assertTrue(gate._slots.attempted.wait(1))
                clock[0] = 12.0
            self.join(worker, errors)
        self.assertEqual(cancelled, [True])
        with gate.operation("recovered"):
            pass

    def test_model_exception_releases_admission_slot(self):
        gate = self.gate()
        with self.assertRaisesRegex(RuntimeError, "failed"):
            with gate.operation("mt"):
                raise RuntimeError("model failed")
        with gate.operation("asr"):
            pass

    def test_seek_while_waiting_for_translation_discards_old_cue_and_memory(self):
        gate = self.gate()
        generation = [0]
        translator = RecordingTranslator()
        incoming, outgoing = self.queues(
            Transcript(0, 1, "古い会話", generation=0),
            Transcript(20, 21, "待って", generation=1),
        )
        with gate.operation("asr"):
            gate._slots.attempted.clear()
            worker, errors = self.spawn(lambda: pipeline.run_translate(
                translator, 3, incoming, outgoing,
                get_generation=lambda: generation[0], gate=gate))
            self.assertTrue(gate._slots.attempted.wait(1))
            generation[0] = 1
        self.join(worker, errors)
        self.assertEqual(translator.calls, [("待って", [])])
        subtitles = [item for item in outgoing.queue if isinstance(item, Subtitle)]
        self.assertEqual([item.generation for item in subtitles], [1])

    def test_seek_during_active_model_discards_response_and_rebuilds_memory(self):
        generation = [0]

        def translate(call):
            if call == 1:
                generation[0] = 1
            return "Wait here."

        translator = RecordingTranslator(translate)
        incoming, outgoing = self.queues(
            Transcript(0, 1, "古い会話", generation=0),
            Transcript(20, 21, "待って", generation=1),
        )
        pipeline.run_translate(translator, 3, incoming, outgoing,
                               get_generation=lambda: generation[0], gate=self.gate())
        self.assertEqual(translator.calls[-1], ("待って", []))
        subtitles = [item for item in outgoing.queue if isinstance(item, Subtitle)]
        self.assertEqual([item.generation for item in subtitles], [1])

    def test_shutdown_during_translation_prevents_subtitle_emission(self):
        stop = threading.Event()
        translator = RecordingTranslator(lambda call: stop.set() or "Wait here.")
        incoming, outgoing = self.queues(Transcript(0, 1, "待って"))
        pipeline.run_translate(translator, 3, incoming, outgoing, stop=stop,
                               gate=self.gate(stop=stop))
        self.assertEqual(list(outgoing.queue), [None])

    def test_active_translation_expiring_does_not_enter_next_cues_memory(self):
        position = [2.0]

        def translate(call):
            if call == 1:
                position[0] = 20.0
            return "Wait here."

        translator = RecordingTranslator(translate)
        incoming, outgoing = self.queues(
            Transcript(2, 3, "古い会話"), Transcript(20, 21, "待って"))
        pipeline.run_translate(translator, 3, incoming, outgoing, live=True,
                               playback_position=lambda: position[0], gate=self.gate())
        self.assertEqual(translator.calls[-1], ("待って", []))
        subtitles = [item for item in outgoing.queue if isinstance(item, Subtitle)]
        self.assertEqual([item.start_s for item in subtitles], [20])

    def test_shutdown_during_asr_discards_transcript(self):
        stop = threading.Event()
        primary = Mock()
        primary.transcribe.side_effect = lambda *args: stop.set() or "待って"
        incoming, outgoing = self.queues(Utterance(2, 3, np.ones(16000, dtype=np.float32)))
        pipeline.run_asr(primary, incoming, outgoing, stop=stop,
                         gate=self.gate(stop=stop))
        self.assertEqual(list(outgoing.queue), [None])

    def test_recoverable_translation_failures_keep_only_japanese_context_and_free_slots(self):
        def translate(call):
            if call in (1, 2, 4):
                raise RuntimeError("transient inference failure")
            return "Wait here."

        translator = RecordingTranslator(translate)
        incoming, outgoing = self.queues(*(Transcript(i, i + 1, f"待って{i}") for i in range(5)))
        gate = self.gate()
        pipeline.run_translate(translator, 3, incoming, outgoing, gate=gate)
        self.assertEqual(translator.calls[2][1], [("待って0", ""), ("待って1", "")])
        self.assertEqual(translator.calls[4][1], [("待って1", ""), ("待って2", "Wait here."), ("待って3", "")])
        self.assertEqual(len([item for item in outgoing.queue if isinstance(item, Subtitle)]), 2)
        with gate.operation("post-recovery"):
            pass

    def test_old_failures_and_seek_time_exception_do_not_stop_new_generation(self):
        generation = [0]

        def translate(call):
            if call == 3:
                generation[0] = 1
            if call < 5:
                raise RuntimeError("old response failed" if call < 4 else "new transient failure")
            return "Wait here."

        translator = RecordingTranslator(translate)
        incoming, outgoing = self.queues(
            *(Transcript(i, i + 1, f"古い会話{i}", generation=0) for i in range(3)),
            Transcript(20, 21, "新しい会話", generation=1),
            Transcript(21, 22, "待って", generation=1),
        )
        pipeline.run_translate(translator, 3, incoming, outgoing,
                               get_generation=lambda: generation[0], gate=self.gate())
        self.assertEqual(len(translator.calls), 5)
        self.assertEqual(translator.calls[-1], ("待って", [("新しい会話", "")]))
        subtitles = [item for item in outgoing.queue if isinstance(item, Subtitle)]
        self.assertEqual([(item.generation, item.start_s) for item in subtitles], [(1, 21)])

    def test_stale_terminal_translator_error_does_not_kill_new_generation(self):
        generation = [0]

        def translate(call):
            if call == 1:
                generation[0] = 1
                raise TranslatorUnavailable("stale server response after seek")
            return "Wait here."

        translator = RecordingTranslator(translate)
        incoming, outgoing = self.queues(
            Transcript(0, 1, "古い会話", generation=0),
            Transcript(20, 21, "待って", generation=1),
        )
        pipeline.run_translate(translator, 3, incoming, outgoing,
                               get_generation=lambda: generation[0], gate=self.gate())
        self.assertEqual(translator.calls[-1], ("待って", []))
        subtitles = [item for item in outgoing.queue if isinstance(item, Subtitle)]
        self.assertEqual([item.generation for item in subtitles], [1])

    def test_terminal_translator_error_during_shutdown_is_discarded(self):
        stop = threading.Event()

        def translate(call):
            stop.set()
            raise TranslatorUnavailable("model closed during shutdown")

        incoming, outgoing = self.queues(Transcript(0, 1, "待って"))
        pipeline.run_translate(RecordingTranslator(translate), 3, incoming, outgoing,
                               stop=stop, gate=self.gate(stop=stop))
        self.assertEqual(list(outgoing.queue), [None])

    def test_current_terminal_translator_error_still_surfaces(self):
        def translate(call):
            raise TranslatorUnavailable("current model is unavailable")

        incoming, outgoing = self.queues(Transcript(0, 1, "待って"))
        with self.assertRaisesRegex(TranslatorUnavailable, "current model"):
            pipeline.run_translate(RecordingTranslator(translate), 3, incoming, outgoing,
                                   gate=self.gate())
        self.assertEqual(list(outgoing.queue), [])

    def test_explicit_reset_marker_resets_transient_failure_count(self):
        def translate(call):
            if call < 4:
                raise RuntimeError("transient failure")
            return "Wait here."

        incoming, outgoing = self.queues(
            Transcript(0, 1, "古い会話"), Transcript(1, 2, "古い会話"),
            AudioProgress(20 * 16000, reset=True),
            Transcript(20, 21, "新しい会話"), Transcript(21, 22, "待って"),
        )
        pipeline.run_translate(RecordingTranslator(translate), 3, incoming, outgoing,
                               gate=self.gate())
        subtitles = [item for item in outgoing.queue if isinstance(item, Subtitle)]
        self.assertEqual([item.start_s for item in subtitles], [21])

    def test_translation_expiring_while_admission_blocked_never_starts_model(self):
        gate = self.gate()
        position = [2.0]
        translator = RecordingTranslator()
        incoming, outgoing = self.queues(Transcript(2, 3, "待って"))
        with gate.operation("asr"):
            gate._slots.attempted.clear()
            worker, errors = self.spawn(lambda: pipeline.run_translate(
                translator, 3, incoming, outgoing, live=True,
                playback_position=lambda: position[0], live_late_window_s=7,
                gate=gate))
            self.assertTrue(gate._slots.attempted.wait(1))
            position[0] = 20.0
        self.join(worker, errors)
        self.assertEqual(translator.calls, [])
        self.assertEqual(list(outgoing.queue), [None])

    def test_asr_expiring_while_admission_blocked_never_starts_model(self):
        gate = self.gate()
        position = [2.0]
        primary = Mock()
        primary.transcribe.return_value = "待って"
        incoming, outgoing = self.queues(Utterance(2, 3, np.ones(16000, dtype=np.float32)))
        with gate.operation("mt"):
            gate._slots.attempted.clear()
            worker, errors = self.spawn(lambda: pipeline.run_asr(
                primary, incoming, outgoing, live=True, playback_position=lambda: position[0],
                live_late_window_s=7, gate=gate))
            self.assertTrue(gate._slots.attempted.wait(1))
            position[0] = 20.0
        self.join(worker, errors)
        primary.transcribe.assert_not_called()
        self.assertEqual(list(outgoing.queue), [None])

    def test_reset_marker_clears_previously_confirmed_bilingual_context(self):
        translator = RecordingTranslator()
        incoming, outgoing = self.queues(
            Transcript(0, 1, "古い会話"),
            AudioProgress(20 * 16000, reset=True),
            Transcript(20, 21, "待って"),
        )
        pipeline.run_translate(translator, 3, incoming, outgoing, gate=self.gate())
        self.assertEqual(translator.calls[-1], ("待って", []))

    def test_safe_mode_reduces_real_pipeline_queue_capacities(self):
        real_queue = queue.Queue

        def run_with_safe_mode(enabled):
            cfg = Config()
            cfg.safe_playback = SafePlaybackConfig(enabled=enabled, queue_size=2, audio_chunks=8)
            capacities = []

            def make_queue(maxsize=0):
                capacities.append(maxsize)
                return real_queue(maxsize=maxsize)

            ingest = SimpleNamespace(
                source=SimpleNamespace(live=False, start_s=0), generation=0,
                run=lambda output: output.put(None), stop=lambda: None,
            )

            def segmenter(model, incoming, outgoing, generation, *args):
                outgoing.put(incoming.get())

            with patch.object(pipeline.queue, "Queue", side_effect=make_queue), \
                    patch.object(pipeline, "run_segmenter", side_effect=segmenter):
                pipeline.run(ingest, cfg, Mock(), RecordingTranslator(), [],
                             threading.Event(), segmenter=object())
            return capacities[:4]

        self.assertEqual(run_with_safe_mode(True), [8, 2, 2, 2])
        self.assertEqual(run_with_safe_mode(False), [64, 16, 16, 16])

    def test_shutdown_while_main_output_waits_prevents_render_and_cache(self):
        stop = threading.Event()
        real_queue = queue.Queue
        created = [0]

        class StopOnSubtitleQueue(real_queue):
            def get(self, *args, **kwargs):
                item = super().get(*args, **kwargs)
                if isinstance(item, Subtitle):
                    stop.set()
                return item

        def make_queue(maxsize=0):
            created[0] += 1
            return (StopOnSubtitleQueue if created[0] == 4 else real_queue)(maxsize=maxsize)

        ingest = SimpleNamespace(
            source=SimpleNamespace(live=False, start_s=0), generation=0,
            run=lambda output: output.put(None), stop=lambda: None,
        )

        def segmenter(model, incoming, outgoing, generation, *args):
            incoming.get()
            outgoing.put(Utterance(2, 3, np.ones(16000, dtype=np.float32)))
            outgoing.put(None)

        output, cache, primary = Mock(), Mock(), Mock()
        cache.match.return_value = None
        primary.transcribe.return_value = "待って"
        with patch.object(pipeline.queue, "Queue", side_effect=make_queue), \
                patch.object(pipeline, "run_segmenter", side_effect=segmenter):
            pipeline.run(ingest, Config(), primary, RecordingTranslator(), [output],
                         stop, session_cache=cache, segmenter=object())
        output.show.assert_not_called()
        cache.add.assert_not_called()


if __name__ == "__main__":
    unittest.main()
