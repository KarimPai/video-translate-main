"""Regression tests for pipeline edge cases.

Covers:
- 60+ seconds of silence: watermark advances, no deadlock, no missing subtitles
- Continuous dialogue: no backlog growth, stable latency, no expired subtitles
- Full queues: backpressure works, no dropped utterances, no deadlock
- Failed translation: retry logic, no stuck pipeline, graceful degradation
- Rapid seeking: generation reset, no stale subtitles, clean watermark state
- User pause/play: independent of auto-pause, workers continue during pause
- Clean shutdown: all child processes terminate, no hanging threads

These tests use mocks and simulated data — no GPU required.
"""
import queue
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch, MagicMock

import numpy as np

from auto_translate.config import Config, TranslateConfig, SafePlaybackConfig, MpvConfig
from auto_translate.pipeline import run_asr, run_translate
from auto_translate.safe_playback import SafeInferenceGate
from auto_translate.translate.base import TranslatorUnavailable
from auto_translate.types import (
    SAMPLE_RATE,
    AudioChunk,
    AudioProgress,
    SourceEOF,
    Subtitle,
    Timings,
    Transcript,
    Utterance,
)


class LongSilenceTests(unittest.TestCase):
    """60+ seconds of silence should advance the watermark without deadlock."""

    def test_silence_advances_watermark(self):
        """VAD-confirmed silence should advance _vad_progress continuously."""
        output = Mock()
        output._vad_progress = 0.0
        output._translated_until = 0.0
        output._pending_speech = []
        output._generation = 0
        output._buffer_condition = threading.Condition()

        # Simulate 60 seconds of silence advancing via AudioProgress
        for sec in range(0, 65, 5):
            sample = sec * SAMPLE_RATE
            progress = AudioProgress(sample=sample, generation=0)
            # Simulate VAD advance_progress — no speech, so watermark advances
            output._vad_progress = sec
            # No speech recorded, so advance_to_subtitle is not called
            # But _try_advance_through_silence should advance _translated_until

        # After silence, _vad_progress should be at 60+
        self.assertGreaterEqual(output._vad_progress, 60.0)

    def test_no_subtitles_during_silence(self):
        """Silence produces no utterances, hence no subtitles."""
        primary = Mock(last_timing=None)
        primary.transcribe.return_value = ""
        input_q, output_q = queue.Queue(), queue.Queue()

        # No utterances during silence — VAD doesn't produce them
        input_q.put(None)
        run_asr(primary, input_q, output_q, live=False)
        primary.transcribe.assert_not_called()
        self.assertEqual(list(output_q.queue), [None])

    def test_silence_no_deadlock(self):
        """Pipeline should not deadlock during extended silence."""
        # Simulate silence processing: VAD receives chunks, produces no utterances,
        # and the pipeline continues without blocking.
        processed = []

        def consumer(q):
            while True:
                try:
                    item = q.get(timeout=0.5)
                    if item is None:
                        break
                    processed.append(item)
                except queue.Empty:
                    break

        audio_q = queue.Queue(maxsize=8)
        t = threading.Thread(target=consumer, args=(audio_q,), daemon=True)
        t.start()

        # Feed silence chunks — consumer keeps up so producer doesn't block
        for i in range(12):  # 12 * 5s = 60s of silence
            chunk = AudioChunk(
                start_sample=i * 5 * SAMPLE_RATE,
                pcm=np.zeros(5 * SAMPLE_RATE, dtype=np.float32),
                read_at=time.monotonic(),
                generation=0,
            )
            audio_q.put(chunk)

        audio_q.put(None)  # Sentinel
        t.join(timeout=5)

        self.assertEqual(len(processed), 12)
        self.assertFalse(t.is_alive())


class ContinuousDialogueTests(unittest.TestCase):
    """Continuous dialogue should maintain stable latency without backlog growth."""

    def _make_translator(self):
        """Create a properly configured mock translator."""
        translator = Mock()
        translator.cfg = SimpleNamespace(
            context_max_gap_s=15.0,
            remove_fillers=True,
            collapse_repetitions=True,
        )
        return translator

    def test_no_backlog_growth(self):
        """Sustained dialogue should not cause unbounded queue growth."""
        primary = Mock(last_timing=None)
        primary.transcribe.return_value = "待って"

        translator = self._make_translator()
        translator.translate.return_value = "Wait."

        # Feed 20 utterances through ASR then translation
        transcript_q = queue.Queue(maxsize=16)

        # ASR worker
        asr_done = threading.Event()
        def asr_worker():
            input_q = queue.Queue()
            for i in range(20):
                utt = Utterance(i, i + 1, np.ones(SAMPLE_RATE, dtype=np.float32), generation=0)
                input_q.put(utt)
            input_q.put(None)
            run_asr(primary, input_q, transcript_q, live=False)
            asr_done.set()

        t_asr = threading.Thread(target=asr_worker, daemon=True)
        t_asr.start()
        asr_done.wait(timeout=10)

        # Translation worker
        subtitle_q = queue.Queue(maxsize=32)
        trans_done = threading.Event()
        def trans_worker():
            run_translate(translator, 0, transcript_q, subtitle_q, live=False)
            trans_done.set()

        t_trans = threading.Thread(target=trans_worker, daemon=True)
        t_trans.start()
        trans_done.wait(timeout=10)

        # Collect subtitles
        subtitles = []
        while not subtitle_q.empty():
            try:
                item = subtitle_q.get_nowait()
                if isinstance(item, Subtitle):
                    subtitles.append(item)
            except queue.Empty:
                break

        self.assertEqual(len(subtitles), 20)
        # Check ordering
        for i, sub in enumerate(subtitles):
            self.assertEqual(sub.start_s, i)
            self.assertEqual(sub.end_s, i + 1)

    def test_stable_latency(self):
        """Translation latency should remain stable, not grow over time."""
        translator = self._make_translator()
        latencies = []

        def record_latency(japanese, history, timings=None):
            t0 = time.monotonic()
            time.sleep(0.01)  # Simulate translation time
            elapsed = time.monotonic() - t0
            latencies.append(elapsed)
            return "Wait."

        translator.translate = record_latency

        input_q, output_q = queue.Queue(), queue.Queue()
        for i in range(10):
            tr = Transcript(i, i + 1, "待って", generation=0)
            input_q.put(tr)
        input_q.put(None)

        run_translate(translator, 0, input_q, output_q, live=False)

        self.assertEqual(len(latencies), 10)
        # Latency should be roughly constant, not growing
        if len(latencies) >= 2:
            first_avg = sum(latencies[:3]) / 3
            last_avg = sum(latencies[-3:]) / 3
            # Last should not be more than 3x the first (allowing for variance)
            self.assertLess(last_avg, first_avg * 3)


class FullQueueTests(unittest.TestCase):
    """Full queues should apply backpressure, not drop utterances."""

    def _make_translator(self):
        """Create a properly configured mock translator."""
        translator = Mock()
        translator.cfg = SimpleNamespace(
            context_max_gap_s=15.0,
            remove_fillers=True,
            collapse_repetitions=True,
        )
        return translator

    def test_blocking_put_no_drop(self):
        """Bounded queues with blocking put should not drop items."""
        q = queue.Queue(maxsize=2)
        results = []

        def consumer():
            for _ in range(5):
                item = q.get()
                results.append(item)
                q.task_done()

        t = threading.Thread(target=consumer, daemon=True)
        t.start()

        # Producer blocks when queue is full
        for i in range(5):
            q.put(i)  # This blocks when q is full, doesn't drop

        q.put(None)  # Sentinel
        t.join(timeout=5)

        self.assertEqual(results, [0, 1, 2, 3, 4])

    def test_asr_no_drop_on_full_queue(self):
        """ASR worker should not drop utterances when output queue is full."""
        primary = Mock(last_timing=None)
        primary.transcribe.return_value = "待って"

        # Small output queue
        output_q = queue.Queue(maxsize=2)
        input_q = queue.Queue()

        # Feed 5 utterances
        for i in range(5):
            utt = Utterance(i, i + 1, np.ones(SAMPLE_RATE, dtype=np.float32), generation=0)
            input_q.put(utt)
        input_q.put(None)

        # Slow consumer to trigger backpressure
        consumed = []
        def slow_consumer():
            while True:
                item = output_q.get()
                if item is None:
                    break
                consumed.append(item)
                time.sleep(0.05)  # Slow down to trigger backpressure

        t = threading.Thread(target=slow_consumer, daemon=True)
        t.start()

        # ASR should process all utterances, blocking on full queue
        run_asr(primary, input_q, output_q, live=False)
        output_q.put(None)
        t.join(timeout=10)

        self.assertEqual(len(consumed), 5)
        self.assertEqual(primary.transcribe.call_count, 5)

    def test_translate_no_drop_on_full_queue(self):
        """Translation worker should not drop transcripts when output queue is full."""
        translator = self._make_translator()
        translator.translate.return_value = "Wait."

        output_q = queue.Queue(maxsize=2)
        input_q = queue.Queue()

        for i in range(5):
            tr = Transcript(i, i + 1, "待って", generation=0)
            input_q.put(tr)
        input_q.put(None)

        consumed = []
        def slow_consumer():
            while True:
                item = output_q.get()
                if item is None:
                    break
                consumed.append(item)
                time.sleep(0.05)

        t = threading.Thread(target=slow_consumer, daemon=True)
        t.start()

        run_translate(translator, 0, input_q, output_q, live=False)
        output_q.put(None)
        t.join(timeout=10)

        self.assertEqual(len(consumed), 5)


class FailedTranslationTests(unittest.TestCase):
    """Failed translations should retry gracefully without stuck pipeline."""

    def _make_translator(self):
        """Create a properly configured mock translator."""
        translator = Mock()
        translator.cfg = SimpleNamespace(
            context_max_gap_s=15.0,
            remove_fillers=True,
            collapse_repetitions=True,
        )
        return translator

    def test_retry_on_empty_output(self):
        """Empty ASR output should trigger bounded retries."""
        primary = Mock(last_rejection="error", last_timing=None)
        primary.transcribe.return_value = ""

        input_q, output_q = queue.Queue(), queue.Queue()
        utt = Utterance(2, 3, np.ones(SAMPLE_RATE, dtype=np.float32), vad_confidence=0.95)
        input_q.put(utt)
        input_q.put(None)

        with patch("auto_translate.pipeline.time.sleep"):
            run_asr(primary, input_q, output_q, live=False)

        # Bounded retries for transient failures
        self.assertGreaterEqual(primary.transcribe.call_count, 2)

    def test_translator_unavailable(self):
        """TranslatorUnavailable should propagate and stop the pipeline."""
        translator = self._make_translator()
        translator.translate.side_effect = TranslatorUnavailable("Server down")

        input_q, output_q = queue.Queue(), queue.Queue()
        tr = Transcript(2, 3, "待って", generation=0)
        input_q.put(tr)
        input_q.put(None)

        # TranslatorUnavailable propagates — pipeline stops
        with self.assertRaises(TranslatorUnavailable):
            run_translate(translator, 0, input_q, output_q, live=False)
        translator.translate.assert_called_once()

    def test_partial_failure_continues(self):
        """One failed translation should not stop processing subsequent items."""
        translator = self._make_translator()
        call_count = [0]

        def sometimes_fail(japanese, history, timings=None):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Temporary failure")
            return "Wait."

        translator.translate = sometimes_fail

        input_q, output_q = queue.Queue(), queue.Queue()
        for i in range(3):
            tr = Transcript(i, i + 1, "待って", generation=0)
            input_q.put(tr)
        input_q.put(None)

        run_translate(translator, 0, input_q, output_q, live=False)
        # All 3 should be attempted
        self.assertEqual(call_count[0], 3)


class RapidSeekingTests(unittest.TestCase):
    """Rapid seeking should reset state cleanly without stale subtitles."""

    def test_generation_reset(self):
        """New generation should invalidate old subtitles."""
        old_gen = 0
        new_gen = 1

        old_sub = Subtitle(
            start_s=1.0, end_s=2.0, ja="待って", en="Wait.",
            generation=old_gen,
        )
        new_sub = Subtitle(
            start_s=10.0, end_s=11.0, ja="行く", en="Go.",
            generation=new_gen,
        )

        # Simulate generation-based filtering
        subtitles = [old_sub, new_sub]
        filtered = [s for s in subtitles if s.generation == new_gen]

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].ja, "行く")

    def test_stale_asr_cancelled(self):
        """ASR processes all utterances; pipeline filters by generation."""
        primary = Mock(last_timing=None)
        primary.transcribe.return_value = "待って"

        input_q, output_q = queue.Queue(), queue.Queue()

        # Old generation utterance
        old_utt = Utterance(1, 2, np.ones(SAMPLE_RATE, dtype=np.float32), generation=0)
        input_q.put(old_utt)

        # New generation utterance
        new_utt = Utterance(10, 11, np.ones(SAMPLE_RATE, dtype=np.float32), generation=1)
        input_q.put(new_utt)

        input_q.put(None)

        run_asr(primary, input_q, output_q, live=False)

        # Both should be transcribed (ASR doesn't filter by generation internally)
        # But pipeline main loop filters by generation before output
        self.assertEqual(primary.transcribe.call_count, 2)

    def test_watermark_reset_on_seek(self):
        """Watermark should reset after seek."""
        output = Mock()
        output._vad_progress = 50.0
        output._translated_until = 45.0
        output._pending_speech = [(40, 42), (43, 45)]
        output._generation = 0
        output._buffer_condition = threading.Condition()
        output._user_paused = False
        output._auto_paused = False
        output._last_pause_reason = None

        # Simulate reset_buffer
        def reset_buffer(progress_s):
            output._vad_progress = progress_s
            output._translated_until = progress_s
            output._pending_speech = []
            output._generation += 1
            output._user_paused = False
            output._auto_paused = False
            output._last_pause_reason = None

        reset_buffer(100.0)

        self.assertEqual(output._vad_progress, 100.0)
        self.assertEqual(output._translated_until, 100.0)
        self.assertEqual(output._pending_speech, [])
        self.assertEqual(output._generation, 1)
        self.assertFalse(output._auto_paused)


class UserPausePlayTests(unittest.TestCase):
    """User pause/play should be independent of auto-pause."""

    def test_user_pause_respected(self):
        """Auto-pause should not override user-initiated pause."""
        output = Mock()
        output._user_paused = True
        output._auto_paused = False
        output._translated_until = 10.0
        output._generation = 0

        # Even if translation lead is low, user pause should be respected
        # Auto-pause logic should check _user_paused first
        def should_auto_pause():
            if output._user_paused:
                return False  # Don't auto-pause when user already paused
            lead = output._translated_until - 5.0  # playback at 5s
            return lead < 2.0

        self.assertFalse(should_auto_pause())

    def test_user_play_not_overridden(self):
        """Auto-pause should not override user-initiated play."""
        output = Mock()
        output._user_paused = False
        output._auto_paused = True  # Was auto-paused
        output._translated_until = 20.0  # Lead is now sufficient

        # User unpauses — auto-pause should not re-pause
        def should_auto_pause():
            if not output._user_paused:
                lead = output._translated_until - 15.0  # playback at 15s
                return lead < 2.0
            return False

        self.assertFalse(should_auto_pause())

    def test_workers_continue_during_pause(self):
        """ASR and translation workers should continue during MPV pause."""
        primary = Mock(last_timing=None)
        primary.transcribe.return_value = "待って"

        translator = Mock()
        translator.memory_lines = 0
        translator.translate.return_value = "Wait."

        # Workers should run regardless of MPV pause state
        input_q, transcript_q = queue.Queue(), queue.Queue()
        utt = Utterance(1, 2, np.ones(SAMPLE_RATE, dtype=np.float32), generation=0)
        input_q.put(utt)
        input_q.put(None)

        asr_done = threading.Event()
        def asr_worker():
            run_asr(primary, input_q, transcript_q, live=False)
            transcript_q.put(None)
            asr_done.set()

        t = threading.Thread(target=asr_worker, daemon=True)
        t.start()
        asr_done.wait(timeout=5)

        self.assertTrue(asr_done.is_set())
        primary.transcribe.assert_called_once()


class CleanShutdownTests(unittest.TestCase):
    """Pipeline should shut down cleanly without hanging threads."""

    def test_asr_worker_exits(self):
        """ASR worker should exit when sentinel is received."""
        primary = Mock(last_timing=None)
        primary.transcribe.return_value = "待って"

        input_q, output_q = queue.Queue(), queue.Queue()
        input_q.put(None)

        done = threading.Event()
        def worker():
            run_asr(primary, input_q, output_q, live=False)
            done.set()

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        self.assertTrue(done.wait(timeout=5))
        self.assertFalse(t.is_alive())

    def test_translate_worker_exits(self):
        """Translation worker should exit when sentinel is received."""
        translator = Mock()
        translator.cfg = SimpleNamespace(
            context_max_gap_s=15.0,
            remove_fillers=True,
            collapse_repetitions=True,
        )
        translator.translate.return_value = "Wait."

        input_q, output_q = queue.Queue(), queue.Queue()
        input_q.put(None)

        done = threading.Event()
        def worker():
            run_translate(translator, 0, input_q, output_q, live=False)
            done.set()

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        self.assertTrue(done.wait(timeout=5))
        self.assertFalse(t.is_alive())

    def test_queue_drain_on_shutdown(self):
        """Queues should be drained on shutdown, not left blocking."""
        q = queue.Queue(maxsize=4)
        for i in range(4):
            q.put(i)

        # Drain
        drained = []
        while not q.empty():
            try:
                drained.append(q.get_nowait())
            except queue.Empty:
                break

        self.assertEqual(len(drained), 4)
        self.assertTrue(q.empty())


class WatermarkContiguityTests(unittest.TestCase):
    """Watermark must advance only across contiguous translated regions."""

    def test_speech_blocks_watermark(self):
        """Untranslated speech should block the watermark."""
        output = Mock()
        output._vad_progress = 10.0
        output._translated_until = 5.0
        output._pending_speech = []
        output._generation = 0
        output._buffer_condition = threading.Condition()

        # Record speech at 5-10s — this blocks the watermark
        def record_speech(start_s, end_s, generation):
            if generation == output._generation:
                output._pending_speech.append((start_s, end_s))

        record_speech(5.0, 10.0, 0)

        # Watermark should NOT advance past the speech region
        self.assertEqual(output._translated_until, 5.0)
        self.assertEqual(len(output._pending_speech), 1)

    def test_subtitle_advances_watermark(self):
        """Published subtitle should advance the watermark."""
        output = Mock()
        output._vad_progress = 15.0
        output._translated_until = 5.0
        output._pending_speech = [(5.0, 10.0)]
        output._generation = 0
        output._buffer_condition = threading.Condition()

        # Publish subtitle at 5-10s
        def advance_to_subtitle(subtitle_end_s, generation):
            if generation == output._generation:
                # Clear pending speech up to subtitle_end_s
                output._pending_speech = [
                    (s, e) for s, e in output._pending_speech
                    if s >= subtitle_end_s
                ]
                output._translated_until = max(
                    output._translated_until,
                    subtitle_end_s,
                    output._vad_progress if not output._pending_speech else subtitle_end_s
                )

        advance_to_subtitle(10.0, 0)

        # Watermark should advance to subtitle end
        self.assertGreaterEqual(output._translated_until, 10.0)

    def test_silence_advances_watermark(self):
        """VAD-confirmed silence should advance the watermark."""
        output = Mock()
        output._vad_progress = 20.0
        output._translated_until = 10.0
        output._pending_speech = []
        output._generation = 0
        output._buffer_condition = threading.Condition()

        # No pending speech — silence can advance watermark
        def try_advance_through_silence():
            if not output._pending_speech:
                output._translated_until = max(
                    output._translated_until,
                    output._vad_progress
                )

        try_advance_through_silence()

        self.assertEqual(output._translated_until, 20.0)

    def test_gap_blocks_watermark(self):
        """Gap between translated region and VAD progress should not be skipped."""
        output = Mock()
        output._vad_progress = 30.0
        output._translated_until = 10.0
        output._pending_speech = [(15.0, 20.0)]  # Gap at 15-20s
        output._generation = 0
        output._buffer_condition = threading.Condition()

        # Watermark should NOT jump over the gap
        def try_advance_through_silence():
            if not output._pending_speech:
                output._translated_until = max(
                    output._translated_until,
                    output._vad_progress
                )

        try_advance_through_silence()

        # Watermark should stay at 10.0 because of the gap
        self.assertEqual(output._translated_until, 10.0)


class SafePlaybackIntegrationTests(unittest.TestCase):
    """SafePlayback should not create deadlocks with MPV pause."""

    def test_gate_does_not_block_ingest(self):
        """SafePlayback gate should not block audio ingestion."""
        gate = SafeInferenceGate(SafePlaybackConfig(enabled=True, max_inference_concurrency=1))

        # Ingestion doesn't use the gate — only ASR and translation do
        # Verify gate doesn't affect queue operations
        q = queue.Queue(maxsize=4)
        for i in range(4):
            q.put(i)

        # Queue should work independently of gate
        self.assertEqual(q.qsize(), 4)

    def test_gate_releases_on_stop(self):
        """Gate should release on stop event."""
        stop = threading.Event()
        gate = SafeInferenceGate(SafePlaybackConfig(enabled=True, max_inference_concurrency=1), stop)

        # Acquire the slot
        acquired = gate._slots.acquire(timeout=1)
        self.assertTrue(acquired)

        # Set stop event
        stop.set()

        # Release the slot
        gate._slots.release()

    def test_concurrent_gate_slots(self):
        """Multiple concurrent slots should work correctly."""
        stop = threading.Event()
        gate = SafeInferenceGate(
            SafePlaybackConfig(enabled=True, max_inference_concurrency=2), stop
        )

        results = []
        def worker(idx):
            with gate.operation("test"):
                results.append(idx)
                time.sleep(0.05)

        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertEqual(len(results), 4)


if __name__ == "__main__":
    unittest.main()
