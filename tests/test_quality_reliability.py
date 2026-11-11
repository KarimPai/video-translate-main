"""CPU-only coverage for adaptive speech, safe retries and quality telemetry."""
import io
import json
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from auto_translate.asr.haas import HaasASR
from auto_translate.config import AsrConfig, MpvConfig, TranslateConfig, VadConfig, vad_for_source
from auto_translate.glossary import Glossary
from auto_translate.output.mpv import MpvOutput
from auto_translate.pipeline import Stats, run_asr
from auto_translate.translate.plamo import PlamoTranslator, _needs_fragment_context
from auto_translate.types import AudioChunk, Subtitle, Timings, Utterance
from auto_translate.vad import FRAME, Segmenter


class AdaptiveVadTests(unittest.TestCase):
    def test_live_uses_shorter_natural_pause_target(self):
        cfg = VadConfig()
        live = vad_for_source(cfg, live=True)
        self.assertEqual(live.adaptive_split_s, 5.5)
        self.assertEqual(cfg.adaptive_split_s, 9.0)
        self.assertEqual(live.max_utterance_s, 7.0)

    def test_earlier_clear_pause_splits_without_losing_samples(self):
        # 12s; clear 300ms pause at 9.2s, well before the VOD 14s cap.
        frames = 1200
        pcm = np.full(frames * FRAME, .25, dtype=np.float32)
        probs = np.full(frames, .98, dtype=np.float32)
        pcm[920 * FRAME:950 * FRAME] = 0
        probs[920:950] = .01

        class FakeVAD:
            def reset(self):
                pass
            def __call__(self, audio):
                return probs[:len(audio) // FRAME]

        with patch("auto_translate.vad.StreamingFireRedVAD", return_value=FakeVAD()):
            segmenter = Segmenter(VadConfig())
            utterances = segmenter.feed(AudioChunk(0, pcm, 0.0))
            utterances += segmenter.flush()
        self.assertGreaterEqual(len(utterances), 2)
        self.assertLess(utterances[0].end_s, 10.0)
        self.assertAlmostEqual(utterances[0].end_s, utterances[1].start_s, places=5)
        self.assertTrue(np.array_equal(np.concatenate([u.audio for u in utterances]), pcm))


class HaasRecoveryTests(unittest.TestCase):
    def _read(self, results, *, live=False, vad_confidence=.95):
        class MockASR:
            def __init__(self):
                self.calls = []
            def transcribe(self, audio, probe=False):
                self.calls.append((len(audio), probe))
                return results[len(self.calls) - 1]

        asr = MockASR()
        in_q, out_q = queue.Queue(), queue.Queue()
        utt = Utterance(0.0, 1.0, np.ones(16000, dtype=np.float32),
                        vad_confidence=vad_confidence)
        in_q.put(utt)
        in_q.put(None)
        run_asr(asr, in_q, out_q, asr_cfg=AsrConfig(), live=live)
        transcript, sentinel = out_q.get(), out_q.get()
        self.assertIsNone(sentinel)
        return asr.calls, transcript

    def test_vod_high_confidence_empty_uses_one_padded_retry(self):
        calls, transcript = self._read(["", "確認した"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][0], 22400)
        self.assertEqual(transcript.ja, "確認した")

    def test_live_does_not_retry(self):
        calls, transcript = self._read(["確認した"], live=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(transcript.ja, "確認した")

    def test_no_low_confidence_retry(self):
        calls, transcript = self._read(["聞こえる"], vad_confidence=.2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(transcript.ja, "聞こえる")

    def test_hung_worker_stdout_returns_error_not_hang(self):
        asr = object.__new__(HaasASR)
        asr.cfg = AsrConfig(response_timeout_s=0.01)
        asr._responses = queue.Queue()
        asr._responses.put(None)
        asr.process = SimpleNamespace(poll=lambda: 1)
        with self.assertRaises(RuntimeError):
            asr._read_response()


class PLaMoQualityTests(unittest.TestCase):
    def test_only_short_dependant_fragments_trigger_context(self):
        self.assertTrue(_needs_fragment_context("それは"))
        self.assertTrue(_needs_fragment_context("行きたいけど"))
        self.assertFalse(_needs_fragment_context("ありがとうございます"))
        self.assertFalse(_needs_fragment_context("明日は車でゲームをしながらゆっくり帰ろうと思います"))

    def test_context_uses_previous_japanese_only_when_needed(self):
        translator = object.__new__(PlamoTranslator)
        translator.cfg = TranslateConfig()
        translator.glossary = Glossary({})
        translator.vod_title = "Example"
        translator.url = "http://127.0.0.1:8000/v1/chat/completions"
        translator.memory_lines = 0
        with patch("auto_translate.translate.plamo.urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value = io.BytesIO(
                json.dumps({"choices": [{"message": {"content": "That one."}}]}).encode()
            )
            translator._chat("それは", [("赤い車がある", "There's a red car")], 2.0)
            request = urlopen.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertIn("Immediately preceding Japanese", body["chat_template_kwargs"]["context"])
        self.assertIn("赤い車がある", body["chat_template_kwargs"]["context"])
        self.assertTrue(body["messages"][-1]["content"] == "それは")

    def test_live_stale_window(self):
        player = object.__new__(MpvOutput)
        player.cfg = MpvConfig(live_late_window_s=7.0)
        player.ingest = SimpleNamespace(source=SimpleNamespace(live=True))
        player._now = 17.0
        player._lines = []
        player._lock = threading.Lock()
        player._subtitle_ready = threading.Event()
        player.show(Subtitle(2.0, 7.0, "待って", "Wait!"))
        self.assertEqual(player._lines, [])

    def test_runtime_p95_and_deadline_misses(self):
        stats = Stats()
        stats.add(Subtitle(0, 1, "はい", "Yes", Timings(
            audio_read=10, asr_start=10, asr_end=11, mt_start=11, mt_end=20)),
            deadline_s=9.0)
        self.assertEqual(stats.deadline_misses, 1)
        self.assertIn("p95", stats.summary())


if __name__ == "__main__":
    unittest.main()
