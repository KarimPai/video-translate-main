"""Regression cases from the read-only review, without loading models."""

import queue
import threading
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock, patch

from auto_translate.config import Config, TranslateConfig
from auto_translate.context_memory import split_completed_japanese_sentences
from auto_translate.glossary import Glossary
from auto_translate.pipeline import StageActivity, run, run_asr, run_translate
from auto_translate.queue_utils import QueueStopped, get_until_stopped, put_until_stopped
from auto_translate.subject_guard import neutralize_unwarranted_gender, unsupported_person_reference
from auto_translate.translate.base import (
    TranslationCancelled, TranslationDeadlineExceeded, TruncatedTranslation,
)
from auto_translate.translate.plamo import PlamoTranslator
from auto_translate.types import Subtitle, Timings


class ReviewPipelineTests(unittest.TestCase):
    @staticmethod
    def translator(**overrides):
        obj = object.__new__(PlamoTranslator)
        obj.cfg = TranslateConfig(**overrides)
        obj.glossary = Glossary({})
        obj.vod_title = ""
        obj.memory_lines = 0
        obj.validation_stats = Counter()
        obj.url = "http://127.0.0.1:8000/v1/chat/completions"
        return obj

    def test_long_in_flight_work_has_a_deadline_not_an_output_rate_requirement(self):
        now = [0.0]
        with patch("auto_translate.pipeline.time.monotonic", side_effect=lambda: now[0]):
            activity = StageActivity()
            with activity.track("asr", 150):
                now[0] = 100
                activity.check(now[0], 0, {"audio": 16})  # valid cold-start work
                now[0] = 149
                activity.heartbeat()
                with self.assertRaisesRegex(RuntimeError, "asr"):
                    activity.check(151, 150, {})  # heartbeat/output cannot extend lease
            now[0] = 250
            with self.assertRaisesRegex(RuntimeError, "stalled"):
                activity.check(now[0], 0, {"audio": 1})

    def test_queue_get_and_full_put_exit_without_a_producer_sentinel(self):
        stop = threading.Event()
        full = queue.Queue(maxsize=1)
        full.put("old work")
        finished = []
        worker = threading.Thread(target=lambda: finished.append(put_until_stopped(full, "new", stop)))
        worker.start()
        stop.set()
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(finished, [False])
        with self.assertRaises(QueueStopped):
            get_until_stopped(queue.Queue(), stop)
        self.assertFalse(put_until_stopped(full, None, stop))

    def test_pipeline_stages_exit_when_upstream_died_without_eof(self):
        for target, args in ((run_asr, (Mock(),)), (run_translate, (None, 0))):
            stop = threading.Event()
            inputs, outputs = queue.Queue(), queue.Queue()
            worker = threading.Thread(target=target, args=(*args, inputs, outputs), kwargs={"stop": stop})
            worker.start()
            stop.set()
            worker.join(1)
            self.assertFalse(worker.is_alive())
            self.assertIsNone(outputs.get_nowait())

    def test_cancel_between_sentences_does_not_translate_the_next_sentence(self):
        translator = self.translator()
        cancelled = threading.Event()
        def first(_ja, _history):
            cancelled.set()
            return "Wait right here."
        translator._translate_one = Mock(side_effect=first)
        with self.assertRaises(TranslationCancelled):
            translator.translate_cancellable(
                "絶対にここで待ってください。すぐにこちらへ戻ってきます。", [],
                cancelled=cancelled.is_set,
            )
        self.assertEqual(translator._translate_one.call_count, 1)

    def test_cancel_before_quality_retry_is_not_swallowed(self):
        translator = self.translator()
        cancelled = threading.Event()
        def chat(*_args, **_kwargs):
            cancelled.set()
            return "He left."
        translator._chat = Mock(side_effect=chat)
        with self.assertRaises(TranslationCancelled):
            translator.translate_cancellable("もう出た", [], cancelled=cancelled.is_set)
        self.assertEqual(translator._chat.call_count, 1)

    def test_subject_option_disables_both_person_and_gaming_rewrites(self):
        translator = self.translator(neutral_subjects=False)
        translator._chat = Mock(return_value="My driving is bad.")
        self.assertEqual(translator.translate("運転が下手", []), "My driving is bad.")
        translator._chat.assert_called_once()

    def test_kinship_substrings_are_not_gender_evidence(self):
        self.assertEqual(unsupported_person_reference("大丈夫です", "He is fine."),
                         "unsupported_male_reference")
        self.assertEqual(unsupported_person_reference("母国へ帰る", "She went home."),
                         "unsupported_female_reference")
        self.assertEqual(unsupported_person_reference("彼岸に行きます", "He will go."),
                         "unsupported_male_reference")
        self.assertIsNone(unsupported_person_reference("父が帰ります", "He is coming home."))
        self.assertIsNone(unsupported_person_reference("お母さんが来た", "She arrived."))

    def test_unknown_her_possessive_does_not_turn_into_object_pronoun(self):
        for sentence in ("Her health is fine.", "I saw her daughter.", "Her position is unknown."):
            self.assertEqual(neutralize_unwarranted_gender("状況は不明です", sentence), sentence)
        self.assertEqual(neutralize_unwarranted_gender("見ました", "I saw her."), "I saw them.")

    def test_quoted_multi_sentence_dialogue_is_not_split_inside_quote(self):
        quoted = "「絶対にここで待っていてね。後ですぐに戻ってきます。」それでは始めましょう。"
        self.assertEqual(split_completed_japanese_sentences(quoted), [quoted])
        unquoted = "絶対にここで待っていてね。後ですぐに戻ってきます。"
        self.assertEqual(len(split_completed_japanese_sentences(unquoted)), 2)

    def test_truncated_response_regenerates_once_and_never_emits_partial(self):
        translator = self.translator(max_tokens=24)
        bodies = []
        replies = iter([
            {"finish_reason": "length", "message": {"content": "This partial"}},
            {"finish_reason": "stop", "message": {"content": "A complete subtitle."}},
        ])
        def post(_url, *, json, timeout):
            bodies.append(dict(json))
            response = Mock()
            response.json.return_value = {"choices": [next(replies)]}
            return response
        translator.client = Mock()
        translator.client.post.side_effect = post
        self.assertEqual(translator.translate("ここにいます", []), "A complete subtitle.")
        self.assertEqual([body["max_tokens"] for body in bodies], [24, 48])
        response = Mock()
        response.json.return_value = {"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]}
        translator.client.post.side_effect = None
        translator.client.post.return_value = response
        with self.assertRaises(TruncatedTranslation):
            translator.translate("ここにいます", [])

    def test_token_retry_shares_original_cue_deadline(self):
        translator = self.translator(cue_timeout_s=10, max_tokens=24)
        now, timeouts = [0.0], []
        def post(_url, *, json, timeout):
            timeouts.append(timeout)
            now[0] += 8
            response = Mock()
            response.json.return_value = {"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]}
            return response
        translator.client = Mock()
        translator.client.post.side_effect = post
        with patch("auto_translate.translate.plamo.time.monotonic", side_effect=lambda: now[0]):
            with self.assertRaises(TranslationDeadlineExceeded):
                translator.translate("ここにいます", [])
        self.assertEqual(timeouts, [10, 2])

    def test_repeated_spoken_pleas_survive_and_final_diagnostics_include_session(self):
        stop = threading.Event()
        ingest = SimpleNamespace(source=SimpleNamespace(live=False, start_s=0), generation=0,
                                 run=lambda q: q.put(None), stop=lambda: None)
        def vad(_segmenter, aq, uq, _generation, _stop=None):
            aq.get(timeout=1)
            uq.put(None)
        def asr(_asr, uq, tq, *_args):
            uq.get(timeout=1)
            tq.put(None)
        def translate(_model, _context, tq, sq, *_args):
            tq.get(timeout=1)
            for start in (0, 2, 4):
                sq.put(Subtitle(start, start + 1, "助けて！", "Help me!",
                                Timings(audio_read=10, asr_start=11, asr_end=12,
                                        mt_start=12, mt_end=14), diagnostic_session="session"))
            sq.put(None)
        output = Mock()
        output.is_manually_paused.return_value = False
        with patch("auto_translate.pipeline.Segmenter"), \
             patch("auto_translate.pipeline.run_segmenter", side_effect=vad), \
             patch("auto_translate.pipeline.run_asr", side_effect=asr), \
             patch("auto_translate.pipeline.run_translate", side_effect=translate), \
             patch("auto_translate.pipeline.append_comparison") as save:
            run(ingest, Config(), Mock(), Mock(), [output], stop)
        self.assertEqual(output.show.call_count, 3)
        self.assertEqual(save.call_count, 3)
        self.assertEqual(save.call_args.args[0]["session_id"], "session")
        self.assertEqual(save.call_args.args[0]["cue_latency_s"], 4)

    def test_fatal_vad_with_a_full_audio_queue_does_not_need_eof_to_shutdown(self):
        stop, full, producer_done = threading.Event(), threading.Event(), threading.Event()
        def produce(audio_q):
            try:
                for i in range(1000):
                    if not put_until_stopped(audio_q, object(), stop):
                        break
                    if i == 63:
                        full.set()
            finally:
                producer_done.set()
        def broken_vad(*_args):
            if not full.wait(1):
                raise RuntimeError("producer never filled audio queue")
            raise ValueError("VAD failure")
        ingest = SimpleNamespace(source=SimpleNamespace(live=False, start_s=0), generation=0,
                                 run=produce, stop=stop.set)
        asr, translator = Mock(), Mock()
        with patch("auto_translate.pipeline.run_segmenter", side_effect=broken_vad), \
             self.assertLogs("auto_translate.pipeline", "ERROR"), \
             self.assertRaisesRegex(RuntimeError, "VAD failure"):
            run(ingest, Config(), asr, translator, [], stop, segmenter=Mock())
        self.assertTrue(producer_done.is_set())
        asr.close.assert_called_once()
        translator.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
