"""Offline tests for VOD fragment joining and rolling ASR context."""
import queue
import threading
import unittest
from unittest.mock import Mock

import numpy as np

from auto_translate import pipeline
from auto_translate.pipeline import _join_fragments
from auto_translate.types import AudioProgress, SourceEOF, Subtitle, Transcript, Utterance

SR = 16000


def joined(*items, gap=.6, span=16.):
    return list(_join_fragments(iter(items), gap, span))


class FragmentJoinTests(unittest.TestCase):
    def test_mid_sentence_cue_joins_close_continuation(self):
        first, second = Transcript(0, 2, "今日はみんなで", generation=1), Transcript(2.4, 4, "ゲームします。", generation=1)
        early, late = AudioProgress(int(2.3 * SR), generation=1), AudioProgress(5 * SR, generation=1)
        result = joined(first, early, second, late)
        self.assertEqual([type(item) for item in result], [Transcript, AudioProgress, AudioProgress])
        self.assertEqual((result[0].start_s, result[0].end_s, result[0].ja), (0, 4, "今日はみんなでゲームします。"))
        self.assertIs(result[0].timings, first.timings)
        # Progress that could precede the continuation waits behind the merged cue.
        self.assertEqual(result[1:], [early, late])

    def test_complete_sentences_pass_through(self):
        cues = [Transcript(0, 1, "はい。"), Transcript(1.1, 2, "そうだね！"), Transcript(2.1, 3, "えっ？")]
        self.assertEqual(joined(*cues), cues)

    def test_limits_and_boundaries_flush_the_held_cue(self):
        held = Transcript(0, 2, "それで")
        for following in (Transcript(3, 4, "行こう。"), Transcript(2.2, 17, "行こう。"),
                          Transcript(2.2, 3, "行こう。", generation=1), Subtitle(2.2, 3, "行こう。", "Let's go."),
                          AudioProgress(0, reset=True), AudioProgress(3 * SR), SourceEOF(), None):
            with self.subTest(following=following):
                self.assertEqual(joined(held, following), [held, following])

    def test_unfinished_continuation_can_keep_joining_within_span(self):
        result = joined(Transcript(0, 2, "あの"), Transcript(2.2, 4, "えっと"), Transcript(4.1, 6, "行こう。"))
        self.assertEqual([(cue.start_s, cue.end_s, cue.ja) for cue in result], [(0, 6, "あのえっと行こう。")])

    def test_translation_thread_merges_only_when_enabled(self):
        for merge, expected in ((None, ["今日は", "晴れです。"]), ((.6, 16.), ["今日は晴れです。"])):
            with self.subTest(merge=merge):
                translator = Mock(spec=["translate"])
                translator.translate.return_value = "It's sunny."
                incoming, outgoing = queue.Queue(), queue.Queue()
                for item in (Transcript(0, 1, "今日は"), Transcript(1.2, 2, "晴れです。"), None):
                    incoming.put(item)
                pipeline.run_translate(translator, 3, incoming, outgoing, merge=merge)
                self.assertEqual([call.args[0] for call in translator.translate.call_args_list], expected)


class LiveCommitPolicyTests(unittest.TestCase):
    def test_tick_releases_held_fragment_only_when_playback_nears_onset(self):
        held = Transcript(10, 12, "それで", generation=1, speech_start_s=10.5)
        progress = AudioProgress(int(12.3 * SR), generation=1)
        position = [0.0]
        due = lambda cue: position[0] >= cue.speech_start_s - 1.5  # noqa: E731
        stream = _join_fragments(iter([held, progress, pipeline._TICK]), .6, 16., due=due)
        self.assertEqual(list(stream), [])  # not due yet; still held at the end
        position[0] = 9.0
        result = list(_join_fragments(iter([held, progress, pipeline._TICK, None]), .6, 16., due=due))
        self.assertEqual(result, [held, progress, None])

    def test_already_due_fragment_is_never_held(self):
        cue = Transcript(10, 12, "それで")
        self.assertEqual(list(_join_fragments(iter([cue]), .6, 16., due=lambda _: True)), [cue])

    def test_continuation_before_deadline_merges(self):
        items = [Transcript(0, 1, "今日は"), pipeline._TICK, Transcript(1.2, 2, "晴れです。")]
        result = list(_join_fragments(iter(items), .6, 16., due=lambda _: False))
        self.assertEqual([cue.ja for cue in result], ["今日は晴れです。"])

    def test_live_translation_thread_merges_and_releases_on_timeout(self):
        translator = Mock(spec=["translate"])
        translator.translate.return_value = "Today."
        incoming, outgoing = queue.Queue(), queue.Queue()
        incoming.put(Transcript(0, 1, "今日は", speech_start_s=0))
        incoming.put(Transcript(1.2, 2, "晴れです。"))
        incoming.put(Transcript(30, 31, "それで", speech_start_s=30))
        position = [-5.0]
        worker = threading.Thread(target=pipeline.run_translate, args=(
            translator, 3, incoming, outgoing), kwargs=dict(
            live=True, playback_position=lambda: position[0], live_late_window_s=1000,
            merge=(.6, 16.), live_merge_margin_s=1.5), daemon=True)
        worker.start()
        first = outgoing.get(timeout=5)
        self.assertEqual(first.ja, "今日は晴れです。")
        with self.assertRaises(queue.Empty):
            outgoing.get(timeout=pipeline.MERGE_TICK_S * 3)  # held: playback far from 30s
        position[0] = 29.0
        self.assertEqual(outgoing.get(timeout=5).ja, "それで")
        incoming.put(None)
        worker.join(5)


class AsrContinuityTests(unittest.TestCase):
    def run_asr(self, asr, *items):
        incoming, outgoing = queue.Queue(), queue.Queue()
        for item in (*items, None):
            incoming.put(item)
        pipeline.run_asr(asr, incoming, outgoing)
        return [item for item in outgoing.queue if isinstance(item, Transcript)]

    def utterance(self, start, end, generation=0):
        return Utterance(start, end, np.full(int((end - start) * SR), .1, dtype=np.float32), generation=generation)

    def test_accepted_transcripts_are_remembered_and_gaps_forget(self):
        asr = Mock(spec=["transcribe", "remember", "forget", "cancel_pending"])
        asr.transcribe.side_effect = ["こんにちは。", "今日はゲームです。", "おかえり。"]
        events = []
        asr.remember.side_effect = lambda text: events.append(("remember", text))
        asr.forget.side_effect = lambda: events.append(("forget",))
        cues = self.run_asr(asr, self.utterance(0, 2), self.utterance(3, 5), self.utterance(30, 32))
        self.assertEqual(len(cues), 3)
        self.assertEqual(events, [("remember", "こんにちは。"), ("remember", "今日はゲームです。"),
                                  ("forget",), ("remember", "おかえり。")])

    def test_rejected_transcripts_are_not_remembered(self):
        asr = Mock(spec=["transcribe", "remember", "forget", "cancel_pending", "last_rejection"])
        asr.last_rejection = "empty"
        asr.transcribe.side_effect = ["", "はい。"]
        self.run_asr(asr, self.utterance(0, 2), self.utterance(3, 5))
        asr.remember.assert_called_once_with("はい。")

    def test_seek_reset_clears_recent_transcripts(self):
        asr = Mock(spec=["transcribe", "remember", "forget", "cancel_pending"])
        asr.transcribe.side_effect = ["こんにちは。", "おかえり。"]
        self.run_asr(asr, self.utterance(0, 2), AudioProgress(0, reset=True), self.utterance(30, 32))
        # Reset clears the ASR's own history; no stale gap triggers a second forget.
        self.assertGreaterEqual(asr.cancel_pending.call_count, 2)
        asr.forget.assert_not_called()


if __name__ == "__main__":
    unittest.main()
