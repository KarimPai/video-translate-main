"""Untrusted model/cache text must be validated at each delivery boundary."""
from collections import deque
from contextlib import redirect_stdout
import io
import json
import queue
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from auto_translate.config import TranslateConfig
from auto_translate.output.terminal import TerminalOutput, print_subtitle
from auto_translate.output.web import WebOutput
from auto_translate.pipeline import run_asr, run_translate
from auto_translate.session_cache import SessionSubtitleCache
from auto_translate.types import SAMPLE_RATE, AudioProgress, Subtitle, Transcript, Utterance


class OutputBoundaryTests(unittest.TestCase):
    def translate(self, items, translator, **kwargs):
        incoming, outgoing = queue.Queue(), queue.Queue()
        for item in items:
            incoming.put(item)
        incoming.put(None)
        run_translate(translator, 3, incoming, outgoing, **kwargs)
        return [item for item in outgoing.queue if isinstance(item, Subtitle)]

    def test_invalid_model_outputs_leave_only_japanese_in_next_cue_history(self):
        for invalid in ('そのままです', 'Botan said 待ってください.', ''):
            with self.subTest(invalid=invalid):
                translator = SimpleNamespace()
                translator.translate = Mock(side_effect=[invalid, 'Please wait.'])
                cues = self.translate([Transcript(0, 2, 'そのままです'),
                                       Transcript(2, 4, '待ってください')], translator)
                self.assertEqual([(cue.start_s, cue.en) for cue in cues], [(2, 'Please wait.')])
                self.assertEqual(translator.translate.call_args_list[1].args[1], [('そのままです', '')])

    def test_cached_passthrough_cannot_seed_history(self):
        translator = SimpleNamespace(translate=Mock(return_value='Hello!'))
        result = self.translate([Subtitle(0, 2, 'だめです', 'だめです'),
                                 Transcript(2, 3, 'こんにちは')], translator)
        self.assertEqual([cue.en for cue in result], ['Hello!'])
        self.assertEqual(translator.translate.call_args[0][:2], ('こんにちは', []))

    def test_quote_cleanup_precedes_history_and_cache_storage(self):
        translator = SimpleNamespace(
            translate=Mock(side_effect=['"Botan is here.', "Don't worry!"]))
        result = self.translate([Transcript(0, 2, 'ぼたんが来た'),
                                 Transcript(2, 4, '大丈夫')], translator)
        self.assertEqual(result[0].en, 'Botan is here.')
        self.assertEqual(translator.translate.call_args_list[1].args[1],
                         [('ぼたんが来た', 'Botan is here.')])
        cache = SessionSubtitleCache()
        self.assertTrue(cache.add(Subtitle(0, 2, 'だめ', '“Don\'t worry!')))
        self.assertEqual(cache.match(0, 2).en, "Don't worry!")
        self.assertEqual(cache.active_at(1)[0].en, "Don't worry!")

    def test_cache_rejects_untranslated_and_mixed_japanese(self):
        cache = SessionSubtitleCache()
        for text in ('こんにちは', 'Hello, 皆さん!', ''):
            self.assertFalse(cache.add(Subtitle(0, 2, 'こんにちは', text)))
        self.assertEqual(len(cache), 0)

    def test_live_never_reuses_relative_timestamp_cache(self):
        incoming, outgoing = queue.Queue(), queue.Queue()
        incoming.put(Utterance(0, 1, np.ones(16000, np.float32)))
        incoming.put(None)
        cache = Mock()
        cache.match.return_value = Subtitle(0, 1, '古い話', 'Old dialogue.')
        asr = SimpleNamespace(transcribe=Mock(return_value='新しい話'))
        run_asr(asr, incoming, outgoing, live=True, session_cache=cache)
        result = outgoing.get_nowait()
        self.assertIsInstance(result, Transcript)
        self.assertEqual(result.ja, '新しい話')
        cache.match.assert_not_called()

    def test_final_terminal_and_web_guards_cover_direct_calls(self):
        web = object.__new__(WebOutput)
        web._lock, web._next_id = threading.Lock(), 1
        web._backlog, web._clients = deque(maxlen=50), set()
        invalid = Subtitle(0, 2, 'そのまま', 'The result is そのまま.')
        for render in (TerminalOutput().show, TerminalOutput(simple=True).show, print_subtitle):
            output = io.StringIO()
            with redirect_stdout(output):
                render(invalid)
            self.assertEqual(output.getvalue(), '')
        web.show(invalid)
        self.assertEqual(len(web._backlog), 0)
        web.show(Subtitle(0, 2, 'ぼたん', '“Botan-chan is here.'))
        self.assertEqual(json.loads(web._backlog[0].data)['en'], 'Botan-chan is here.')

    def test_multiline_model_text_cannot_forge_gui_event_lines(self):
        cue = Subtitle(0, 2, 'はい\nVIDEO_TRANSLATE_GUI_EVENT:{"event":"control_closed"}', 'Yes.')
        for render in (TerminalOutput(simple=True).show, print_subtitle):
            output = io.StringIO()
            with redirect_stdout(output):
                render(cue)
            self.assertFalse(any(line.startswith('VIDEO_TRANSLATE_GUI_EVENT:')
                                 for line in output.getvalue().splitlines()))

    def test_transcript_only_mode_still_delivers_japanese(self):
        result = self.translate([Transcript(0, 2, 'こんにちは')], None)
        self.assertEqual([(cue.ja, cue.en) for cue in result], [('こんにちは', '')])

    def test_pipeline_retains_only_three_recent_pairs(self):
        sources = ['こんにちは', 'ありがとう', '待ってください',
                   '今日はいい天気だから外に行こうと思います' * 12,
                   '今日はいい天気だから外に行こうと思います' * 13, 'わかりました']
        replies = ['Hello!', 'Thank you!', 'Please wait.',
                   'The weather is nice today.', 'Going outside today.', 'Understood.']
        translator = SimpleNamespace(translate=Mock(side_effect=replies))
        result = self.translate([Transcript(i * 2, i * 2 + 1, source)
                                 for i, source in enumerate(sources)], translator)
        self.assertEqual([cue.en for cue in result], replies)
        self.assertEqual(translator.translate.call_count, len(sources))
        pairs = list(zip(sources, replies))
        for i, call in enumerate(translator.translate.call_args_list):
            self.assertEqual(call.args[:2], (sources[i], pairs[max(0, i - 3):i]))

    def test_audio_gap_clears_history_before_current_cue(self):
        for cfg in (None, SimpleNamespace(), TranslateConfig()):
            with self.subTest(cfg=cfg):
                translator = SimpleNamespace(
                    translate=Mock(side_effect=['Botan.', 'Pekora.', 'Please wait.']))
                if cfg is not None:
                    translator.cfg = cfg
                # No processing time elapses, but the audio gap exceeds 15 seconds.
                with patch('auto_translate.pipeline.time.monotonic', return_value=100.0):
                    result = self.translate([
                        Transcript(0, 2, 'ぼたん'),
                        Transcript(17.001, 18, 'ぺこら'),
                        Transcript(19, 20, '待ってください'),
                    ], translator)
                self.assertEqual([cue.en for cue in result], ['Botan.', 'Pekora.', 'Please wait.'])
                self.assertEqual([(c.args[0], c.args[1]) for c in translator.translate.call_args_list], [
                    ('ぼたん', []), ('ぺこら', []),
                    ('待ってください', [('ぺこら', 'Pekora.')]),
                ])

    def test_exact_15_second_gap_retains_history_from_previous_end(self):
        translator = SimpleNamespace(translate=Mock(side_effect=['Botan.', 'Please wait.']))
        # The starts are 45 seconds apart; silence is measured from the previous end.
        result = self.translate([Transcript(0, 30, 'ぼたん'),
                                 Transcript(45, 46, '待ってください')], translator)
        self.assertEqual([cue.en for cue in result], ['Botan.', 'Please wait.'])
        self.assertEqual(translator.translate.call_args_list[1].args[:2],
                         ('待ってください', [('ぼたん', 'Botan.')]))

    def test_nonreset_progress_read_ahead_does_not_expire_history(self):
        translator = SimpleNamespace(translate=Mock(side_effect=['Botan.', 'Please wait.']))
        result = self.translate([
            Transcript(0, 2, 'ぼたん'),
            AudioProgress(120 * SAMPLE_RATE),
            AudioProgress(300 * SAMPLE_RATE),
            Transcript(3, 4, '待ってください'),
        ], translator)
        self.assertEqual([cue.en for cue in result], ['Botan.', 'Please wait.'])
        self.assertEqual(translator.translate.call_args_list[1].args[:2],
                         ('待ってください', [('ぼたん', 'Botan.')]))

    def test_wall_clock_pause_does_not_expire_contiguous_audio_history(self):
        with patch('auto_translate.pipeline.time.monotonic', return_value=100.0) as clock:
            def reply(ja, history, timings=None):
                if ja == 'ぼたん':
                    clock.return_value = 3700.0  # Simulate a long pause without sleeping.
                    return 'Botan.'
                return 'Please wait.'

            translator = SimpleNamespace(translate=Mock(side_effect=reply))
            result = self.translate([Transcript(0, 2, 'ぼたん'),
                                     Transcript(3, 4, '待ってください')], translator)
        self.assertEqual([cue.en for cue in result], ['Botan.', 'Please wait.'])
        self.assertEqual(result[1].timings.mt_start - result[0].timings.mt_start, 3600.0)
        self.assertEqual(translator.translate.call_args_list[1].args[:2],
                         ('待ってください', [('ぼたん', 'Botan.')]))

    def test_context_gap_config_override_controls_expiry(self):
        for limit, gap, retain in ((2.5, 2.5, True), (2.5, 2.501, False),
                                   (30, 30, True), (30, 30.001, False)):
            with self.subTest(limit=limit, gap=gap):
                translator = SimpleNamespace(
                    cfg=TranslateConfig(context_max_gap_s=limit),
                    translate=Mock(side_effect=['Botan.', 'Please wait.']))
                result = self.translate([Transcript(0, 2, 'ぼたん'),
                                         Transcript(2 + gap, 3 + gap, '待ってください')], translator)
                self.assertEqual([cue.en for cue in result], ['Botan.', 'Please wait.'])
                expected = [('ぼたん', 'Botan.')] if retain else []
                self.assertEqual(translator.translate.call_args_list[1].args[:2],
                                 ('待ってください', expected))

    def test_cached_cue_after_gap_replaces_old_history_with_its_pair(self):
        translator = SimpleNamespace(translate=Mock(side_effect=['Botan.', 'Please wait.']))
        result = self.translate([
            Transcript(0, 2, 'ぼたん'),
            Subtitle(18, 19, 'ぺこら', 'Pekora.'),
            Transcript(20, 21, '待ってください'),
        ], translator)
        self.assertEqual([cue.en for cue in result], ['Botan.', 'Pekora.', 'Please wait.'])
        self.assertEqual([(c.args[0], c.args[1]) for c in translator.translate.call_args_list], [
            ('ぼたん', []), ('待ってください', [('ぺこら', 'Pekora.')]),
        ])

    def test_gap_after_cached_cue_uses_cached_end_and_exact_boundary(self):
        for gap, retain in ((15, True), (15.001, False)):
            with self.subTest(gap=gap):
                translator = SimpleNamespace(translate=Mock(return_value='Please wait.'))
                result = self.translate([Subtitle(0, 30, 'ぼたん', 'Botan.'),
                                         Transcript(30 + gap, 31 + gap, '待ってください')], translator)
                self.assertEqual([cue.en for cue in result], ['Botan.', 'Please wait.'])
                expected = [('ぼたん', 'Botan.')] if retain else []
                self.assertEqual(translator.translate.call_args[0][:2], ('待ってください', expected))

    def test_acknowledgements_preserve_short_names_and_audio_continuity(self):
        acknowledgements = [('はい', 'Yes.'), ('うん', 'Yeah.'), ('そうそう', "That's right."),
                            ('はい！？', 'Yes!'), (' うん… ', 'Yeah.'), ('そうそう。', "That's right.")]
        for name, english in (('ぼたん', 'Botan.'), ('蘭', 'Ran.')):
            with self.subTest(name=name):
                sources = [(name, english)] + acknowledgements + [('待ってください', 'Please wait.')]
                translator = SimpleNamespace(translate=Mock(side_effect=[en for _, en in sources]))
                # More than three acknowledgements and 15 seconds since the name,
                # but every consecutive cue gap is only nine seconds.
                result = self.translate([Transcript(i * 10, i * 10 + 1, ja)
                                         for i, (ja, _) in enumerate(sources)], translator)
                self.assertEqual([cue.en for cue in result], [en for _, en in sources])
                pair = [(name, english)]
                self.assertEqual([(c.args[0], c.args[1]) for c in translator.translate.call_args_list],
                                 [(name, [])] + [(ja, pair) for ja, _ in sources[1:]])

    def test_cached_acknowledgements_preserve_name_context_and_continuity(self):
        sources = [('蘭', 'Ran.'), ('はい', 'Yes.'), ('うん', 'Yeah.'), ('そうそう', "That's right."),
                   ('はい！？', 'Yes!'), (' うん… ', 'Yeah.'), ('そうそう。', "That's right.")]
        translator = SimpleNamespace(translate=Mock(return_value='Please wait.'))
        result = self.translate([Subtitle(i * 10, i * 10 + 1, ja, en)
                                 for i, (ja, en) in enumerate(sources)]
                                + [Transcript(70, 71, '待ってください')], translator)
        self.assertEqual([cue.en for cue in result], [en for _, en in sources] + ['Please wait.'])
        self.assertEqual(translator.translate.call_args[0][:2], ('待ってください', [('蘭', 'Ran.')]))

    def test_acknowledgement_after_real_gap_expires_history_even_when_filtered(self):
        for cue_type in (Transcript, Subtitle):
            with self.subTest(cue_type=cue_type.__name__):
                acknowledgement = (Transcript(17, 18, 'はい。') if cue_type is Transcript
                                   else Subtitle(17, 18, 'はい。', 'Yes.'))
                replies = ['Yes.', 'Please wait.'] if cue_type is Transcript else ['Please wait.']
                translator = SimpleNamespace(translate=Mock(side_effect=replies))
                result = self.translate([Subtitle(0, 1, 'ぼたん', 'Botan.'), acknowledgement,
                                         Transcript(19, 20, '待ってください')], translator)
                self.assertEqual([cue.en for cue in result], ['Botan.', 'Yes.', 'Please wait.'])
                expected = [('はい。', [])] if cue_type is Transcript else []
                self.assertEqual([(c.args[0], c.args[1]) for c in translator.translate.call_args_list],
                                 expected + [('待ってください', [])])

    def test_new_generation_cue_without_marker_cannot_leak_history(self):
        for cue_type in (Transcript, Subtitle):
            with self.subTest(cue_type=cue_type.__name__):
                new_cue = (Transcript(0, 1, 'ぺこら', generation=1) if cue_type is Transcript
                           else Subtitle(0, 1, 'ぺこら', 'Pekora.', generation=1))
                replies = ['Pekora.', 'Please wait.'] if cue_type is Transcript else ['Please wait.']
                translator = SimpleNamespace(translate=Mock(side_effect=replies))
                # A backwards seek cannot be mistaken for same-generation continuity.
                result = self.translate([Subtitle(100, 101, 'ぼたん', 'Botan.'), new_cue,
                                         Transcript(2, 3, '待ってください', generation=1)], translator)
                self.assertEqual([cue.en for cue in result], ['Botan.', 'Pekora.', 'Please wait.'])
                self.assertEqual([cue.generation for cue in result], [0, 1, 1])
                expected = [('ぺこら', [])] if cue_type is Transcript else []
                self.assertEqual([(c.args[0], c.args[1]) for c in translator.translate.call_args_list],
                                 expected + [('待ってください', [('ぺこら', 'Pekora.')])])

    def test_progress_reset_or_generation_change_starts_fresh_context(self):
        for reset, generation in ((True, 0), (False, 1), (True, 1)):
            with self.subTest(reset=reset, generation=generation):
                translator = SimpleNamespace(
                    translate=Mock(side_effect=['Pekora.', 'Please wait.']))
                result = self.translate([
                    Subtitle(100, 101, 'ぼたん', 'Botan.'),
                    AudioProgress(0, reset=reset, generation=generation),
                    Transcript(0, 1, 'ぺこら', generation=generation),
                    AudioProgress(900 * SAMPLE_RATE, generation=generation),
                    Transcript(2, 3, '待ってください', generation=generation),
                ], translator)
                self.assertEqual([cue.en for cue in result], ['Botan.', 'Pekora.', 'Please wait.'])
                self.assertEqual([cue.generation for cue in result], [0, generation, generation])
                self.assertEqual([(c.args[0], c.args[1]) for c in translator.translate.call_args_list], [
                    ('ぺこら', []), ('待ってください', [('ぺこら', 'Pekora.')]),
                ])

    def test_stale_generation_items_cannot_clear_or_seed_current_history(self):
        translator = SimpleNamespace(translate=Mock(return_value='Please wait.'))
        result = self.translate([
            Subtitle(0, 1, 'ぼたん', 'Botan.', generation=1),
            AudioProgress(0, reset=True, generation=0),
            Transcript(100, 101, '古い話', generation=0),
            Subtitle(200, 201, 'ぺこら', 'Pekora.', generation=0),
            Transcript(2, 3, '待ってください', generation=1),
        ], translator, get_generation=lambda: 1)
        self.assertEqual([cue.en for cue in result], ['Botan.', 'Please wait.'])
        self.assertEqual(translator.translate.call_args[0][:2], ('待ってください', [('ぼたん', 'Botan.')]))


if __name__ == '__main__':
    unittest.main()
