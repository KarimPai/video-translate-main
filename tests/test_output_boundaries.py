"""Untrusted model/cache text must be validated at each delivery boundary."""
from collections import deque
from contextlib import redirect_stdout
import io
import json
import queue
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from auto_translate.output.terminal import TerminalOutput, print_subtitle
from auto_translate.output.web import WebOutput
from auto_translate.pipeline import run_asr, run_translate
from auto_translate.session_cache import SessionSubtitleCache
from auto_translate.types import Subtitle, Transcript, Utterance


class OutputBoundaryTests(unittest.TestCase):
    def translate(self, items, translator):
        incoming, outgoing = queue.Queue(), queue.Queue()
        for item in items:
            incoming.put(item)
        incoming.put(None)
        run_translate(translator, 3, incoming, outgoing)
        return [item for item in outgoing.queue if isinstance(item, Subtitle)]

    def test_invalid_model_outputs_never_enter_next_cue_history(self):
        for invalid in ('そのままです', 'Botan said 待ってください.', ''):
            with self.subTest(invalid=invalid):
                translator = SimpleNamespace()
                translator.translate = Mock(side_effect=[invalid, 'Please wait.'])
                cues = self.translate([Transcript(0, 2, 'そのままです'),
                                       Transcript(2, 4, '待ってください')], translator)
                self.assertEqual([(cue.start_s, cue.en) for cue in cues], [(2, 'Please wait.')])
                self.assertEqual(translator.translate.call_args_list[1].args[1], [])

    def test_cached_passthrough_cannot_seed_history(self):
        translator = SimpleNamespace(translate=Mock(return_value='Hello!'))
        result = self.translate([Subtitle(0, 2, 'だめです', 'だめです'),
                                 Transcript(2, 3, 'こんにちは')], translator)
        self.assertEqual([cue.en for cue in result], ['Hello!'])
        translator.translate.assert_called_once_with('こんにちは', [])

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
            self.assertEqual(call.args, (sources[i], pairs[max(0, i - 3):i]))


if __name__ == '__main__':
    unittest.main()
