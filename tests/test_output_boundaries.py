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
import httpx

from auto_translate.config import TranslateConfig
from auto_translate.glossary import Glossary
from auto_translate.output.terminal import TerminalOutput, print_subtitle
from auto_translate.output.web import WebOutput
from auto_translate.pipeline import run_asr, run_translate
from auto_translate.session_cache import SessionSubtitleCache
from auto_translate.translate.plamo import PlamoTranslator
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
                translator = SimpleNamespace(memory_lines=0)
                translator.translate = Mock(side_effect=[invalid, 'Please wait.'])
                cues = self.translate([Transcript(0, 2, 'そのままです'),
                                       Transcript(2, 4, '待ってください')], translator)
                self.assertEqual([(cue.start_s, cue.en) for cue in cues], [(2, 'Please wait.')])
                self.assertEqual(translator.translate.call_args_list[1].args[1], [])

    def test_cached_passthrough_cannot_seed_history(self):
        translator = SimpleNamespace(memory_lines=0, translate=Mock(return_value='Hello!'))
        result = self.translate([Subtitle(0, 2, 'だめです', 'だめです'),
                                 Transcript(2, 3, 'こんにちは')], translator)
        self.assertEqual([cue.en for cue in result], ['Hello!'])
        translator.translate.assert_called_once_with('こんにちは', [])

    def test_quote_cleanup_precedes_history_and_cache_storage(self):
        translator = SimpleNamespace(memory_lines=0,
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
        asr = SimpleNamespace(transcribe=Mock(return_value='新しい話'),
                              last_raw_transcript='新しい話\n')
        run_asr(asr, incoming, outgoing, live=True, session_cache=cache)
        result = outgoing.get_nowait()
        self.assertIsInstance(result, Transcript)
        self.assertEqual(result.ja, '新しい話')
        self.assertEqual(result.name_context['voxtral_raw'], '新しい話\n')
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

    def test_transcript_only_mode_still_delivers_japanese(self):
        result = self.translate([Transcript(0, 2, 'こんにちは')], None)
        self.assertEqual([(cue.ja, cue.en) for cue in result], [('こんにちは', '')])

    def test_alternative_profiles_retain_three_trusted_pairs_through_pipeline(self):
        # Two valid translations of long sources are unsuitable as few-shot
        # examples. They must not displace earlier trusted short exchanges.
        for profile in ('overlay', 'hybrid'):
            with self.subTest(profile=profile):
                bodies = []
                replies = iter(['Hello!', 'Thank you!', 'Please wait.',
                                'The weather is nice today.', 'Going outside today.', 'Understood.'])
                def respond(request):
                    bodies.append(json.loads(request.content))
                    return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                        'message': {'content': next(replies)}}]})
                translator = object.__new__(PlamoTranslator)
                translator.cfg = TranslateConfig(context_profile=profile, memory_lines=0,
                                                 sentence_context=False)
                translator.memory_lines, translator.vod_title = 0, ''
                translator.glossary = Glossary({})
                translator.url = 'http://127.0.0.1:8000/v1/chat/completions'
                translator.client = httpx.Client(transport=httpx.MockTransport(respond))
                self.addCleanup(translator.client.close)
                sources = ['こんにちは', 'ありがとう', '待ってください',
                           '今日はいい天気だから外に行こうと思います' * 12,
                           '今日はいい天気だから外に行こうと思います' * 13, 'わかりました']
                self.translate([Transcript(i * 2, i * 2 + 1, source)
                                for i, source in enumerate(sources)], translator)
                examples = [message['content'] for message in bodies[-1]['messages']
                            if message['role'] == 'assistant']
                self.assertEqual(examples, ['Hello!', 'Thank you!', 'Please wait.'])


if __name__ == '__main__':
    unittest.main()
