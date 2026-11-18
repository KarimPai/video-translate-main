"""Observed artifacts and negative controls, without claiming acoustic accuracy."""
from pathlib import Path
import queue
import unittest
from dataclasses import replace

from auto_translate.audio_events import AudioEvent, EventHistory
from auto_translate.config import TranslateConfig, load_config
from auto_translate.text_artifacts import clean_asr_text
from auto_translate.subtitle_clean import sanitize_english_output
from auto_translate.pipeline import run_translate
from auto_translate.types import Transcript, Subtitle, AudioProgress
from auto_translate.session_cache import SessionSubtitleCache
from test_english_reliability import fake_translator


class ArtifactTests(unittest.TestCase):
    def test_all_observed_malformed_tags(self):
        for tag in ('smtbarking', 'HClackank', 'giggy', 'SYak', 'SGigh', 'CBDurring'):
            with self.subTest(tag=tag):
                self.assertEqual(clean_asr_text(f'*{tag}*').text, '')
                result = clean_asr_text(f'ありがとう *{tag}* みこち')
                self.assertEqual(result.text, 'ありがとう みこち')
                self.assertTrue(result.removed)
                self.assertEqual(sanitize_english_output(f'*{tag}*'), '')

    def test_multiple_and_nonverbal_labels(self):
        for value in ('[laughter]', '*screaming*', '[crying]', '(笑い声)', 'はははは', 'hahaha'):
            self.assertEqual(clean_asr_text(value).text, '')
        self.assertEqual(clean_asr_text('*giggy* やばい! *SGigh*').text, 'やばい!')
        self.assertEqual(clean_asr_text('[laughter] えっ?').text, 'えっ?')
        self.assertEqual(clean_asr_text('[shouting] 逃げて！').text, '逃げて！')

    def test_corrupt_unicode_without_fabricating_words(self):
        self.assertEqual(clean_asr_text('ฟ ฟฟฟ �ฟありがとうฟ ฟฟฟ �ฟ').text, 'ありがとう')
        self.assertEqual(clean_asr_text('� ありがとう �').text, 'ありがとう')
        self.assertEqual(clean_asr_text('ca�nt').text, '')
        for value in ('�', '\ud800', '***!?', '\x00'):
            self.assertEqual(clean_asr_text(value).text, '')
        self.assertEqual(clean_asr_text('こんにちは\x00 world').text, 'こんにちは world')

    def test_genuine_speech_names_code_switching_and_profanity(self):
        for value in ('やばい!', 'えっ?', 'はい', 'うん', 'ああ', '草', '逃げて！',
                      'ラプ', 'ラプ様', 'ラプちゃん', 'ラプラス', 'みこ', 'みこち', 'わため',
                      'まつり', 'ぼたん', 'Minecraftで遊ぶ', 'Hello world',
                      'fuck', "fuckin'", 'shit', 'f*ck', "f*ckin'", 'sh*t',
                      '*Minecraft*', '*重要*', 'ฟ้า', '김민수', 'それで'):
            with self.subTest(value=value):
                self.assertEqual(clean_asr_text(value).text, value)

    def test_final_formatting_is_minimal_and_idempotent(self):
        cases = {
            'Pouring coffee *smtbarking* This should help me recover. I see.':
                'Pouring coffee. This should help me recover. I see.',
            'Ah—uh... what?!': 'Ah—uh… what?!',
            'Wait!!! What???': 'Wait! What?',
            'I was going to': 'I was going to',
            'No! No! No!': 'No! No! No!',
            "Don't censor f*ck or sh*t.": "Don't censor f*ck or sh*t.",
            '*Minecraft* is fun.': 'Minecraft is fun.',
            'He said "go': 'He said go',
        }
        for before, after in cases.items():
            self.assertEqual(sanitize_english_output(before), after)
            self.assertEqual(sanitize_english_output(after), after)

    def test_cache_cleans_at_insert_and_read_preserving_timestamps(self):
        cache = SessionSubtitleCache()
        sub = Subtitle(4, 6, 'やばい', '*SGigh* This is bad!', speech_start_s=4.3)
        self.assertTrue(cache.add(sub))
        self.assertEqual(cache.match(4, 6).en, 'This is bad!')
        self.assertEqual(cache.active_at(4.5)[0].speech_start_s, 4.3)
        self.assertFalse(cache.add(Subtitle(8, 9, '笑い声', '*giggy*')))
        # Simulate an older in-session entry bypassing insertion cleanup.
        cache._entries[(400, 600)].en = '*SGigh* This is bad!'
        self.assertEqual(cache.match(4, 6).en, 'This is bad!')

    def test_retry_and_history_never_receive_tags(self):
        obj = fake_translator('*SGigh*', '*giggy* Hello!')
        result = obj.translate('*smtbarking* こんにちは', [('*CBDurring* 前の会話', '')])
        self.assertEqual(result, 'Hello!')
        self.assertEqual(obj._chat.call_count, 2)
        for call in obj._chat.call_args_list:
            self.assertNotIn('*', call.args[0])
        empty = fake_translator()
        self.assertEqual(empty.translate('*SGigh*', []), '')
        empty._chat.assert_not_called()

    def test_style_default_and_prompt(self):
        style = '简洁、自然的口语，保留原文的脏话，不要审查'
        self.assertEqual(TranslateConfig().style, style)
        self.assertEqual(load_config(Path('config.example.toml')).translate.style, style)
        obj = fake_translator('Shit!')
        self.assertEqual(obj.translate('くそ！', []), 'Shit!')
        self.assertIn(style, obj._chat.call_args.args[0])
        self.assertFalse(obj.cfg.collapse_repetitions)


class EventTests(unittest.TestCase):
    def event(self, **kw):
        return replace(AudioEvent(1, 2, 'laughter', verified=True, evidence='human_audio_review'), **kw)

    def test_unverified_asr_labels_do_not_create_context(self):
        history = EventHistory()
        for event in (self.event(verified=False), self.event(evidence='BuzzASR'), self.event(kind='SGigh')):
            self.assertFalse(history.add(event))
        self.assertEqual(history.context(1, 2, 0), '')

    def test_expiration_generation_and_unknown_speaker(self):
        history = EventHistory()
        self.assertTrue(history.add(self.event()))
        self.assertIn('laughter', history.context(1.5, 2.5, 0))
        self.assertEqual(history.context(3, 4, 0), '')
        history.add(self.event(speaker_id='A', speaker_verified=True))
        self.assertIn('laughter', history.context(3, 4, 0, 'A'))
        self.assertEqual(history.context(3, 4, 0, 'B'), '')
        self.assertEqual(history.context(7, 8, 0, 'A'), '')
        self.assertEqual(len(history.events), 0)
        history.add(self.event())
        self.assertEqual(history.context(1, 2, 1), '')

    def test_event_memory_is_bounded(self):
        history = EventHistory()
        for i in range(100):
            history.add(self.event(start_s=i, end_s=i+1))
        self.assertEqual(len(history.events), 16)

    def test_pipeline_events_history_timing_and_resets(self):
        for live in (False, True):
            with self.subTest(live=live):
                obj = fake_translator('Oh!', 'Go!')
                obj.cfg.event_context = True
                source, target = queue.Queue(), queue.Queue()
                for item in (self.event(), Transcript(1, 2, '*SGigh*'),
                             Transcript(1.5, 2.5, '[laughter] えっ?', speech_start_s=1.7),
                             AudioProgress(0, reset=True, generation=1),
                             Transcript(0, 1, '行こう!', generation=1), None):
                    source.put(item)
                run_translate(obj, 6, source, target, live=live)
                output = []
                while not target.empty():
                    output.append(target.get())
                subs = [s for s in output if isinstance(s, Subtitle)]
                self.assertEqual([s.en for s in subs], ['Oh!', 'Go!'])
                self.assertEqual((subs[0].start_s, subs[0].end_s, subs[0].speech_start_s), (1.5, 2.5, 1.7))
                first, second = [c.args[0] for c in obj._chat.call_args_list]
                self.assertIn('Verified nearby audio', first)
                self.assertNotIn('SGigh', first)
                self.assertNotIn('Verified nearby audio', second)
                self.assertNotIn('えっ', second)


if __name__ == '__main__':
    unittest.main()
