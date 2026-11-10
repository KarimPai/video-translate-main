"""Offline tests: Whisper Large-v3 Japanese transcription -> PLaMo English.

No GPU, model downloads, or external llama.cpp server are used.
"""
from __future__ import annotations

import io
import json
import queue
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from auto_translate.config import TranslateConfig, WhisperConfig, load_config
from auto_translate.glossary import Glossary
from auto_translate.output.terminal import TerminalOutput
from auto_translate.pipeline import run
from auto_translate.two_stage import run_plamo_translate, run_whisper_asr
from auto_translate.types import AudioProgress, Subtitle, Transcript, Utterance
from auto_translate.whisper_v3 import WhisperDecode, WhisperLargeV3
from auto_translate.translate.plamo import PlamoTranslator


def audio_utt(start=1.0, end=2.0, *, generation=0, handoff=False, probe=False):
    return Utterance(
        start, end, np.full(16000, 0.1, dtype=np.float32),
        generation=generation, speaker_handoff=handoff, probe=probe,
    )


def run_stage(func, model, entries, **kwargs):
    incoming, outgoing = queue.Queue(), queue.Queue()
    for entry in entries:
        incoming.put(entry)
    incoming.put(None)
    func(model, incoming, outgoing, **kwargs)
    result = []
    while True:
        item = outgoing.get_nowait()
        if item is None:
            return result
        result.append(item)


class FakeWhisper:
    def __init__(self, texts):
        self.cfg = WhisperConfig(retry_uncertain=False)
        self.outputs = iter(texts)
        self.calls = []

    def decode_audio(self, audio, *, prompt=None, hotwords=None, temperature=None):
        self.calls.append((prompt, hotwords, temperature))
        return WhisperDecode(next(self.outputs), avg_logprob=-0.2)


class FakeTranslator:
    def __init__(self):
        self.cfg = TranslateConfig()
        self.calls = []

    def translate(self, ja, history, *, fragment_context=()):
        self.calls.append((ja, list(history), tuple(fragment_context)))
        return {"今正面の道": "The road straight ahead",
                "ところに行ったんですけど": "I went to that place, but...",
                "ありがとう": "Thank you"}.get(ja, "Translated: " + ja)


class TwoStageTests(unittest.TestCase):
    def test_expected_model_defaults_and_local_vram_headroom(self):
        cfg = WhisperConfig()
        self.assertEqual(
            (cfg.model, cfg.device, cfg.compute_type, cfg.language,
             cfg.task, cfg.beam_size),
            ("large-v3", "cuda", "int8_float16", "ja", "transcribe", 5),
        )
        self.assertEqual(TranslateConfig().plamo_context, 4096)
        self.assertIn("Q4_K_M", TranslateConfig().plamo_hf_model)

    def test_legacy_direct_translation_setting_is_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.toml"
            p.write_text(
                '[whisper]\ntask = "translate"\nbeam_size = 5\n'
                '[translate]\nplamo_context = 4096\n',
                encoding="utf-8",
            )
            loaded = load_config(p)
        self.assertEqual(loaded.whisper.task, "transcribe")
        self.assertEqual(loaded.translate.plamo_context, 4096)

    def test_whisper_faster_whisper_uses_japanese_transcription_only(self):
        class Model:
            def transcribe(self, audio, **kwargs):
                self.kwargs = kwargs
                return iter([SimpleNamespace(
                    text="ごめんマジでガチでバグってた",
                    avg_logprob=-0.2,
                    compression_ratio=1.2,
                    no_speech_prob=0.02,
                )]), None

        backend = Model()
        asr = WhisperLargeV3(WhisperConfig(), model=backend)
        self.assertEqual(
            asr.transcribe_audio(np.full(16000, .2, np.float32)),
            "ごめんマジでガチでバグってた",
        )
        self.assertEqual(backend.kwargs["task"], "transcribe")
        self.assertEqual(backend.kwargs["language"], "ja")
        self.assertEqual(backend.kwargs["beam_size"], 5)
        self.assertFalse(backend.kwargs["vad_filter"])

    def test_asr_preserves_japanese_timestamps_and_speaker_handoff(self):
        asr = FakeWhisper(["マリンが着いた", "今正面の道"])
        result = run_stage(run_whisper_asr, asr, [
            audio_utt(7, 8, handoff=True),
            audio_utt(8.2, 9.2),
        ])
        self.assertEqual([x.ja for x in result],
                         ["マリンが着いた", "今正面の道"])
        self.assertEqual([(x.start_s, x.end_s) for x in result],
                         [(7, 8), (8.2, 9.2)])
        self.assertTrue(result[0].speaker_handoff)
        self.assertEqual(asr.calls, [(None, None, None), (None, None, None)])

    def test_vad_probes_and_silence_do_not_generate_text(self):
        asr = FakeWhisper(["ありがとう"])
        silence = audio_utt()
        silence.audio[:] = 0
        result = run_stage(run_whisper_asr, asr, [
            audio_utt(probe=True), silence, audio_utt(start=5, end=6)
        ])
        self.assertEqual([x.ja for x in result], ["ありがとう"])
        self.assertEqual(len(asr.calls), 1)

    def test_uncertain_transcription_retry_same_audio_no_previous_echo(self):
        class Uncertain(FakeWhisper):
            def decode_audio(self, audio, *, prompt=None, hotwords=None, temperature=None):
                self.calls.append((len(audio), prompt, hotwords, temperature))
                if len(self.calls) == 1:
                    return WhisperDecode("怪しい", avg_logprob=-0.9)
                return WhisperDecode("危ない", avg_logprob=-0.2)
        asr = Uncertain([])
        asr.cfg = WhisperConfig()
        result = run_stage(run_whisper_asr, asr, [audio_utt()])
        self.assertEqual([x.ja for x in result], ["危ない"])
        self.assertEqual(len(asr.calls), 2)
        self.assertEqual(asr.calls[0][0], asr.calls[1][0])

    def test_translator_receives_recent_japanese_without_merging_subtitles(self):
        translator = FakeTranslator()
        output = run_stage(run_plamo_translate, translator, [
            Transcript(10.0, 10.5, "今正面の道"),
            Transcript(10.7, 11.2, "ところに行ったんですけど"),
        ])
        self.assertEqual([x.ja for x in output],
                         ["今正面の道", "ところに行ったんですけど"])
        self.assertEqual([(x.start_s, x.end_s) for x in output],
                         [(10, 10.5), (10.7, 11.2)])
        self.assertEqual(translator.calls[1][2], ("今正面の道",))
        self.assertEqual(output[1].en, "I went to that place, but...")

    def test_seek_and_speaker_handoff_clears_recent_ja(self):
        translator = FakeTranslator()
        run_stage(run_plamo_translate, translator, [
            Transcript(0.0, .5, "今正面の道", generation=0),
            AudioProgress(0, reset=True, generation=1),
            Transcript(.7, 1.2, "ところに行ったんですけど", generation=1),
            Transcript(1.4, 1.7, "今正面の道", generation=1, speaker_handoff=True),
            Transcript(1.8, 2.1, "ところに行ったんですけど", generation=1),
        ])
        self.assertEqual(translator.calls[1][2], ())
        self.assertEqual(translator.calls[2][2], ())
        self.assertEqual(translator.calls[3][2], ("今正面の道",))

    def test_cached_subtitle_bypasses_both_models(self):
        from auto_translate.session_cache import SessionSubtitleCache
        cache = SessionSubtitleCache()
        self.assertTrue(cache.add(Subtitle(1, 2, "ありがとう", "Thank you")))
        asr = FakeWhisper([])
        cached = run_stage(run_whisper_asr, asr, [audio_utt()], session_cache=cache)
        self.assertEqual(asr.calls, [])
        self.assertIsInstance(cached[0], Subtitle)
        translator = FakeTranslator()
        output = run_stage(run_plamo_translate, translator, cached)
        self.assertEqual(output[0].en, "Thank you")
        self.assertEqual(translator.calls, [])

    def test_english_only_model_output_is_from_plamo_not_whisper(self):
        asr = FakeWhisper(["ありがとう"])
        jp = run_stage(run_whisper_asr, asr, [audio_utt()])
        tr = FakeTranslator()
        en = run_stage(run_plamo_translate, tr, jp)
        self.assertEqual((en[0].ja, en[0].en), ("ありがとう", "Thank you"))

    def test_terminal_prints_jp_en(self):
        out = io.StringIO()
        with redirect_stdout(out):
            TerminalOutput(simple=True).show(
                Subtitle(1, 2, "ごめん", "Sorry"),
            )
        self.assertEqual(out.getvalue(), "JP: ごめん\nEN: Sorry\n")

    def test_runtime_requires_plamo(self):
        with self.assertRaisesRegex(RuntimeError, "PLaMo translator is required"):
            run(None, None, None, [], None)

    def test_plamo_prompt_and_numbers_are_kept(self):
        model = PlamoTranslator.__new__(PlamoTranslator)
        model.cfg = TranslateConfig()
        model.memory_lines = 24
        model.glossary = Glossary({"マリン": "Marine"})
        model.vod_title = "Hololive GTA"
        model.url = "http://127.0.0.1:8000/v1/chat/completions"
        requests = []

        def fake_urlopen(req, timeout=None):
            requests.append(json.loads(req.data.decode("utf-8")))
            return io.BytesIO(
                b'{"choices":[{"message":{"content":"Marin goes to 828."}}]}'
            )

        with patch("auto_translate.translate.plamo.urllib.request.urlopen",
                   side_effect=fake_urlopen):
            english = model.translate(
                "マリンは八千二十八に行った", [],
                fragment_context=("今正面の道",),
            )
        self.assertEqual(english, "Marine goes to 8028.")
        body = requests[0]
        self.assertEqual(body["messages"][-1]["content"],
                         "マリンは八千二十八に行った")
        self.assertIn("今正面の道", body["chat_template_kwargs"]["context"])
        self.assertEqual(body["chat_template_kwargs"]["glossary"][0],
                         {"source": "マリン", "target": "Marine"})


if __name__ == "__main__":
    unittest.main()
