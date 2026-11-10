"""Offline accuracy-safety regressions for the Anime Whisper experiment.

These tests use synthetic ASR hypotheses. They don't claim real Japanese
recognition accuracy without audio-ground-truth evaluation.
"""
from __future__ import annotations

import io
import json
import queue
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from auto_translate.anime_whisper import AnimeWhisperASR
from auto_translate.config import WhisperConfig, TranslateConfig
from auto_translate.glossary import Glossary
from auto_translate.translation_integrity import contextual_translation_hints
from auto_translate.translate.plamo import PlamoTranslator
from auto_translate.two_stage import run_whisper_asr
from auto_translate.types import Transcript, Utterance
from auto_translate.whisper_v3 import WhisperDecode


def run_asr(hypotheses: list[WhisperDecode], *, cfg: WhisperConfig | None = None):
    cfg = cfg or WhisperConfig(device="cpu", compute_type="float32")
    asr = AnimeWhisperASR(cfg, model=object(), processor=object())
    reads = iter(hypotheses)
    calls: list[tuple[float | None, int | None]] = []

    def fake_decode(audio, *, prompt=None, hotwords=None,
                    temperature=None, beam_size=None):
        if prompt is not None or hotwords is not None:
            raise AssertionError("ASR must never be primed")
        calls.append((temperature, beam_size))
        return next(reads)

    asr.decode_audio = fake_decode
    q_in, q_out = queue.Queue(), queue.Queue()
    q_in.put(Utterance(
        1.0, 2.0, np.full(16000, 0.2, dtype=np.float32),
    ))
    q_in.put(None)
    run_whisper_asr(asr, q_in, q_out)
    results = []
    while True:
        output = q_out.get_nowait()
        if output is None:
            break
        results.append(output)
    return results, calls


class ContextualTranslationTests(unittest.TestCase):
    def test_wireless_mouse_homophones_need_mouse_context(self):
        hints = contextual_translation_hints(
            "無線マウスに慣れると遊星に戻れない")
        self.assertTrue(any("有線" in hint for hint in hints))
        hints = contextual_translation_hints("無銭のマウス使っちゃった")
        self.assertTrue(any("無線のマウス" in hint for hint in hints))
        self.assertFalse(contextual_translation_hints("遊星の写真を見た"))

    def test_collision_hint_requires_traffic_context_not_gunfire(self):
        self.assertTrue(contextual_translation_hints(
            "車が弾付き事故になった"))
        self.assertFalse(contextual_translation_hints(
            "銃で撃つ弾付き事故"))
        self.assertFalse(contextual_translation_hints(
            "弾付き事故"))

    def test_observed_ferris_wheel_and_honeymoon_variants(self):
        self.assertTrue(contextual_translation_hints(
            "観覧車の上で覡断車が見えた"))
        self.assertFalse(contextual_translation_hints(
            "覡断車が見えた"))
        self.assertTrue(contextual_translation_hints(
            "挙式終わってハレムーンかな"))
        self.assertFalse(contextual_translation_hints(
            "ハレムーン"))

    def test_glossary_has_wholesome_tee_tee_and_traffic_terms(self):
        glossary = Glossary.load(Path("glossary.toml"))
        hits = dict(glossary.find("てーてーしてるな"))
        self.assertIn("てーてーしてる", hits)
        self.assertIn("affectionate", hits["てーてーしてる"])
        self.assertEqual(
            dict(glossary.find("玉突き事故になった"))["玉突き事故"],
            "multi-car pile-up",
        )
        self.assertEqual(
            dict(glossary.find("無線マウス使ってる"))["無線マウス"],
            "wireless mouse",
        )

    def test_plamo_context_hints_do_not_rewrite_raw_asr(self):
        translator = PlamoTranslator.__new__(PlamoTranslator)
        translator.cfg = TranslateConfig()
        translator.glossary = Glossary({})
        translator.vod_title = "Hololive gaming"
        translator.url = "http://127.0.0.1:8000/v1/chat/completions"
        bodies = []

        def fake_request(req, timeout=None):
            body = json.loads(req.data.decode("utf-8"))
            bodies.append(body)
            return io.BytesIO(
                b'{"choices":[{"message":{"content":"I cannot go back to wired."}}]}'
            )

        with patch("auto_translate.translate.plamo.urllib.request.urlopen",
                   side_effect=fake_request):
            translator._chat(
                "無銭のマウス使ったら遊星に戻れない", [],
                timeout=1.0,
            )
        self.assertEqual(
            bodies[0]["messages"][-1]["content"],
            "無銭のマウス使ったら遊星に戻れない",
        )
        context = bodies[0]["chat_template_kwargs"]["context"]
        self.assertIn("無線のマウス", context)
        self.assertIn("有線", context)


class SelectiveRetryTests(unittest.TestCase):
    def test_confident_speech_never_retried(self):
        results, calls = run_asr([
            WhisperDecode("車が来た", avg_logprob=-0.2),
        ])
        self.assertEqual(len(calls), 1)
        self.assertEqual(results[0].ja, "車が来た")

    def test_low_confidence_alternative_needs_score_gain_and_similarity(self):
        results, calls = run_asr([
            WhisperDecode("無線マウスに慣れた", avg_logprob=-1.1),
            WhisperDecode("無線マウスに慣れたよ", avg_logprob=-0.3),
        ])
        self.assertEqual(results[0].ja, "無線マウスに慣れたよ")
        self.assertEqual(calls[1], (0.0, 8))

    def test_numbers_are_not_silently_replaced(self):
        results, calls = run_asr([
            WhisperDecode("車が2台きた", avg_logprob=-1.0),
            WhisperDecode("車が3台きた", avg_logprob=-0.2),
        ])
        self.assertEqual(results[0].ja, "車が2台きた")
        self.assertEqual(len(calls), 2)

    def test_unrelated_retry_cannot_rewrite_original(self):
        results, calls = run_asr([
            WhisperDecode("こっちから行こう", avg_logprob=-1.0),
            WhisperDecode("お菓子を買おう", avg_logprob=-0.2),
        ])
        self.assertEqual(results[0].ja, "こっちから行こう")
        self.assertEqual(len(calls), 2)

    def test_outro_can_trigger_retry_but_missing_scores_keep_first(self):
        results, calls = run_asr([
            WhisperDecode("ご視聴ありがとうございました"),
            WhisperDecode("今日はどうも"),
        ])
        self.assertEqual(results[0].ja, "ご視聴ありがとうございました")
        self.assertEqual(len(calls), 2)

    def test_clearly_better_non_outro_can_replace_suspect_outro(self):
        results, _ = run_asr([
            WhisperDecode("ご視聴ありがとうございました", avg_logprob=-1.1),
            WhisperDecode("今日はどうもありがとう", avg_logprob=-0.25),
        ])
        self.assertEqual(results[0].ja, "今日はどうもありがとう")


if __name__ == "__main__":
    unittest.main()
