"""Offline regression tests for the Haas + PLaMo retest feature branch."""

import queue
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from auto_translate.config import (
    Config, IngestConfig, TranslateConfig, VadConfig, load_config,
    mpv_youtube_format,
)
from auto_translate.pipeline import run_translate
from auto_translate.subtitle_clean import (
    clean_english_output, clean_malay_output, has_generation_loop, looks_like_prompt_echo,
)
from auto_translate.types import Transcript


class FakePLaMo:
    def __init__(self, text):
        self.cfg = TranslateConfig()
        self.memory_lines = self.cfg.memory_lines
        self.text = text
        self.calls = []

    def translate(self, japanese, history):
        self.calls.append((japanese, list(history)))
        return self.text


class HaasPlamoRetestTests(unittest.TestCase):
    def _translate(self, text, japanese="これで大丈夫"):
        model = FakePLaMo(text)
        inp, out = queue.Queue(), queue.Queue()
        inp.put(Transcript(1.0, 2.0, japanese))
        inp.put(None)
        run_translate(model, 6, inp, out)
        result = []
        while not out.empty():
            item = out.get_nowait()
            if item is not None:
                result.append(item)
        return result, model

    def test_fire_red_strict_vad_with_short_reply_exception(self):
        vad = VadConfig()
        self.assertEqual(
            (vad.threshold, vad.min_speech_ms, vad.min_silence_ms,
             vad.speech_pad_ms, vad.onset_ms),
            (0.38, 200, 550, 320, 70),
        )
        self.assertEqual(vad.short_reply_ms, 120)
        self.assertEqual(vad.short_reply_probability, 0.85)
        self.assertEqual(vad.max_utterance_s + vad.max_split_extension_s, 16.0)
        self.assertEqual(vad.fallback_s, 0.0)

    def test_glossary_and_haas_and_plamo_preserved(self):
        cfg = Config()
        self.assertEqual(cfg.asr.haas_model, "haas-ja-v1.0")
        self.assertEqual(cfg.asr.haas_device, "Vulkan0")
        self.assertEqual(cfg.translate.plamo_hf_model,
                         "pfnet/plamo-3-translate-2b-GGUF:Q4_K_M")
        self.assertEqual((cfg.translate.context_lines,
                          cfg.translate.memory_lines,
                          cfg.translate.plamo_context), (4, 8, 4096))
        self.assertEqual(cfg.translate.glossary, "glossary-ms.toml")
        self.assertEqual(cfg.translate.target_lang, "ms")
        self.assertEqual(cfg.translate.max_tokens, 192)
        self.assertEqual(cfg.translate.timeout_s, 30.0)
        self.assertTrue(cfg.translate.retry_suspicious)
        self.assertEqual(cfg.translate.retry_timeout_s, 15.0)

    def test_selected_resolution_keeps_lower_fallback(self):
        for quality in (360, 480, 720, 1080):
            self.assertIn(f"[height<={quality}]", mpv_youtube_format(quality, live=True))
            self.assertIn(f"[height<={quality}]", mpv_youtube_format(quality, live=False))
        self.assertIn("protocol=https", mpv_youtube_format(720, live=False))
        self.assertIn("protocol^=m3u8", mpv_youtube_format(720, live=True))

    def test_plain_japanese_translates_into_malay(self):
        subs, model = self._translate("Saya boleh memandu kereta.", japanese="車を運転できる")
        self.assertEqual([s.en for s in subs], ["Saya boleh memandu kereta."])
        self.assertEqual(model.calls[0][0], "車を運転できる")

    def test_preserve_negation_and_hololive_names(self):
        ms = "Watame bukan Watanabe. Saya tak mahu pergi."
        self.assertEqual(clean_malay_output(ms), ms)
        result, _ = self._translate(ms)
        self.assertEqual(result[0].en, ms)

    def test_nonverbal_and_filler_cleanup(self):
        self.assertEqual(clean_english_output("Um…"), "")
        self.assertEqual(clean_english_output("Hahahaha!"), "")
        self.assertEqual(clean_english_output("Um, I don't know."), "I don't know.")
        self.assertEqual(clean_english_output("No, no, no!"), "No, no, no!")

    def test_runaway_loop_is_not_subtitled_or_added_to_memory(self):
        repeated = "Ah, it's a little weak. It's not very strong. But it's okay. "
        self.assertTrue(has_generation_loop(repeated * 7))
        subs, model = self._translate(repeated * 7)
        self.assertEqual(subs, [])
        self.assertEqual(model.calls[0][1], [])

    def test_model_context_leakage_is_rejected(self):
        leaked = (
            "Earlier Japanese for scene continuity (do not translate): "
            "わため / そうです。"
        )
        self.assertTrue(looks_like_prompt_echo(leaked))
        subs, _ = self._translate(leaked)
        self.assertEqual(subs, [])

    def test_preserve_manual_local_vad_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text("[vad]\nthreshold = 0.35\nmin_speech_ms = 100\n",
                            encoding="utf-8")
            cfg = load_config(path)
        self.assertEqual(cfg.vad.threshold, 0.35)
        self.assertEqual(cfg.vad.min_speech_ms, 100)
        self.assertEqual(cfg.vad.min_silence_ms, 550)


if __name__ == "__main__":
    unittest.main()
