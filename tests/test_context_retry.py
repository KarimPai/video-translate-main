"""Unit regressions for bounded Whisper context and selective retry."""
import logging
import queue
import unittest
from types import SimpleNamespace

import numpy as np

from auto_translate.config import WhisperConfig
from auto_translate.context import RecentSpeechContext, strip_context_echo
from auto_translate.pipeline import run_whisper
from auto_translate.types import AudioProgress, Subtitle, Utterance
from auto_translate.whisper_v3 import WhisperDecode, WhisperLargeV3


def utt(start=0., end=1., generation=0, handoff=False):
    return Utterance(
        start, end, np.full(round((end-start)*16000), .1, dtype=np.float32),
        generation=generation, speaker_handoff=handoff,
        vad_confidence=.92,
    )


def run_once(engine, items):
    source, result = queue.Queue(), queue.Queue()
    for item in items:
        source.put(item)
    source.put(None)
    run_whisper(engine, source, result)
    out = []
    while (item := result.get_nowait()) is not None:
        out.append(item)
    return out


class ContextTests(unittest.TestCase):
    def test_tail_is_bounded_and_clear_on_handoff(self):
        cfg = WhisperConfig(use_text_context=True)
        state = RecentSpeechContext(cfg)
        state.accept(utt(0, 3), "Watch the road.")
        self.assertEqual(len(state.audio_context(utt(3.2, 4))), 19200)
        self.assertIsNotNone(state.prompt(utt(3.2, 4)))
        self.assertIsNone(state.audio_context(utt(3.2, 4, handoff=True)))
        state.prepare(utt(3.2, 4, handoff=True))
        self.assertEqual(state.last_text, "")

    def test_seek_and_long_gap_cannot_leak_history(self):
        state = RecentSpeechContext(WhisperConfig(use_text_context=True))
        state.accept(utt(0, 2), "Let's go.")
        late = utt(11, 12)
        state.prepare(late)
        self.assertIsNone(state.prompt(late))
        state.accept(late, "A new subject.")
        next_generation = utt(11, 12, generation=1)
        state.prepare(next_generation)
        self.assertIsNone(state.audio_context(next_generation))

    def test_opt_in_context_limited_to_two_lines_and_120_characters(self):
        state = RecentSpeechContext(WhisperConfig(
            use_text_context=True, text_context_chars=28,
        ))
        state.accept(utt(0, 1), "Watch out!")
        state.accept(utt(1, 2), "There's a car over there!")
        prompt = state.prompt(utt(2.2, 3))
        self.assertLessEqual(len(prompt), 28)
        self.assertIn("car over there", prompt)

    def test_optional_hotwords_are_bounded(self):
        cfg = WhisperConfig(hotwords=["Okayu", "Botan", "", "RPG", "invalid\ninstruction"])
        prompt = RecentSpeechContext(cfg).hotwords()
        self.assertIn("Okayu", prompt)
        self.assertIn("Botan", prompt)
        self.assertNotIn("\n", prompt)
        self.assertNotIn("invalid instruction", prompt)

    def test_exact_overlap_returns_only_new_text(self):
        self.assertEqual(
            strip_context_echo("We're going to the casino.",
                               "to the casino, now let's buy armor!"),
            "now let's buy armor!",
        )
        self.assertIsNone(strip_context_echo("We're going to the casino.",
                                             "Completely unrelated output."))
        self.assertIsNone(strip_context_echo("Watch out!", "Watch out!"))


class FakeDecoder:
    def __init__(self, outputs, cfg=None):
        self.cfg = cfg or WhisperConfig()
        self.outputs = iter(outputs)
        self.calls = []

    def decode_audio(self, audio, *, prompt=None, hotwords=None, temperature=None):
        self.calls.append((len(audio), prompt, hotwords, temperature))
        return next(self.outputs)


class RetryTests(unittest.TestCase):
    def test_uncertain_audio_retried_once(self):
        decoder = FakeDecoder([
            WhisperDecode("Not good.", avg_logprob=-.95, decoded_segments=1),
            WhisperDecode("That's great!", avg_logprob=-.3, decoded_segments=1),
        ])
        cues = run_once(decoder, [utt()])
        self.assertEqual([c.en for c in cues], ["That's great!"])
        self.assertEqual(len(decoder.calls), 2)
        self.assertEqual(decoder.calls[-1][-1], .2)

    def test_confident_audio_does_not_retry(self):
        decoder = FakeDecoder([WhisperDecode("Let's go!", avg_logprob=-.2)])
        self.assertEqual(run_once(decoder, [utt()])[0].en, "Let's go!")
        self.assertEqual(len(decoder.calls), 1)

    def test_contextual_retry_de_duplicates_and_preserves_timestamps(self):
        decoder = FakeDecoder([
            WhisperDecode("We're going to the casino.", avg_logprob=-.15),
            WhisperDecode("Thank you for watching.", avg_logprob=-.25),
            WhisperDecode("to the casino, now let's buy armor!", avg_logprob=-.1),
        ])
        cues = run_once(decoder, [utt(0,2), utt(2.2,4.2)])
        self.assertEqual([c.en for c in cues],
                         ["We're going to the casino.", "Now let's buy armor!"])
        self.assertGreater(decoder.calls[-1][0], decoder.calls[1][0])
        self.assertEqual([(c.start_s,c.end_s) for c in cues],
                         [(0,2),(2.2,4.2)])

    def test_unmatched_audio_overlap_keeps_primary_translation(self):
        decoder = FakeDecoder([
            WhisperDecode("We're going to the casino.", avg_logprob=-.2),
            WhisperDecode("Thank you for watching.", avg_logprob=-.2),
            WhisperDecode("Invented unrelated sentence.", avg_logprob=-.1),
        ])
        cues = run_once(decoder, [utt(0,2), utt(2.2,4.2)])
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].en, "We're going to the casino.")

    def test_context_cannot_cross_seek_reset(self):
        decoder = FakeDecoder([
            WhisperDecode("Watch out!", avg_logprob=-.2),
            WhisperDecode("Thank you for watching.", avg_logprob=-.2),
            WhisperDecode("Let's go!", avg_logprob=-.1),
        ])
        cues = run_once(decoder, [
            utt(0, 2, 0),
            AudioProgress(sample=0, generation=1, reset=True),
            utt(0, 2, 1),
        ])
        self.assertEqual([x.en for x in cues if isinstance(x, Subtitle)],
                         ["Watch out!", "Let's go!"])
        self.assertEqual(decoder.calls[1][0], 32000)

    def test_retry_can_be_disabled(self):
        decoder = FakeDecoder([
            WhisperDecode("Bad but retained.", avg_logprob=-.9),
        ], WhisperConfig(retry_uncertain=False))
        self.assertEqual(run_once(decoder, [utt()])[0].en, "Bad but retained.")
        self.assertEqual(len(decoder.calls), 1)

    def test_diagnostics_include_vad_and_whisper_scores(self):
        decoder = FakeDecoder([
            WhisperDecode("Let's go!", avg_logprob=-.2, no_speech_prob=.05,
                          compression_ratio=1.1),
        ], WhisperConfig(diagnostics=True))
        with self.assertLogs("auto_translate.pipeline", level=logging.INFO) as captured:
            run_once(decoder, [utt()])
        joined = "\n".join(captured.output)
        self.assertIn("vad_confidence=0.92", joined)
        self.assertIn("avg_logprob=-0.2", joined)
        self.assertIn("retry=False", joined)

    def test_decoder_forwards_prompt_hotwords_and_beam_six(self):
        class FakeModel:
            def transcribe(self, audio, **kwargs):
                self.kwargs = kwargs
                return iter([SimpleNamespace(
                    text="The car is ready.", avg_logprob=-.25,
                    no_speech_prob=.05, compression_ratio=1.1,
                )]), None
        fake = FakeModel()
        engine = WhisperLargeV3(WhisperConfig(), model=fake)
        result = engine.decode_audio(utt().audio, prompt="Let's go!", hotwords="Okayu")
        self.assertEqual(result.text, "The car is ready.")
        self.assertAlmostEqual(result.avg_logprob, -.25)
        self.assertEqual(fake.kwargs["beam_size"], 6)
        self.assertEqual(fake.kwargs["initial_prompt"], "Let's go!")
        self.assertEqual(fake.kwargs["hotwords"], "Okayu")


if __name__ == "__main__":
    unittest.main()
