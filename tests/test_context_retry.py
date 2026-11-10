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
        cfg = WhisperConfig(use_text_context=True, audio_context_enabled=True)
        state = RecentSpeechContext(cfg)
        state.accept(utt(0, 3), "Watch the road.")
        self.assertEqual(len(state.audio_context(utt(3.2, 4))), 19200)
        self.assertIsNotNone(state.prompt(utt(3.2, 4)))
        self.assertIsNone(state.audio_context(utt(3.2, 4, handoff=True)))
        state.prepare(utt(3.2, 4, handoff=True))
        self.assertEqual(state.last_text, "")

    def test_seek_and_long_gap_cannot_leak_history(self):
        state = RecentSpeechContext(WhisperConfig(
            use_text_context=True, audio_context_enabled=True,
        ))
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
    def test_two_matching_weak_passes_can_recover_confident_vad_speech(self):
        weak = lambda phrase, confidence: WhisperDecode(
            "", discarded_segments=1, no_speech_prob=confidence,
            weak_text=phrase, weak_avg_logprob=-1.31,
            weak_no_speech_prob=confidence,
        )
        decoder = FakeDecoder([
            weak("Get out of the car!", .08),
            weak("Get out of the car.", .10),
        ])
        cues = run_once(decoder, [utt()])
        self.assertEqual([cue.en for cue in cues], ["Get out of the car!"])
        self.assertEqual(len(decoder.calls), 2)

    def test_weak_rescue_rejects_disagreement_and_probable_noise(self):
        cases = [
            (WhisperDecode("", discarded_segments=1, no_speech_prob=.15,
                           weak_text="Stop the car.", weak_avg_logprob=-1.4,
                           weak_no_speech_prob=.15),
             WhisperDecode("", discarded_segments=1, no_speech_prob=.15,
                           weak_text="Turn the wheel.", weak_avg_logprob=-1.35,
                           weak_no_speech_prob=.15)),
            (WhisperDecode("", discarded_segments=1, no_speech_prob=.56,
                           weak_text="Stop the car.", weak_avg_logprob=-1.4,
                           weak_no_speech_prob=.56),
             WhisperDecode("", discarded_segments=1, no_speech_prob=.55,
                           weak_text="Stop the car.", weak_avg_logprob=-1.35,
                           weak_no_speech_prob=.55)),
        ]
        for first, second in cases:
            decoder = FakeDecoder([first, second])
            self.assertEqual(run_once(decoder, [utt()]), [])

    def test_uncertain_audio_retried_once(self):
        decoder = FakeDecoder([
            WhisperDecode("Not good.", avg_logprob=-.95, decoded_segments=1),
            WhisperDecode("That's great!", avg_logprob=-.3, decoded_segments=1),
        ])
        cues = run_once(decoder, [utt()])
        self.assertEqual([c.en for c in cues], ["Not good."])
        self.assertEqual(len(decoder.calls), 2)
        self.assertEqual(decoder.calls[-1][-1], .2)

    def test_loop_retry_is_unprompted_and_preserves_source_time(self):
        cfg = WhisperConfig(hotwords=["Watame"])
        decoder = FakeDecoder([WhisperDecode("get out " * 4),
                               WhisperDecode("Get out!")], cfg)
        cues = run_once(decoder, [utt(10, 13)])
        self.assertEqual([(c.start_s, c.end_s, c.en) for c in cues],
                         [(10, 13, "Get out!")])
        self.assertEqual(decoder.calls[-1], (48000, None, None, .2))

    def test_implausible_length_does_not_publish_an_unrelated_retry(self):
        decoder = FakeDecoder([WhisperDecode("a long sentence " * 8),
                               WhisperDecode("I won!")])
        self.assertEqual(run_once(decoder, [utt()]), [])
        self.assertEqual(len(decoder.calls), 2)

    def test_weak_vad_and_decoder_silence_require_corroboration(self):
        audio = utt()
        audio.vad_confidence = .55
        first = WhisperDecode("That's great.", avg_logprob=-.45, no_speech_prob=.69)
        for second, expected in [
            (WhisperDecode("No!", avg_logprob=-.2, no_speech_prob=.2), []),
            (WhisperDecode("That's great!", avg_logprob=-.3, no_speech_prob=.2), ["That's great!"]),
        ]:
            model = FakeDecoder([first, second])
            self.assertEqual([c.en for c in run_once(model, [audio])], expected)

    def test_retry_audio_limit(self):
        model = FakeDecoder([WhisperDecode("Watame!", avg_logprob=-.9)])
        self.assertEqual(run_once(model, [utt(0, 13)])[0].en, "Watame!")
        self.assertEqual(len(model.calls), 1)

    def test_short_interjections_and_genuine_signoffs(self):
        for text in ("Hmm!", "No!", "Thank you for watching.", "Haha, Watame!"):
            model = FakeDecoder([WhisperDecode(text, avg_logprob=-.2)])
            self.assertEqual(run_once(model, [utt()])[0].en, text)

    def test_confident_audio_does_not_retry(self):
        decoder = FakeDecoder([WhisperDecode("Let's go!", avg_logprob=-.2)])
        self.assertEqual(run_once(decoder, [utt()])[0].en, "Let's go!")
        self.assertEqual(len(decoder.calls), 1)

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
        self.assertEqual(fake.kwargs["beam_size"], 5)
        self.assertEqual(fake.kwargs["initial_prompt"], "Let's go!")
        self.assertEqual(fake.kwargs["hotwords"], "Okayu")


if __name__ == "__main__":
    unittest.main()
