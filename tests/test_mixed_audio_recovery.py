"""Behavioral regressions for Qwen mixed-language gameplay audio recovery."""

import queue
import threading
from unittest import mock

import numpy as np

from auto_translate.asr.qwen3_asr import Qwen3ASR, parse_qwen_asr_response
from auto_translate.config import AsrConfig
from auto_translate.pipeline import run_asr, run_translate
from auto_translate.types import Subtitle, Transcript, Utterance


def _mock_qwen():
    """Bare Qwen adapter with no GPU/server/models initialized."""
    model = object.__new__(Qwen3ASR)
    model.cfg = AsrConfig(response_timeout_s=15.0)
    model._generation = 0
    model._closed = False
    model._inference_lock = threading.Lock()
    model.last_detected_language = "en"
    model.last_raw_transcript = "We have to go."
    return model


def test_english_dominant_tag_does_not_hide_recognized_japanese_text():
    lang, transcript = parse_qwen_asr_response(
        "language English<asr_text>Go, go! 危ない、逃げて！"
    )
    assert lang == "ja"
    assert "危ない" in transcript


def test_qwen_short_windows_recover_japanese_without_forced_language():
    model = _mock_qwen()
    predictions = iter([
        ("en", "We have to go"),
        ("ja", "危ない、逃げて！"),
        ("en", "Get in the car"),
    ])
    calls = []

    def mock_decode(audio, probe=False, timeout=None, **kwargs):
        calls.append((len(audio), timeout))
        language, words = next(predictions)
        model.last_detected_language = language
        model.last_raw_transcript = words
        return words

    with mock.patch.object(model, "_transcribe_impl", side_effect=mock_decode):
        recovered = model.recover_japanese_from_mixed_audio(
            np.zeros(8 * 16000, dtype=np.float32),
        )

    assert len(calls) == 3
    assert len(recovered) == 1
    start_s, end_s, japanese = recovered[0]
    assert 2 < start_s < 3
    assert 5 < end_s < 6
    assert japanese == "危ない、逃げて！"
    assert model.last_detected_language == "en"
    assert model.last_raw_transcript == "We have to go."


def test_qwen_english_only_windows_remain_hidden():
    model = _mock_qwen()

    def mock_decode(audio, probe=False, timeout=None, **kwargs):
        model.last_detected_language = "en"
        return "Get in the car"

    with mock.patch.object(model, "_transcribe_impl", side_effect=mock_decode):
        recovered = model.recover_japanese_from_mixed_audio(
            np.zeros(6 * 16000, dtype=np.float32),
        )

    assert recovered == []


def test_qwen_too_short_or_probe_audio_does_not_trigger_second_pass():
    model = _mock_qwen()
    with mock.patch.object(model, "_transcribe_impl") as decode:
        assert model.recover_japanese_from_mixed_audio(
            np.zeros(16000, dtype=np.float32),
        ) == []
        assert model.recover_japanese_from_mixed_audio(
            np.zeros(4 * 16000, dtype=np.float32), probe=True,
        ) == []
        decode.assert_not_called()


class FakeMixedASR:
    def __init__(self):
        self.last_detected_language = "und"
        self.last_raw_transcript = ""
        self.last_rejection = None
        self.last_match_score = 0.0
        self.cfg = AsrConfig(response_timeout_s=15.0)
        self.recovery_calls = 0
        self.recover_value = [(2.0, 4.5, "危ない、逃げて！")]

    def transcribe(self, audio, probe=False):
        self.last_detected_language = "en"
        self.last_raw_transcript = "Get in the car, quick!"
        return self.last_raw_transcript

    def recover_japanese_from_mixed_audio(self, audio, probe=False):
        self.recovery_calls += 1
        return self.recover_value

    def cancel_pending(self):
        pass

    def remember(self, _text):
        pass

    def forget(self):
        pass


class FakeTranslator:
    def __init__(self):
        self.calls = []

    def translate(self, japanese, history):
        self.calls.append((japanese, list(history)))
        return "Watch out! Run!"


def _asr_utterance(asr, *, confidence=0.94, duration=6.0, probe=False,
                   get_generation=None):
    inputs, outputs = queue.Queue(), queue.Queue()
    inputs.put(Utterance(
        start_s=100.0, end_s=100.0 + duration,
        audio=np.zeros(round(duration * 16000), dtype=np.float32),
        vad_confidence=confidence, probe=probe, generation=0,
    ))
    inputs.put(None)
    run_asr(asr, inputs, outputs, get_generation=get_generation)
    return [item for item in list(outputs.queue) if isinstance(item, Transcript)]


def test_pekora_japanese_survives_english_gta_game_dialogue():
    """English stays in context; recovered Japanese alone becomes a subtitle."""
    asr = FakeMixedASR()
    transcripts = _asr_utterance(asr)
    assert [tr.language for tr in transcripts] == ["en", "ja"]
    assert transcripts[1].ja == "危ない、逃げて！"
    assert transcripts[1].start_s == 102.0
    assert transcripts[1].end_s == 104.5
    assert asr.recovery_calls == 1

    input_q, output_q = queue.Queue(), queue.Queue()
    for transcript in transcripts:
        input_q.put(transcript)
    input_q.put(None)
    translator = FakeTranslator()
    run_translate(translator, 6, input_q, output_q)

    subtitles = [item for item in list(output_q.queue) if isinstance(item, Subtitle)]
    assert len(subtitles) == 1
    assert subtitles[0].en == "Watch out! Run!"
    assert subtitles[0].ja == "危ない、逃げて！"
    assert translator.calls[0][0] == "危ない、逃げて！"
    assert any("Get in the car" in previous for previous, _ in translator.calls[0][1])


def test_english_background_only_still_yields_no_subtitles():
    asr = FakeMixedASR()
    asr.recover_value = []
    transcripts = _asr_utterance(asr)
    assert [tr.language for tr in transcripts] == ["en"]
    input_q, output_q = queue.Queue(), queue.Queue()
    for tr in transcripts:
        input_q.put(tr)
    input_q.put(None)
    translator = FakeTranslator()
    run_translate(translator, 6, input_q, output_q)
    assert not [x for x in list(output_q.queue) if isinstance(x, Subtitle)]
    assert translator.calls == []


def test_weak_game_audio_does_not_start_japanese_recovery():
    asr = FakeMixedASR()
    _asr_utterance(asr, confidence=0.25)
    assert asr.recovery_calls == 0


def test_qwen_seek_during_second_pass_discards_japanese_recovery():
    asr = FakeMixedASR()
    current_generation = [0]

    def perform_recovery(audio, probe=False):
        current_generation[0] = 1
        return [(1.0, 2.0, "危ない")]

    asr.recover_japanese_from_mixed_audio = perform_recovery
    transcripts = _asr_utterance(
        asr, get_generation=lambda: current_generation[0],
    )
    assert not [tr for tr in transcripts if tr.language == "ja"]


def test_qwen_deadline_keeps_validated_partial_japanese_recovery(monkeypatch):
    import threading

    from auto_translate.asr import qwen3_asr as qwen
    from auto_translate.config import AsrConfig

    model = object.__new__(qwen.Qwen3ASR)
    model.cfg = AsrConfig(response_timeout_s=12.0)
    model._generation = 0
    model._closed = False
    model._state_lock = threading.Lock()
    model._inference_lock = threading.Lock()
    model._cancel_event = threading.Event()
    model.last_detected_language = "en"
    model.last_raw_transcript = "Get in the car, quick!"
    checks = 0

    def remaining(_deadline, _generation):
        nonlocal checks
        checks += 1
        if checks == 4:
            raise qwen._UtteranceDeadline("window recovery exhausted its deadline")
        return 10.0

    def decode(_audio, **_kwargs):
        model.last_detected_language = "ja"
        return "危ない、逃げて！"

    monkeypatch.setattr(model, "_remaining", remaining)
    monkeypatch.setattr(model, "_decode", decode)

    recovered = model.recover_japanese_from_mixed_audio(
        np.zeros(6 * 16000, dtype=np.float32),
    )

    assert recovered == [(0.0, 2.25, "危ない、逃げて！")]
    assert model.last_detected_language == "en"
    assert model.last_raw_transcript == "Get in the car, quick!"
