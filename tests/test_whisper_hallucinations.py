"""No-model regressions for Whisper hallucinations and EN/JA gameplay routing.

These tests use synthetic decoder metadata. Real-audio accuracy must also be
benchmarked; synthetic tests cannot establish speech-vs-noise discrimination.
"""
from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from unittest import mock
import queue
import threading

import numpy as np

from auto_translate.asr.whisper_turbo import WhisperASR
from auto_translate.config import AsrConfig
from auto_translate.pipeline import run_asr, run_translate
from auto_translate.types import Transcript, Subtitle, Utterance


def _seg(text, start=0.0, end=3.0, *, no_speech=0.02,
         logprob=-0.2, compression=1.1):
    return SimpleNamespace(
        text=text, start=start, end=end,
        no_speech_prob=no_speech, avg_logprob=logprob,
        compression_ratio=compression,
    )


def _output(text, lang="ja", probability=0.96, **kwargs):
    return (
        iter([_seg(text, **kwargs)]),
        SimpleNamespace(language=lang, language_probability=probability),
    )


def _whisper():
    """Construct a model-free Whisper instance with true backend state."""
    w = object.__new__(WhisperASR)
    w.cfg = AsrConfig(response_timeout_s=12.0, recent_context_lines=3)
    w.context = ""
    w._name_hint_override = False
    w._recent = deque(maxlen=3)
    w._accepted_keys = deque(maxlen=32)
    w._state_lock = threading.Lock()
    w._inference_lock = threading.Lock()
    w._generation = 0
    w._closed = False
    w._failures = 0
    w._last_decode_prompt = ""
    w.last_match_score = 0.0
    w.last_raw_transcript = ""
    w.last_rejection = None
    w.last_language_spans = []
    w.last_detected_language = "und"
    w.last_detected_language_probability = 0.0
    w._model = mock.Mock()
    return w


def _audio(duration=3.0, amplitude=0.07):
    return np.full(int(duration * 16000), amplitude, dtype=np.float32)


def test_whisper_true_digital_silence_bypasses_inference():
    w = _whisper()
    assert w._transcribe_impl(np.zeros(16000 * 3, dtype=np.float32)) == ""
    assert w.last_detected_language == "none"
    w._model.transcribe.assert_not_called()


def test_whisper_quiet_japanese_survives_energy_guard():
    w = _whisper()
    w._model.transcribe.return_value = _output("助けて！", end=3.0)
    # Far below the old 0.003 absolute-RMS early-exit threshold.
    text = w._transcribe_impl(_audio(amplitude=0.0004))
    assert text == "助けて！"
    assert w.last_detected_language == "ja"
    w._model.transcribe.assert_called_once()


def test_whisper_low_confidence_noise_hypothesis_is_suppressed():
    w = _whisper()
    w._model.transcribe.return_value = _output(
        "はい", no_speech=0.98, logprob=-0.6,
    )
    assert w.transcribe(_audio()) == ""
    assert w.last_language_spans == []


def test_whisper_genuine_short_reaction_is_not_blocklisted():
    w = _whisper()
    w._model.transcribe.return_value = _output(
        "はい！", no_speech=0.14, logprob=-0.25,
    )
    assert w.transcribe(_audio(0.8)) == "はい！"


def test_whisper_phantom_outro_is_suppressed():
    w = _whisper()
    w._model.transcribe.return_value = _output("ご視聴ありがとうございました")
    assert w.transcribe(_audio()) == ""
    assert w.last_rejection == "phantom_outro"


def test_whisper_english_speech_stays_foreign():
    w = _whisper()
    w._model.transcribe.return_value = _output("Get in the car!", lang="en")
    assert w.transcribe(_audio()) == "Get in the car!"
    assert w.last_detected_language == "en"
    assert w.last_language_spans[0][2] == "en"


def test_whisper_nonenglish_latin_script_not_mislabeled_english():
    w = _whisper()
    w._model.transcribe.return_value = _output("¡Hola amigo!", lang="es")
    w.transcribe(_audio())
    assert w.last_detected_language == "es"


def test_whisper_japanese_nicknames_with_latin_letters_stay_japanese():
    for text in ("Jasonが来た", "Mikoちゃん", "GTA6をやろう"):
        assert WhisperASR._detect_language(text) == "ja"
        assert WhisperASR._split_text_languages(text) == [("ja", text)]


def test_whisper_isolates_yes_yeah_from_adjacent_japanese():
    """Common English NPC interjections must not leak into JP subtitles."""
    assert WhisperASR._split_text_languages("Yeah やばい！") == [
        ("en", "Yeah"), ("ja", "やばい！"),
    ]
    assert WhisperASR._split_text_languages("危ない Yeah") == [
        ("ja", "危ない"), ("en", "Yeah"),
    ]
    assert WhisperASR._split_text_languages("危ない Get in the car") == [
        ("ja", "危ない"), ("en", "Get in the car"),
    ]
    # Spoken names remain part of the Japanese phrase.
    assert WhisperASR._split_text_languages("Miko ちゃん") == [
        ("ja", "Miko ちゃん"),
    ]


def test_whisper_explicit_mixed_clauses_have_separate_languages():
    w = _whisper()
    w._model.transcribe.return_value = _output(
        "Get in the car! 危ない、逃げて！", lang="en", end=4.0,
    )
    assert w.transcribe(_audio(4.0)) == "Get in the car! 危ない、逃げて！"
    assert w.last_detected_language == "mixed"
    assert [p[2] for p in w.last_language_spans] == ["en", "ja"]
    assert [p[3] for p in w.last_language_spans] == [
        "Get in the car!", "危ない、逃げて！",
    ]


class _MixedWhisper:
    def __init__(self):
        self.cfg = AsrConfig(response_timeout_s=12.0)
        self.last_detected_language = "mixed"
        self.last_rejection = None
        self.last_raw_transcript = ""
        self.last_language_spans = [
            (0.0, 2.0, "en", "Get in the car!"),
            (2.0, 4.0, "ja", "危ない、逃げて！"),
        ]

    def transcribe(self, audio, probe=False):
        return "Get in the car! 危ない、逃げて！"

    def remember(self, text):
        pass


class _Translator:
    def __init__(self):
        self.calls = []

    def translate(self, japanese, history):
        self.calls.append((japanese, list(history)))
        return "Watch out! Run!"


def test_whisper_mixed_pipeline_english_context_not_translated():
    asr_in, asr_out = queue.Queue(), queue.Queue()
    asr_in.put(Utterance(
        start_s=100.0, end_s=104.0, audio=_audio(4.0),
        vad_confidence=0.95, generation=0,
    ))
    asr_in.put(None)
    run_asr(_MixedWhisper(), asr_in, asr_out)
    transcripts = [x for x in list(asr_out.queue) if isinstance(x, Transcript)]
    assert [x.language for x in transcripts] == ["en", "ja"]
    assert [x.ja for x in transcripts] == ["Get in the car!", "危ない、逃げて！"]
    assert transcripts[1].start_s == 102.0

    trans_in, trans_out = queue.Queue(), queue.Queue()
    for tr in transcripts:
        trans_in.put(tr)
    trans_in.put(None)
    mt = _Translator()
    run_translate(mt, 6, trans_in, trans_out)
    subtitles = [x for x in list(trans_out.queue) if isinstance(x, Subtitle)]
    assert len(subtitles) == 1
    assert subtitles[0].ja == "危ない、逃げて！"
    assert [x[0] for x in mt.calls] == ["危ない、逃げて！"]
    assert any("Get in the car" in history_line[0]
               for history_line in mt.calls[0][1])


def test_whisper_recovery_uses_auto_language_not_forced_japanese():
    w = _whisper()
    w._model.transcribe.side_effect = [
        _output("Get in the car", lang="en", end=2.2),
        _output("危ない、逃げて！", lang="ja", probability=0.95, end=2.2),
        _output("Let's go", lang="en", end=2.2),
    ]
    candidates = w.recover_japanese_from_mixed_audio(_audio(6.0))
    assert len(candidates) == 1
    assert candidates[0][2] == "危ない、逃げて！"
    assert w._model.transcribe.call_count == 3
    for call in w._model.transcribe.call_args_list:
        assert call.kwargs["language"] is None
        assert call.kwargs["initial_prompt"] is None
        assert call.kwargs["condition_on_previous_text"] is False


def test_whisper_recovery_rejects_japanese_text_with_english_audio_label():
    w = _whisper()
    w._model.transcribe.return_value = _output(
        "危ない！", lang="en", probability=0.99,
    )
    assert w.recover_japanese_from_mixed_audio(_audio(6.0)) == []


def test_whisper_recovery_rejects_weak_language_probability():
    w = _whisper()
    w._model.transcribe.return_value = _output(
        "危ない！", lang="ja", probability=0.45,
    )
    assert w.recover_japanese_from_mixed_audio(_audio(6.0)) == []


def test_whisper_recovery_skips_probes_and_short_audio():
    w = _whisper()
    assert w.recover_japanese_from_mixed_audio(_audio(1.0)) == []
    assert w.recover_japanese_from_mixed_audio(_audio(6.0), probe=True) == []
    w._model.transcribe.assert_not_called()


def test_whisper_recovery_discards_generation_changed_during_decode():
    w = _whisper()

    def _decode(*args, **kwargs):
        w._generation += 1
        return _output("危ない", lang="ja", probability=0.99)

    w._model.transcribe.side_effect = _decode
    assert w.recover_japanese_from_mixed_audio(_audio(6.0)) == []


def test_whisper_suspected_long_prompt_echo_retried_without_context():
    w = _whisper()
    repeated = "今日は新しいゲームを始めましょうそろそろ出発しますね"
    assert len(repeated) >= 24
    w.context = repeated
    w._model.transcribe.side_effect = [
        _output(repeated, lang="ja", end=5.0),
        _output("次はどこに行こうかな", lang="ja", end=5.0),
    ]
    assert w.transcribe(_audio(5.0)) == ""
    assert w.last_rejection == "context_echo"
    assert w.last_match_score >= 0.97
    recovered = w.recover_context_echo(_audio(5.0), timeout=8.0)
    assert recovered == "次はどこに行こうかな"
    calls = w._model.transcribe.call_args_list
    assert calls[0].kwargs["initial_prompt"] == repeated
    assert calls[1].kwargs["initial_prompt"] is None


def test_whisper_forget_clears_previous_dialogue_on_seek():
    w = _whisper()
    w._recent.extend(["前の会話", "次の会話"])
    w._accepted_keys.append("old")
    w.forget()
    assert not w._recent
    assert not w._accepted_keys
