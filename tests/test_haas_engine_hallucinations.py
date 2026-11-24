"""Haas-specific engine-rev false-positive regressions.

These tests exercise real Haas decoder and pipeline methods using a mocked
speech.cpp server. They do not claim to classify an engine waveform without
real audio / labels.
"""
from __future__ import annotations

import queue
import threading
from collections import deque
from unittest import mock

import numpy as np

from auto_translate.asr.haas import (
    HaasASR,
    is_english_engine_interjection,
    is_katakana_engine_interjection,
)
from auto_translate.config import AsrConfig, VadConfig, vad_for_haas_game_audio
from auto_translate.pipeline import run_asr
from auto_translate.types import Transcript, Utterance


def _bare_haas(*outputs):
    asr = object.__new__(HaasASR)
    asr.cfg = AsrConfig(response_timeout_s=10.0, recent_context_lines=3)
    asr.context = ""
    asr._recent = deque(maxlen=3)
    asr._accepted_keys = deque(maxlen=32)
    asr._recent_engine_interjections = deque(maxlen=8)
    asr.last_match_score = 0.0
    asr.last_raw_transcript = ""
    asr.last_rejection = None
    asr.last_timing = None
    asr._state_lock = threading.Lock()
    asr._inference_lock = threading.Lock()
    asr._generation = 0
    asr._closed = False
    asr._failures = 0
    asr._restarts = 0
    asr.server = mock.Mock()
    asr.server.alive.return_value = True
    asr.server.transcribe_with_timing.side_effect = [
        (out, {"http_request_s": 0.01, "total_s": 0.02})
        for out in outputs
    ]
    return asr


def _utterance(start, duration=1.0, *, probe=False, confidence=0.9):
    return Utterance(
        start_s=start, end_s=start + duration,
        audio=np.full(round(duration * 16000), .025, dtype=np.float32),
        probe=probe, vad_confidence=confidence, generation=0,
    )


def test_haas_standalone_latin_yeah_variants_are_noise_candidates():
    for transcript in ("yeah", "Yeah!", "YEAAAH!!", "yeah yeah", "yea", "yah!"):
        assert is_english_engine_interjection(transcript), transcript
    for transcript in (
        "すごい、yeah！", "イェー！みんなありがとう！", "yeah、やった！",
        "Yae Miko", "Yes", "やばい",
    ):
        assert not is_english_engine_interjection(transcript), transcript


def test_haas_katakana_reaction_only_classification():
    for transcript in ("イェー！", "イエー", "イェーイ", "イェイ"):
        assert is_katakana_engine_interjection(transcript), transcript
    for transcript in ("イェー！みんなありがとう！", "すごいイェー！", "イエロー", "やった！"):
        assert not is_katakana_engine_interjection(transcript), transcript


def test_haas_decoder_drops_english_yeah_even_when_model_returns_it():
    for text in ("Yeah!", "yeah yeah!", "Yeaaah"):
        asr = _bare_haas(text)
        with mock.patch("auto_translate.asr.haas.check_running"):
            assert asr.transcribe(_utterance(0).audio) == ""
        assert asr.last_rejection == "haas_non_japanese_engine_interjection"
        assert asr.last_raw_transcript == text
        asr.server.transcribe_with_timing.assert_called_once()


def test_haas_decoder_keeps_real_japanese_sentence_and_names():
    for text in ("やった！イェーイ！", "みこちゃん、車が来た！", "はい！"):
        asr = _bare_haas(text)
        with mock.patch("auto_translate.asr.haas.check_running"):
            assert asr.transcribe(_utterance(0).audio) == text
        assert asr.last_rejection is None


def test_haas_katakana_false_yeah_requires_vad_evidence():
    asr = _bare_haas()
    assert asr.reject_utterance("イェー!", _utterance(10, probe=True)) == "haas_weak_katakana_interjection"
    assert asr.reject_utterance("イェー!", _utterance(12, confidence=0.34)) == "haas_weak_katakana_interjection"
    assert asr.reject_utterance("イェー!", _utterance(13, confidence=0.95)) is None


def test_haas_repeated_katakana_engine_revs_do_not_flood_subtitles():
    asr = _bare_haas()
    assert asr.reject_utterance("イェー！", _utterance(100, confidence=0.90)) is None
    assert asr.reject_utterance("イェー！", _utterance(102, confidence=0.95)) == "haas_repeated_engine_interjection"
    assert asr.reject_utterance("イェーイ！", _utterance(103, confidence=0.90)) == "haas_repeated_engine_interjection"
    # Real speech elsewhere in this interval is unaffected.
    assert asr.reject_utterance("すごい！やった！", _utterance(104, confidence=0.90)) is None
    assert asr.reject_utterance("イェー！", _utterance(120, confidence=0.95)) is None


def test_haas_seek_and_silence_gap_clear_interjection_history():
    asr = _bare_haas()
    assert asr.reject_utterance("イェー！", _utterance(90)) is None
    asr.cancel_pending()
    assert asr.reject_utterance("イェー！", _utterance(1)) is None
    asr.forget()
    assert asr.reject_utterance("イェー！", _utterance(2)) is None


def test_haas_pipeline_discards_engine_rev_yeah_but_keeps_japanese():
    asr = _bare_haas("Yeah!", "危ない！")
    incoming, outgoing = queue.Queue(), queue.Queue()
    incoming.put(_utterance(0, confidence=.90))
    incoming.put(_utterance(2, confidence=.90))
    incoming.put(None)
    with mock.patch("auto_translate.asr.haas.check_running"):
        run_asr(asr, incoming, outgoing)
    published = [x for x in list(outgoing.queue) if isinstance(x, Transcript)]
    assert [x.ja for x in published] == ["危ない！"]
    assert asr.server.transcribe_with_timing.call_count == 2


def test_haas_pipeline_filters_vad_negative_katakana_probe():
    asr = _bare_haas("イェー！", "危ない！")
    incoming, outgoing = queue.Queue(), queue.Queue()
    incoming.put(_utterance(10, probe=True, confidence=None))
    incoming.put(_utterance(12, probe=False, confidence=.95))
    incoming.put(None)
    with mock.patch("auto_translate.asr.haas.check_running"):
        run_asr(asr, incoming, outgoing)
    published = [x for x in list(outgoing.queue) if isinstance(x, Transcript)]
    assert [x.ja for x in published] == ["危ない！"]


def test_haas_pipeline_retains_one_valid_high_confidence_katakana_cheer():
    asr = _bare_haas("イェー！", "イェー！")
    incoming, outgoing = queue.Queue(), queue.Queue()
    incoming.put(_utterance(10, confidence=.94))
    incoming.put(_utterance(12, confidence=.94))
    incoming.put(None)
    with mock.patch("auto_translate.asr.haas.check_running"):
        run_asr(asr, incoming, outgoing)
    published = [x for x in list(outgoing.queue) if isinstance(x, Transcript)]
    assert [x.ja for x in published] == ["イェー！"]



def test_haas_new_engine_rev_interjections_ugh_yeah_phew():
    for text in (
        "ugh yeah", "Ugh, Yeah!", "UGH YEAH!!", "ugh... yeaaah",
        "Phew", "phew!", "PHEEEW...", "whew", "Wheeeew!",
        "ugh", "uh", "Oof!", "ugh yeah phew", "phew, phew",
    ):
        assert is_english_engine_interjection(text), text


def test_haas_engine_rev_words_inside_real_dialogue_are_not_blocked():
    for text in (
        "ああ、phew！助かった！",
        "Ugh, yeah, that car is fast",
        "yeah、ちょっと待って！",
        "みこちゃん、車が来た！",
        "Miko",
        "Phewter",  # don't classify names merely containing similar phonemes
    ):
        assert not is_english_engine_interjection(text), text


def test_haas_decoder_drops_ugh_yeah_and_phew_without_extra_inference():
    for text in ("Ugh, Yeah!", "Phew!", "wheeew...", "Ugh! Phew!"):
        asr = _bare_haas(text)
        with mock.patch("auto_translate.asr.haas.check_running"):
            assert asr.transcribe(_utterance(5).audio) == ""
        assert asr.last_rejection == "haas_non_japanese_engine_interjection"
        assert asr.last_raw_transcript == text
        assert asr.server.transcribe_with_timing.call_count == 1


def test_haas_pipeline_drops_new_engine_words_but_preserves_japanese():
    asr = _bare_haas("Ugh, yeah!", "Phew!", "ちょっと待って！")
    inbound, outbound = queue.Queue(), queue.Queue()
    for start in (0, 2, 4):
        inbound.put(_utterance(start, confidence=.88))
    inbound.put(None)
    with mock.patch("auto_translate.asr.haas.check_running"):
        run_asr(asr, inbound, outbound)
    assert [tr.ja for tr in list(outbound.queue) if isinstance(tr, Transcript)] == [
        "ちょっと待って！"
    ]
    assert asr.server.transcribe_with_timing.call_count == 3


def test_haas_vad_profile_reduces_sensitivity_without_mutating_config():
    original = VadConfig(threshold=.24, neg_threshold=.16, fallback_s=12.0)
    guarded = vad_for_haas_game_audio(original)
    assert guarded is not original
    assert guarded.threshold == .28
    assert guarded.neg_threshold == original.neg_threshold
    assert guarded.fallback_s == 0.0
    assert guarded.short_reply_ms == original.short_reply_ms
    assert guarded.short_reply_probability == original.short_reply_probability
    assert original.threshold == .24
    assert original.fallback_s == 12.0


def test_haas_vad_profile_honors_custom_threshold():
    for threshold in (.31, .35, .20):
        cfg = VadConfig(threshold=threshold, neg_threshold=.16, fallback_s=5.0)
        guarded = vad_for_haas_game_audio(cfg)
        assert guarded.threshold == threshold
        assert guarded.fallback_s == 0.0


def test_haas_rare_short_japanese_reply_survives_sensitivity_change():
    cfg = vad_for_haas_game_audio(VadConfig())
    # The separate high-confidence short-reply gate stays enabled to catch
    # genuine soft short reactions even when the main threshold is raised.
    assert cfg.short_reply_ms == 80
    assert cfg.short_reply_probability == .85
    asr = _bare_haas("はい！")
    with mock.patch("auto_translate.asr.haas.check_running"):
        assert asr.transcribe(_utterance(10, .5, confidence=.95).audio) == "はい！"
