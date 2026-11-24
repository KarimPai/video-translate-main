"""Regression tests for bounded noise-adaptive Silero VAD thresholds.

No real Silero model, GPU, network or subprocess is started by these tests.
"""
import numpy as np
import pytest

from auto_translate.config import VadConfig, vad_for_haas_game_audio
from auto_translate.vad import FRAME, Segmenter


class _SilentModel:
    def reset(self):
        pass

    def __call__(self, samples):
        return np.zeros(len(samples) // FRAME, dtype=np.float32)


@pytest.fixture
def segmenter(monkeypatch):
    from auto_translate import vad

    monkeypatch.setattr(vad, "StreamingSileroVAD", lambda *_args, **_kwargs: _SilentModel())

    def make(**settings):
        return Segmenter(VadConfig(fallback_s=0.0, **settings))

    return make


def _observe(seg, prob, seconds):
    frame = np.zeros(FRAME, dtype=np.float32)
    for _ in range(int(seconds * 16_000 / FRAME)):
        seg._step(frame, prob)


def test_fixed_mode_does_not_adjust_threshold(segmenter):
    seg = segmenter(dynamic_threshold=False)
    _observe(seg, 0.20, seconds=65)
    assert seg._speech_threshold == pytest.approx(0.24)
    assert seg._exit_threshold == pytest.approx(0.16)
    assert seg.cfg.threshold == pytest.approx(0.24)


def test_dynamic_mode_rises_on_sustained_subthreshold_background(segmenter):
    seg = segmenter(dynamic_threshold=True)
    _observe(seg, 0.20, seconds=35)
    assert 0.24 < seg._speech_threshold <= 0.36
    assert seg._exit_threshold == pytest.approx(seg._speech_threshold - 0.08)
    # Dynamic updates do not mutate the shared backend-agnostic config.
    assert seg.cfg.threshold == pytest.approx(0.24)
    assert seg.cfg.neg_threshold == pytest.approx(0.16)


def test_dynamic_mode_does_not_raise_gate_on_silence(segmenter):
    seg = segmenter(dynamic_threshold=True)
    _observe(seg, 0.0, seconds=35)
    assert seg._speech_threshold == pytest.approx(0.24)
    assert seg._exit_threshold == pytest.approx(0.16)


def test_dynamic_gate_never_exceeds_configured_cap(segmenter):
    seg = segmenter(dynamic_threshold=True, dynamic_max_increase=0.05)
    _observe(seg, 0.23, seconds=70)
    assert 0.24 < seg._speech_threshold <= 0.29 + 1e-6


def test_threshold_recovers_when_background_quiets(segmenter):
    seg = segmenter(dynamic_threshold=True)
    _observe(seg, 0.20, seconds=35)
    noisy = seg._speech_threshold
    assert noisy > seg.cfg.threshold
    _observe(seg, 0.0, seconds=75)
    assert seg._speech_threshold < noisy
    assert seg._speech_threshold >= seg.cfg.threshold


def test_dynamic_setting_loads_from_global_toml(tmp_path):
    from auto_translate.config import load_config

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[vad]\ndynamic_threshold = true\ndynamic_window_s = 25.0\n"
        "dynamic_update_s = 4.0\ndynamic_max_increase = 0.09\n",
        encoding="utf-8",
    )
    cfg = load_config(config_path)
    assert cfg.vad.dynamic_threshold is True
    assert cfg.vad.dynamic_window_s == pytest.approx(25.0)
    assert cfg.vad.dynamic_update_s == pytest.approx(4.0)
    assert cfg.vad.dynamic_max_increase == pytest.approx(0.09)


def test_active_utterance_freezes_dynamic_gate(segmenter):
    seg = segmenter(dynamic_threshold=True)
    _observe(seg, 0.20, seconds=30)
    learned = seg._speech_threshold
    _observe(seg, 0.95, seconds=2)
    assert seg._in_speech
    _observe(seg, 0.95, seconds=4)
    assert seg._speech_threshold == pytest.approx(learned)


def test_seek_resets_adaptation_and_observations(segmenter):
    seg = segmenter(dynamic_threshold=True)
    _observe(seg, 0.20, seconds=30)
    assert seg._speech_threshold > seg.cfg.threshold
    assert seg._dynamic_idle_probs
    seg._reset(next_sample=160_000, generation=1)
    assert seg._speech_threshold == seg.cfg.threshold
    assert seg._exit_threshold == seg.cfg.neg_threshold
    assert not seg._dynamic_idle_probs
    assert seg._dynamic_idle_since_update == 0


def test_haas_guard_preserves_dynamic_config_and_baseline():
    guarded = vad_for_haas_game_audio(VadConfig(dynamic_threshold=True, fallback_s=0.0))
    assert guarded.threshold == pytest.approx(0.28)
    assert guarded.dynamic_threshold is True
    assert guarded.neg_threshold == pytest.approx(0.16)


@pytest.mark.parametrize("changes", [
    {"dynamic_threshold": 1},
    {"dynamic_window_s": 0},
    {"dynamic_update_s": -1},
    {"dynamic_update_s": 31},
    {"dynamic_quantile": 0},
    {"dynamic_quantile": 1},
    {"dynamic_noise_margin": -0.1},
    {"dynamic_max_increase": 1.5},
])
def test_invalid_dynamic_vad_options_rejected(changes):
    with pytest.raises(ValueError):
        VadConfig(**changes)
