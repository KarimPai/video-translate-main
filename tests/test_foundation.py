"""Tests for foundation milestone — config, registry, types, and interfaces."""

import sys
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_types_import():
    """Types module imports and dataclasses are usable."""
    import numpy as np

    from auto_translate.types import (
        AudioChunk,
        SourceEOF,
        Subtitle,
        Timings,
        Transcript,
        Utterance,
    )

    timings = Timings()
    assert timings.audio_read == 0.0

    chunk = AudioChunk(start_sample=0, pcm=np.zeros(16000), read_at=0.0)
    assert chunk.start_sample == 0
    assert chunk.generation == 0

    utterance = Utterance(start_s=0.0, end_s=1.0, audio=np.zeros(16000))
    assert utterance.start_s == 0.0

    transcript = Transcript(start_s=0.0, end_s=1.0, ja="こんにちは")
    assert transcript.ja == "こんにちは"

    subtitle = Subtitle(start_s=0.0, end_s=1.0, ja="こんにちは", en="Hello")
    assert subtitle.en == "Hello"

    eof = SourceEOF()
    assert eof.generation == 0


def test_config_defaults():
    """Config loads with defaults when no config file exists."""
    from auto_translate.config import Config, load_config

    cfg = load_config(Path("/nonexistent/config.toml"))
    assert isinstance(cfg, Config)
    assert cfg.models.asr_backend == "haas"
    assert cfg.models.translate_backend == "gemma4"
    assert cfg.models.vad_backend == "silero"
    assert cfg.vad.device == "cpu"
    assert cfg.translate.context_lines == 6


def test_config_example():
    """Config loads from example file."""
    from auto_translate.config import load_config

    example = Path(__file__).parent.parent.parent / "config.example.toml"
    if example.exists():
        cfg = load_config(example)
        assert cfg.models.asr_backend == "haas"
        assert cfg.models.translate_backend == "gemma4"


def test_registry_creation():
    """ModelRegistry creates and lists backends."""
    from auto_translate.registry import ModelRegistry

    registry = ModelRegistry()
    backends = registry.list_backends()
    assert "asr" in backends
    assert "translate" in backends
    assert "vad" in backends


def test_registry_default():
    """Default registry registers Silero VAD."""
    from auto_translate.registry import get_default_registry

    registry = get_default_registry()
    vad_backends = registry.list_backends("vad")
    assert "silero" in vad_backends["vad"]

    caps = registry.get_capabilities("vad", "silero")
    assert caps is not None
    assert caps.runtime == "silero"
    assert caps.device == "cpu"


def test_registry_unknown_backend():
    """Creating an unknown backend raises ValueError."""
    from auto_translate.registry import ModelRegistry

    registry = ModelRegistry()
    try:
        registry.create_asr("nonexistent", {})
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Unknown ASR backend" in str(e)

    try:
        registry.create_translate("nonexistent", {})
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Unknown translation backend" in str(e)


def test_asr_interface():
    """ASRBackend protocol is defined."""
    from auto_translate.asr.base import ASRBackend
    assert hasattr(ASRBackend, "transcribe")
    assert hasattr(ASRBackend, "cancel_pending")
    assert hasattr(ASRBackend, "close")


def test_translate_interface():
    """TranslationBackend protocol is defined."""
    from auto_translate.translate.base import TranslationBackend, TranslatorUnavailable
    assert hasattr(TranslationBackend, "translate")
    assert hasattr(TranslationBackend, "close")
    assert hasattr(TranslationBackend, "interrupt")
    assert issubclass(TranslatorUnavailable, RuntimeError)


def test_vad_interface():
    """VADBackend protocol is defined."""
    from auto_translate.vad.base import VADBackend
    assert hasattr(VADBackend, "feed")
    assert hasattr(VADBackend, "flush")
    assert hasattr(VADBackend, "reset")
    assert hasattr(VADBackend, "close")


def test_subtitle_clean_import():
    """Subtitle cleaning module imports."""
    from auto_translate.subtitle_clean import clean_english_output
    assert callable(clean_english_output)


def test_glossary_import():
    """Glossary module imports."""
    from auto_translate.glossary import Glossary
    assert hasattr(Glossary, "load")


def test_session_cache_import():
    """Session cache module imports."""
    from auto_translate.session_cache import SessionSubtitleCache
    cache = SessionSubtitleCache()
    assert cache is not None


def test_backend_config_merge():
    """Backend config merges common and profile settings."""
    from auto_translate.config import Config, get_backend_config

    cfg = Config()
    cfg.translate_profiles["test"] = {"temperature": 0.5, "max_tokens": 256}
    merged = get_backend_config(cfg, "translate", "test")
    assert merged["temperature"] == 0.5
    assert merged["max_tokens"] == 256
    # Common values preserved
    assert merged["context_lines"] == 6


if __name__ == "__main__":
    test_types_import()
    test_config_defaults()
    test_config_example()
    test_registry_creation()
    test_registry_default()
    test_registry_unknown_backend()
    test_asr_interface()
    test_translate_interface()
    test_vad_interface()
    test_subtitle_clean_import()
    test_glossary_import()
    test_session_cache_import()
    test_backend_config_merge()
    print("All foundation tests passed!")
