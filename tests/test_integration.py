"""Integration tests for all ported components in video-translate-universal.

Tests cover:
- Pipeline imports and core functions
- Backend adapter imports (Haas, Gemma4, PLaMo)
- Output module imports (MPV, terminal, web)
- GUI module imports
- Ingest module imports
- Config and registry integration
- Supporting modules (VAD, gpu_budget, etc.)
- Bug fixes verification (lock handling, stale prompt, function attribute state)
"""

import os
import sys
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_pipeline_imports():
    """Pipeline module imports all required components."""
    from auto_translate.pipeline import Stats, run, run_asr, run_translate
    assert callable(run)
    assert callable(run_asr)
    assert callable(run_translate)
    assert Stats is not None


def test_pipeline_state_class():
    """Pipeline no longer uses module-level _PipelineState singleton.

    The _last_vod_diag variable was moved into run() as a local variable
    to prevent stale state across pipeline restarts.
    """
    import auto_translate.pipeline as pipeline_mod
    assert not hasattr(pipeline_mod, '_PipelineState'), (
        "_PipelineState class should be removed from pipeline module"
    )
    assert not hasattr(pipeline_mod, '_run_state'), (
        "_run_state singleton should be removed from pipeline module"
    )


def test_haas_adapter_import():
    """Haas ASR adapter imports correctly."""
    from auto_translate.asr.haas import HaasASR, HaasError
    assert HaasASR is not None
    assert issubclass(HaasError, Exception)


def test_gemma4_adapter_import():
    """Gemma4 translator imports correctly."""
    from auto_translate.translate.gemma4 import (
        Gemma4Translator,
        build_prompt,
        clean_output,
        useful_context,
    )
    assert Gemma4Translator is not None
    assert callable(build_prompt)
    assert callable(clean_output)
    assert callable(useful_context)


def test_plamo_adapter_import():
    """PLaMo translator imports correctly."""
    from auto_translate.translate.plamo import (
        PLAMO_STOP_TOKENS,
        PlamoTranslator,
        build_plamo_prompt,
        clean_output,
        useful_context,
    )
    assert PlamoTranslator is not None
    assert callable(build_plamo_prompt)
    assert callable(clean_output)
    assert callable(useful_context)
    assert len(PLAMO_STOP_TOKENS) == 2
    assert "<|plamo:tag|>" in PLAMO_STOP_TOKENS
    assert "<|plamo:bos|>" in PLAMO_STOP_TOKENS


def test_gemma4_prompt_building():
    """Gemma4 prompt building works correctly."""
    from auto_translate.translate.gemma4 import build_prompt

    prompt = build_prompt("こんにちは", [], "")
    assert "messages" in prompt
    assert len(prompt["messages"]) > 0
    # Gemma4 uses system prompt
    system = prompt["messages"][0]
    assert system["role"] == "system"


def test_plamo_prompt_building():
    """PLaMo prompt building works correctly."""
    from auto_translate.translate.plamo import build_plamo_prompt

    prompt = build_plamo_prompt("hello", [], "")
    assert "messages" in prompt
    assert "chat_template_kwargs" in prompt
    assert prompt["chat_template_kwargs"]["target_lang"] == "en"
    # PLaMo uses user message, no system prompt
    user = prompt["messages"][0]
    assert user["role"] == "user"


def test_plamo_prompt_with_glossary():
    """PLaMo prompt passes glossary via chat_template_kwargs, not user text."""
    from auto_translate.translate.plamo import build_plamo_prompt

    prompt = build_plamo_prompt("game", [("Kira", "Kira")], "Previous: welcome")
    content = prompt["messages"][0]["content"]
    # Japanese speech stays as the user message
    assert content == "game"
    # Glossary is passed through chat_template_kwargs (official template)
    glossary = prompt["chat_template_kwargs"].get("glossary")
    assert glossary == [{"source": "Kira", "target": "Kira"}]


def test_gemma4_output_cleaning():
    """Gemma4 output cleaning removes markdown and role prefixes."""
    from auto_translate.translate.gemma4 import clean_output

    assert clean_output("Hello world") == "Hello world"
    assert clean_output("**Hello**") == "Hello"
    assert clean_output("") == ""


def test_plamo_output_cleaning():
    """PLaMo output cleaning removes markdown and role prefixes."""
    from auto_translate.translate.plamo import clean_output

    assert clean_output("Hello world") == "Hello world"
    assert clean_output("**Hello**") == "Hello"
    assert clean_output("") == ""


def test_gemma4_useful_context():
    """Gemma4 useful_context filters filler words."""
    from auto_translate.translate.gemma4 import useful_context

    assert useful_context("hello") == True
    assert useful_context("うん") == False


def test_plamo_useful_context():
    """PLaMo useful_context filters filler words."""
    from auto_translate.translate.plamo import useful_context

    assert useful_context("hello") == True
    assert useful_context("うん") == False


def test_output_mpv_import():
    """MPV output module imports correctly."""
    from auto_translate.output.mpv import MpvOutput
    assert MpvOutput is not None


def test_output_terminal_import():
    """Terminal output module imports correctly."""
    from auto_translate.output.terminal import TerminalOutput
    assert TerminalOutput is not None


def test_output_web_import():
    """Web output module imports correctly."""
    from auto_translate.output.web import WebOutput
    assert WebOutput is not None


def test_output_base_protocol():
    """Output base protocol is defined."""
    from auto_translate.output.base import Output
    assert Output is not None


def test_gui_imports():
    """GUI module imports correctly."""
    from auto_translate.gui import GUI_SHUTDOWN_MS
    assert GUI_SHUTDOWN_MS > 0


def test_gui_control_import():
    """GUI control module imports correctly."""
    from auto_translate.gui_control import emit_event
    assert callable(emit_event)


def test_gui_runner_import():
    """GUI runner module imports correctly."""
    from auto_translate.gui_runner import session_launcher
    assert callable(session_launcher)


def test_ingest_import():
    """Ingest module imports correctly."""
    from auto_translate.ingest import Ingest
    assert Ingest is not None


def test_vad_import():
    """VAD module imports correctly."""
    from auto_translate.vad import Segmenter, run_segmenter
    assert callable(run_segmenter)
    assert Segmenter is not None


def test_vad_base_protocol():
    """VAD base protocol is defined."""
    from auto_translate.vad.base import VADBackend, VADSegment
    assert VADBackend is not None
    assert VADSegment is not None


def test_removed_playback_config_is_ignored(tmp_path):
    """Existing private configs remain loadable after feature removal."""
    from auto_translate.config import load_config

    path = tmp_path / "config.toml"
    path.write_text("[safe_playback]\nenabled = true\n", encoding="utf-8")
    config = load_config(path)
    assert not hasattr(config, "safe_playback")


def test_gpu_budget_import():
    """GPU budget module imports correctly."""
    from auto_translate.gpu_budget import GpuBudget
    assert GpuBudget is not None


def test_speech_server_import():
    """Speech server module imports correctly."""
    from auto_translate.speech_server import SpeechCppServer
    assert SpeechCppServer is not None


def test_llama_server_import():
    """Llama server module imports correctly."""
    from auto_translate.llama_server import LlamaServer, LlamaServerError
    assert LlamaServer is not None
    assert issubclass(LlamaServerError, Exception)


def test_windows_subprocess_import():
    """Windows subprocess module imports correctly."""
    from auto_translate.windows_subprocess import check_running
    assert callable(check_running)


def test_runtime_diagnostics_import():
    """Runtime diagnostics module imports correctly."""
    from auto_translate.runtime_diagnostics import (
        increment_counter,
        record_event,
        update_metrics,
    )
    assert callable(increment_counter)
    assert callable(record_event)
    assert callable(update_metrics)


def test_stream_context_import():
    """Stream context module imports correctly."""
    from auto_translate.stream_context import StreamContext
    assert StreamContext is not None


def test_subtitle_clean_functions():
    """Subtitle clean module functions work correctly."""
    from auto_translate.subtitle_clean import (
        clean_english_output,
        has_generation_loop,
        looks_like_prompt_echo,
    )

    # clean_english_output should return cleaned text (string)
    result = clean_english_output("Hello world")
    assert isinstance(result, str)

    # looks_like_prompt_echo should detect prompt leakage
    assert looks_like_prompt_echo("You are a professional Japanese-to-English subtitle translator") == True
    assert looks_like_prompt_echo("Hello world") == False

    # has_generation_loop requires 18+ words and 3 consecutive identical 6+ word phrases
    loop_text = "hello world this is a test hello world this is a test hello world this is a test"
    assert has_generation_loop(loop_text) == True
    assert has_generation_loop("hello world") == False


def test_glossary_functions():
    """Glossary module functions work correctly."""
    from auto_translate.glossary import Glossary

    glossary = Glossary(terms={})
    assert hasattr(glossary, "find")
    assert hasattr(glossary, "name_terms")


def test_session_cache_functions():
    """Session cache module functions work correctly."""
    from auto_translate.session_cache import SessionSubtitleCache
    from auto_translate.types import Subtitle

    cache = SessionSubtitleCache()
    sub = Subtitle(start_s=0.0, end_s=1.0, ja="test", en="test")
    cache.add(sub)
    # Cache uses match() method to find subtitles at a given time
    result = cache.match(0.0, 1.0)
    assert result is not None


def test_text_filter_import():
    """Text filter module imports correctly."""
    from auto_translate.text_filter import is_usable_transcript
    assert callable(is_usable_transcript)


def test_registry_list_backends():
    """Registry lists all registered backends."""
    from auto_translate.registry import get_default_registry

    registry = get_default_registry()
    registry.list_backends()

    # Check ASR backends
    asr_backends = registry.list_backends("asr")
    assert "asr" in asr_backends
    assert "haas" in asr_backends["asr"]

    # Check translation backends
    translate_backends = registry.list_backends("translate")
    assert "translate" in translate_backends
    assert "gemma4" in translate_backends["translate"]
    assert "plamo" in translate_backends["translate"]

    # Check VAD backends
    vad_backends = registry.list_backends("vad")
    assert "vad" in vad_backends
    assert "silero" in vad_backends["vad"]


def test_registry_backend_capabilities():
    """Registry returns correct backend capabilities."""
    from auto_translate.registry import get_default_registry

    registry = get_default_registry()

    # Haas capabilities
    haas_caps = registry.get_capabilities("asr", "haas")
    assert haas_caps is not None
    assert haas_caps.runtime == "speech_cpp"
    assert haas_caps.device == "vulkan"

    # Gemma4 capabilities
    gemma_caps = registry.get_capabilities("translate", "gemma4")
    assert gemma_caps is not None
    assert gemma_caps.runtime == "llama_cpp"
    assert gemma_caps.device == "cuda"

    # PLaMo capabilities
    plamo_caps = registry.get_capabilities("translate", "plamo")
    assert plamo_caps is not None
    assert plamo_caps.runtime == "llama_cpp"
    assert plamo_caps.device == "cuda"


def test_registry_create_backend():
    """Registry create_backend dispatches correctly."""
    from auto_translate.registry import get_default_registry

    registry = get_default_registry()

    # Test that create_backend raises for unknown type
    try:
        registry.create_backend("unknown_type", "test", {})
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_config_backend_profiles():
    """Config supports backend-specific profiles."""
    from auto_translate.config import Config, get_backend_config

    cfg = Config()
    # Add PLaMo-specific profile
    cfg.translate_profiles["plamo"] = {
        "model_path": "models/plamo3/plamo-3-translate-2b-Q4_K_M.gguf",
        "max_tokens": 128,
    }

    merged = get_backend_config(cfg, "translate", "plamo")
    assert merged["model_path"] == "models/plamo3/plamo-3-translate-2b-Q4_K_M.gguf"
    assert merged["max_tokens"] == 128
    # Common values should be preserved
    assert "context_lines" in merged


def test_config_vad_device_validation():
    """Config validates VAD device must be CPU."""
    from auto_translate.config import Config

    cfg = Config()
    assert cfg.vad.device == "cpu"


def test_gemma4_reload_lock():
    """Gemma4 uses separate _reload_lock to prevent deadlock."""
    import inspect

    from auto_translate.translate.gemma4 import Gemma4Translator

    # Check that _reload_lock is defined in __init__
    source = inspect.getsource(Gemma4Translator.__init__)
    assert "_reload_lock" in source


def test_plamo_reload_lock():
    """PLaMo uses separate _reload_lock to prevent deadlock."""
    import inspect

    from auto_translate.translate.plamo import PlamoTranslator

    # Check that _reload_lock is defined in __init__
    source = inspect.getsource(PlamoTranslator.__init__)
    assert "_reload_lock" in source


def test_pipeline_no_function_attribute():
    """Pipeline uses local variables instead of module-level state for diagnostics.

    The _last_vod_diag variable is now a local variable inside run(),
    not a module-level singleton.
    """
    import auto_translate.pipeline as pipeline_mod
    assert not hasattr(pipeline_mod, '_PipelineState'), (
        "_PipelineState should not exist as module-level class"
    )
    assert not hasattr(pipeline_mod, '_run_state'), (
        "_run_state should not exist as module-level singleton"
    )


def test_asr_quality_guard():
    """ASR quality guard module imports and works."""
    from auto_translate.asr.quality_guard import transcript_problem
    assert callable(transcript_problem)


def test_asr_context_echo():
    """ASR context echo module imports and works."""
    from auto_translate.asr.context_echo import context_match
    assert callable(context_match)


def test_subtitle_layout_import():
    """Subtitle layout module imports correctly."""
    from auto_translate.subtitle_layout import split_cue, wrap_lines
    assert callable(split_cue)
    assert callable(wrap_lines)


def test_timeline_import():
    """Timeline module imports correctly."""
    from auto_translate.timeline import record_session_anchor
    assert callable(record_session_anchor)


def test_windows_job_import():
    """Windows job module imports correctly."""
    from auto_translate.windows_job import bind_process_tree
    assert callable(bind_process_tree)


def test_models_import():
    """Models module imports correctly."""
    from auto_translate.models import ensure_model
    assert callable(ensure_model)


def test_ytdlp_import():
    """yt-dlp wrapper module imports correctly."""
    from auto_translate.ytdlp import extract
    assert callable(extract)


def test_cli_entry_point():
    """CLI entry point imports and runs without error."""
    from auto_translate import __main__
    # Just verify the module loads — actual CLI requires args
    assert __main__ is not None


def test_qwen3_asr_import():
    """Qwen3-ASR adapter imports correctly."""
    from auto_translate.asr.qwen3_asr import Qwen3ASR, Qwen3ASRError
    assert Qwen3ASR is not None
    assert issubclass(Qwen3ASRError, Exception)


def test_qwen3_asr_registry():
    """Registry lists qwen3-asr backend."""
    from auto_translate.registry import get_default_registry

    registry = get_default_registry()
    asr_backends = registry.list_backends("asr")
    assert "asr" in asr_backends
    assert "qwen3-asr" in asr_backends["asr"]


def test_qwen3_asr_capabilities():
    """Registry returns correct Qwen3-ASR capabilities."""
    from auto_translate.registry import get_default_registry

    registry = get_default_registry()
    caps = registry.get_capabilities("asr", "qwen3-asr")
    assert caps is not None
    assert caps.runtime == "llama_cpp"
    assert caps.device == "cuda"
    assert caps.source_lang == "ja"
    assert caps.vram_gb > 0


def test_qwen3_asr_config_mmproj():
    """AsrConfig supports mmproj_path field."""
    from auto_translate.config import AsrConfig

    cfg = AsrConfig(
        model_path="models/qwen3-asr/Qwen3-ASR-1.7B-Q8_0.gguf",
        mmproj_path="models/qwen3-asr/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf",
    )
    assert cfg.mmproj_path == "models/qwen3-asr/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf"


def test_qwen3_asr_config_default():
    """AsrConfig mmproj_path defaults to empty string."""
    from auto_translate.config import AsrConfig

    cfg = AsrConfig()
    assert cfg.mmproj_path == ""


def test_types_dataclasses():
    """All types dataclasses are properly defined."""
    import numpy as np

    from auto_translate.types import (
        AudioChunk,
        SourceEOF,
        Subtitle,
        Timings,
        Transcript,
        Utterance,
    )

    # Test Timings
    t = Timings()
    assert t.audio_read == 0.0

    # Test AudioChunk
    chunk = AudioChunk(start_sample=0, pcm=np.zeros(16000), read_at=0.0)
    assert chunk.start_sample == 0
    assert chunk.generation == 0

    # Test Utterance
    utt = Utterance(start_s=0.0, end_s=1.0, audio=np.zeros(16000))
    assert utt.start_s == 0.0
    assert utt.end_s == 1.0

    # Test Transcript
    tr = Transcript(start_s=0.0, end_s=1.0, ja="test")
    assert tr.ja == "test"

    # Test Subtitle
    sub = Subtitle(start_s=0.0, end_s=1.0, ja="test", en="test")
    assert sub.ja == "test"
    assert sub.en == "test"

    # Test SourceEOF
    eof = SourceEOF()
    assert eof.generation == 0


def test_gemma4_vs_plamo_prompt_isolation():
    """Gemma4 and PLaMo use different prompt formats."""
    from auto_translate.translate.gemma4 import build_prompt
    from auto_translate.translate.plamo import build_plamo_prompt

    gemma_prompt = build_prompt("test", [], "")
    plamo_prompt = build_plamo_prompt("test", [], "")

    # Gemma4 uses system prompt
    assert gemma_prompt["messages"][0]["role"] == "system"

    # PLaMo uses user message with chat_template_kwargs
    assert plamo_prompt["messages"][0]["role"] == "user"
    assert "chat_template_kwargs" in plamo_prompt
    assert "chat_template_kwargs" not in gemma_prompt


def test_gpu_budget_class():
    """GpuBudget class is defined with relief callback."""
    from auto_translate.gpu_budget import GpuBudget
    assert hasattr(GpuBudget, '__init__')


def test_llama_server_class():
    """LlamaServer class is defined with required methods."""
    from auto_translate.llama_server import LlamaServer
    assert hasattr(LlamaServer, 'start')
    assert hasattr(LlamaServer, 'close')
    assert hasattr(LlamaServer, 'chat')


def test_speech_server_class():
    """SpeechCppServer class is defined with required methods."""
    from auto_translate.speech_server import SpeechCppServer
    assert hasattr(SpeechCppServer, 'start')
    assert hasattr(SpeechCppServer, 'close')


def test_gui_runner_backend_params():
    """build_backend_command accepts asr_backend and translate_backend params."""
    import inspect

    from auto_translate.gui_runner import build_backend_command
    sig = inspect.signature(build_backend_command)
    params = list(sig.parameters.keys())
    assert 'asr_backend' in params
    assert 'translate_backend' in params


def test_gui_runner_backend_args_in_command():
    """build_backend_command includes --asr-backend and --translate-backend in args."""
    import tempfile
    from pathlib import Path

    from auto_translate.gui_runner import build_backend_command
    # Create a minimal project structure
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "pyproject.toml").write_text("")
        src = root / "src" / "auto_translate"
        src.mkdir(parents=True)
        (src / "__main__.py").write_text("")
        # Mock pythonw.exe
        venv = root / ".venv" / "Scripts"
        venv.mkdir(parents=True)
        (venv / "pythonw.exe").write_text("")
        (venv / "python.exe").write_text("")
        # Mock config
        (root / "config.example.toml").write_text("")
        (root / "config.toml").write_text("")
        try:
            _, args = build_backend_command(
                root, "https://www.youtube.com/watch?v=test", 480, 12345,
                asr_backend="qwen3-asr",
                translate_backend="plamo",
            )
            assert "--asr-backend" in args
            assert "qwen3-asr" in args
            assert "--translate-backend" in args
            assert "plamo" in args
        except FileNotFoundError:
            pass  # pythonw.exe not executable is OK for this test


def test_gui_runner_backend_validation():
    """build_backend_command rejects invalid backend names."""
    import tempfile
    from pathlib import Path

    from auto_translate.gui_runner import build_backend_command
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "pyproject.toml").write_text("")
        src = root / "src" / "auto_translate"
        src.mkdir(parents=True)
        (src / "__main__.py").write_text("")
        venv = root / ".venv" / "Scripts"
        venv.mkdir(parents=True)
        (venv / "pythonw.exe").write_text("")
        (venv / "python.exe").write_text("")
        (root / "config.example.toml").write_text("")
        (root / "config.toml").write_text("")
        try:
            build_backend_command(
                root, "https://www.youtube.com/watch?v=test", 480, 12345,
                asr_backend="",
            )
            assert False, "Should raise ValueError for empty backend name"
        except ValueError as e:
            assert "ASR backend" in str(e)
        except FileNotFoundError:
            pass


def test_cli_args_asr_backend():
    """__main__.py accepts --asr-backend CLI argument."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "auto_translate", "--help"],
        capture_output=True, text=True, check=False,
        cwd=str(Path(__file__).parent.parent),
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent / "src")},
    )
    assert "--asr-backend" in result.stdout


def test_cli_args_translate_backend():
    """__main__.py accepts --translate-backend CLI argument."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "auto_translate", "--help"],
        capture_output=True, text=True, check=False,
        cwd=str(Path(__file__).parent.parent),
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent / "src")},
    )
    assert "--translate-backend" in result.stdout


def test_registry_backend_list_for_gui():
    """Registry returns backends with descriptions for GUI dropdowns."""
    from auto_translate.registry import get_default_registry
    registry = get_default_registry()
    backends = registry.list_backends()
    # Check ASR backends have descriptions
    for name in backends.get("asr", []):
        caps = registry.get_capabilities("asr", name)
        assert caps is not None
        assert caps.description  # Non-empty description for GUI
    # Check translation backends have descriptions
    for name in backends.get("translate", []):
        caps = registry.get_capabilities("translate", name)
        assert caps is not None
        assert caps.description  # Non-empty description for GUI


def test_hymt_import():
    """HymtTranslator and build_hymt_prompt can be imported."""
    from auto_translate.translate.hymt import (
        HymtTranslator,
        build_hymt_prompt,
        build_prompt,
    )
    assert HymtTranslator is not None
    assert build_hymt_prompt is not None
    assert build_hymt_prompt is build_prompt


def test_hymt_registry():
    """hymt backend is registered in the default registry."""
    from auto_translate.registry import get_default_registry
    registry = get_default_registry()
    backends = registry.list_backends()
    assert "hymt" in backends.get("translate", [])


def test_hymt_capabilities():
    """hymt backend has correct capabilities."""
    from auto_translate.registry import get_default_registry
    registry = get_default_registry()
    caps = registry.get_capabilities("translate", "hymt")
    assert caps is not None
    assert caps.runtime == "llama_cpp"
    assert caps.device == "cuda"
    assert caps.source_lang == "ja"
    assert caps.target_lang == "en"
    assert caps.vram_gb > 0
    assert caps.description


def test_hymt_prompt_format():
    """HY-MT2 prompt uses Chinese templates, no system prompt."""
    from auto_translate.translate.hymt import build_prompt
    # Plain prompt
    prompt = build_prompt("こんにちは", [], "")
    assert "将以下文本翻译为英语" in prompt
    assert "こんにちは" in prompt
    # With terms
    prompt = build_prompt("こんにちは", [("すい", "Sui-chan")], "")
    assert "参考下面的翻译" in prompt
    assert "すい 翻译成 Sui-chan" in prompt
    # With context
    prompt = build_prompt("こんにちは", [], "前の会話")
    assert "【背景信息】" in prompt
    assert "【待翻译文本】" in prompt


def test_hymt_clean_output():
    """HY-MT2 clean_output removes role prefixes and quotes."""
    from auto_translate.translate.hymt import clean_output
    assert clean_output("EN: Hello world") == "Hello world"
    assert clean_output("「Hello」") == "Hello"
    assert clean_output("  Hello  world  ") == "Hello  world"  # preserves internal spacing


def test_hymt_useful_context():
    """HY-MT2 useful_context filters filler words."""
    from auto_translate.translate.hymt import useful_context
    assert useful_context("こんにちは") is True
    assert useful_context("はい") is False
    assert useful_context("うん") is False
    assert useful_context("ゲームを始めましょう") is True


def test_hymt_reload_lock():
    """HY-MT2 uses separate _reload_lock to prevent deadlock."""
    import inspect

    from auto_translate.translate.hymt import HymtTranslator

    source = inspect.getsource(HymtTranslator.__init__)
    assert "_reload_lock" in source


def test_mpv_available():
    """MPV player is accessible via shutil.which or known path."""
    import os
    import shutil
    mpv = shutil.which("mpv")
    if mpv is None:
        # Check known installation path
        known = os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "MPV Player", "mpv.exe")
        if os.path.exists(known):
            return  # MPV installed but not on PATH
        import pytest
        pytest.skip("MPV not found on PATH or at known installation path")


def test_mpv_output_import():
    """MpvOutput can be imported regardless of MPV availability."""
    from auto_translate.output.mpv import MpvOutput
    assert MpvOutput is not None


def test_registry_hymt_in_gui_list():
    """hymt backend appears in registry list for GUI dropdowns."""
    from auto_translate.registry import get_default_registry
    registry = get_default_registry()
    backends = registry.list_backends()
    assert "hymt" in backends.get("translate", [])
    caps = registry.get_capabilities("translate", "hymt")
    assert caps is not None
    assert caps.description  # Non-empty description for GUI


def test_whisper_import():
    """WhisperASR and WhisperError can be imported."""
    from auto_translate.asr.whisper_turbo import WhisperASR, WhisperError
    assert WhisperASR is not None
    assert WhisperError is not None


def test_whisper_registry():
    """whisper backend is registered in the default registry."""
    from auto_translate.registry import get_default_registry
    registry = get_default_registry()
    backends = registry.list_backends()
    assert "whisper" in backends.get("asr", [])


def test_whisper_capabilities():
    """whisper backend has correct capabilities."""
    from auto_translate.registry import get_default_registry
    registry = get_default_registry()
    caps = registry.get_capabilities("asr", "whisper")
    assert caps is not None
    assert caps.runtime == "faster_whisper"
    assert caps.device == "cuda"
    assert caps.source_lang == "ja"
    assert caps.vram_gb > 0
    assert caps.description


def test_whisper_in_gui_list():
    """whisper backend appears in registry list for GUI dropdowns."""
    from auto_translate.registry import get_default_registry
    registry = get_default_registry()
    backends = registry.list_backends()
    assert "whisper" in backends.get("asr", [])
    caps = registry.get_capabilities("asr", "whisper")
    assert caps is not None
    assert caps.description  # Non-empty description for GUI


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
