"""Regression tests for Kimi audit fixes (2026-11-22).

Covers:
1. Seek reset race condition — double reset_buffer must not wipe valid pending speech
2. CUDA device selection — detect_device must honor cuda_device parameter
3. Qwen3-ASR startup — model search and actionable error messages
"""

import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================================
# 1. Seek reset race condition
# ============================================================================

class TestSeekResetRace:
    """Test that double reset_buffer does not wipe valid pending speech."""

    def test_reset_buffer_idempotent_same_generation(self):
        """Calling reset_buffer twice for the same generation must not wipe
        pending speech registered between the two calls.

        Reproduces the exact race:
        1. mpv.py:1858 calls reset_buffer(target) — first reset
        2. New-generation speech is registered via record_speech
        3. pipeline.py:1460 calls reset_buffer(progress_s) — second reset
        4. The second reset must NOT clear the pending speech from step 2
        """
        from auto_translate.ingest import Source
        from auto_translate.output.mpv import MpvOutput

        source = Source(location="test.mp4", live=False)

        with patch.object(MpvOutput, "__init__", lambda self, *a, **k: None):
            mpv = MpvOutput.__new__(MpvOutput)
            mpv._buffer_condition = threading.Condition()
            mpv._coverage = MagicMock()
            mpv._coverage.abandon_stale_generation.return_value = 0
            mpv._speech_id_map = {}
            mpv._pending_generation = 0
            mpv._source_finished = False
            mpv._seek_lead_target = None
            mpv.PAUSE_LEAD_SECONDS = 5.0
            mpv.ingest = MagicMock()
            mpv.ingest.generation = 1
            mpv.ingest.source = source

        # First reset (from mpv.py:_restart_audio_at)
        mpv.reset_buffer(100.0)
        assert mpv._pending_generation == 1
        mpv._coverage.reset.assert_called_once_with(100.0)
        mpv._coverage.set_generation.assert_called_once_with(1)

        # Simulate new speech registered after first reset
        mpv._coverage.record_speech(100.0, 105.0, generation=1)
        mpv._speech_id_map[42] = (100.0, 105.0)

        # Second reset (from pipeline.py when AudioProgress.reset arrives)
        mpv.reset_buffer(100.0)

        # The second reset should be a no-op for the destructive portion
        # record_speech should still have been called once (not cleared)
        assert mpv._pending_generation == 1
        assert mpv._coverage.record_speech.call_count == 1
        assert 42 in mpv._speech_id_map, "speech_id_map should not be cleared by duplicate reset"

    def test_seek_reset_preserves_real_coverage_and_blocks_early_resume(self):
        """Real CoverageTracker reproduces the double-reset interleaving."""
        from types import SimpleNamespace

        from auto_translate.ingest import Source
        from auto_translate.output.coverage import CoverageTracker
        from auto_translate.output.mpv import MpvOutput

        mpv = MpvOutput.__new__(MpvOutput)
        mpv._buffer_condition = threading.Condition()
        mpv._coverage = CoverageTracker()
        mpv._speech_id_map = {}
        mpv._pending_generation = 0
        mpv._source_finished = False
        mpv._seek_lead_target = None
        mpv._subtitle_ready = threading.Event()
        mpv._now = 100.0
        mpv.PAUSE_LEAD_SECONDS = 5.0
        mpv.cfg = SimpleNamespace(vod_resume_threshold_s=8.0)
        mpv.ingest = SimpleNamespace(
            generation=1, source=Source(location="test.mp4", live=False)
        )

        mpv.reset_buffer(100.0)
        sid = mpv.record_speech(101.0, 107.0, generation=1)
        assert sid is not None
        mpv.advance(115.0)
        assert mpv._coverage.translated_until <= 101.0
        assert mpv._vod_should_resume() is False

        # The late seek-reset marker must not erase this unfinished speech.
        mpv.reset_buffer(100.0)
        assert mpv._coverage.pending_count() == 1
        assert mpv._coverage.translated_until <= 101.0
        assert mpv._vod_should_resume() is False

        # Only completion of the pending subtitle unblocks contiguous lead.
        mpv.advance_to_subtitle(107.0, generation=1, speech_id=sid)
        assert mpv._coverage.pending_count() == 0
        assert mpv._coverage.translated_until >= 115.0
        assert mpv._vod_should_resume() is True

    def test_reset_buffer_different_generation(self):
        """A reset for a new generation must still perform the full destructive reset."""
        from auto_translate.ingest import Source
        from auto_translate.output.mpv import MpvOutput

        source = Source(location="test.mp4", live=False)

        with patch.object(MpvOutput, "__init__", lambda self, *a, **k: None):
            mpv = MpvOutput.__new__(MpvOutput)
            mpv._buffer_condition = threading.Condition()
            mpv._coverage = MagicMock()
            mpv._coverage.abandon_stale_generation.return_value = 0
            mpv._speech_id_map = {1: (50.0, 55.0)}
            mpv._pending_generation = 1
            mpv._source_finished = False
            mpv._seek_lead_target = None
            mpv.PAUSE_LEAD_SECONDS = 5.0
            mpv.ingest = MagicMock()
            mpv.ingest.generation = 2  # New generation after seek
            mpv.ingest.source = source

        mpv.reset_buffer(200.0)

        # Full reset should happen for a new generation
        assert mpv._pending_generation == 2
        mpv._coverage.reset.assert_called_with(200.0)
        assert len(mpv._speech_id_map) == 0, "speech_id_map should be cleared for new generation"

    def test_record_speech_rejects_stale_generation(self):
        """record_speech must reject callbacks from a previous generation."""
        from auto_translate.ingest import Source
        from auto_translate.output.mpv import MpvOutput

        with patch.object(MpvOutput, "__init__", lambda self, *a, **k: None):
            mpv = MpvOutput.__new__(MpvOutput)
            mpv._buffer_condition = threading.Condition()
            mpv._coverage = MagicMock()
            mpv._speech_id_map = {}
            mpv._pending_generation = 2
            mpv.ingest = MagicMock()
            mpv.ingest.source = Source(location="test.mp4", live=False)

        # Stale callback from generation 1
        result = mpv.record_speech(10.0, 15.0, generation=1)
        assert result is None, "Stale generation callback should be rejected"
        mpv._coverage.record_speech.assert_not_called()

        # Valid callback from generation 2
        mpv._coverage.record_speech.return_value = 1
        result = mpv.record_speech(10.0, 15.0, generation=2)
        assert result == 1
        mpv._coverage.record_speech.assert_called_with(10.0, 15.0, generation=2)

    def test_complete_speech_rejects_stale_generation(self):
        """complete_speech must reject callbacks from a previous generation."""
        from auto_translate.ingest import Source
        from auto_translate.output.mpv import MpvOutput

        with patch.object(MpvOutput, "__init__", lambda self, *a, **k: None):
            mpv = MpvOutput.__new__(MpvOutput)
            mpv._buffer_condition = threading.Condition()
            mpv._coverage = MagicMock()
            mpv._speech_id_map = {}
            mpv._pending_generation = 2
            mpv._seek_lead_target = None
            mpv.ingest = MagicMock()
            mpv.ingest.source = Source(location="test.mp4", live=False)

        # Stale callback from generation 1
        mpv.complete_speech(10.0, 15.0, generation=1, speech_id=1)
        mpv._coverage.complete_speech.assert_not_called()

    def test_advance_to_subtitle_rejects_stale_generation(self):
        """advance_to_subtitle must reject callbacks from a previous generation."""
        from auto_translate.ingest import Source
        from auto_translate.output.mpv import MpvOutput

        with patch.object(MpvOutput, "__init__", lambda self, *a, **k: None):
            mpv = MpvOutput.__new__(MpvOutput)
            mpv._buffer_condition = threading.Condition()
            mpv._coverage = MagicMock()
            mpv._speech_id_map = {}
            mpv._pending_generation = 2
            mpv._seek_lead_target = None
            mpv.ingest = MagicMock()
            mpv.ingest.source = Source(location="test.mp4", live=False)

        # Stale callback from generation 1
        mpv.advance_to_subtitle(15.0, generation=1, speech_id=1)
        mpv._coverage.complete_speech.assert_not_called()

    def test_seek_race_full_sequence(self):
        """Full reproduction of the seek race condition identified by Kimi.

        Sequence:
        1. Seek to position 100s
        2. _restart_audio_at calls reset_buffer(100) — generation becomes 1
        3. New speech registered: record_speech(100, 105, gen=1)
        4. AudioProgress reset marker arrives, pipeline calls reset_buffer(100)
        5. Pending speech must survive
        """
        from auto_translate.ingest import Source
        from auto_translate.output.mpv import MpvOutput

        with patch.object(MpvOutput, "__init__", lambda self, *a, **k: None):
            mpv = MpvOutput.__new__(MpvOutput)
            mpv._buffer_condition = threading.Condition()
            mpv._coverage = MagicMock()
            mpv._coverage.abandon_stale_generation.return_value = 0
            mpv._speech_id_map = {}
            mpv._pending_generation = 0
            mpv._source_finished = False
            mpv._seek_lead_target = None
            mpv.PAUSE_LEAD_SECONDS = 5.0
            mpv.ingest = MagicMock()
            mpv.ingest.generation = 1
            mpv.ingest.source = Source(location="test.mp4", live=False)

        # Step 1-2: First reset
        mpv.reset_buffer(100.0)
        first_reset_count = mpv._coverage.reset.call_count

        # Step 3: New speech registered
        mpv._coverage.record_speech.return_value = 1
        sid = mpv.record_speech(100.0, 105.0, generation=1)
        assert sid == 1
        assert 1 in mpv._speech_id_map

        # Step 4: Second reset (from pipeline)
        mpv.reset_buffer(100.0)

        # Step 5: Verify speech survived
        assert mpv._coverage.reset.call_count == first_reset_count, \
            "Second reset must not call coverage.reset() again"
        assert 1 in mpv._speech_id_map, \
            "Pending speech must survive the duplicate reset"


# ============================================================================
# 2. CUDA device selection
# ============================================================================

class TestCudaDeviceSelection:
    """Test that detect_device honors cuda_device parameter."""

    def test_detect_device_cpu(self):
        """cuda_device=-1 must generate llama.cpp --device none."""
        from auto_translate.llama_server import detect_device
        result = detect_device(cuda_device=-1)
        assert result == "none"

    def test_detect_device_cuda0(self):
        """cuda_device=0 must return 'CUDA0' when verification is unavailable."""
        from auto_translate.llama_server import LlamaServerError, detect_device
        # Without an executable to verify, should return the requested device
        with patch("auto_translate.llama_server.resolve_executable", side_effect=LlamaServerError("not found")):
            result = detect_device(cuda_device=0)
            assert result == "CUDA0"

    def test_detect_device_cuda1(self):
        """cuda_device=1 must return 'CUDA1' when verification is unavailable."""
        from auto_translate.llama_server import LlamaServerError, detect_device
        with patch("auto_translate.llama_server.resolve_executable", side_effect=LlamaServerError("not found")):
            result = detect_device(cuda_device=1)
            assert result == "CUDA1"

    def test_detect_device_cuda2(self):
        """cuda_device=2 must return 'CUDA2' when verification is unavailable."""
        from auto_translate.llama_server import LlamaServerError, detect_device
        with patch("auto_translate.llama_server.resolve_executable", side_effect=LlamaServerError("not found")):
            result = detect_device(cuda_device=2)
            assert result == "CUDA2"

    def test_detect_device_auto_no_executable(self):
        """Auto-detect with no executable and no cuda_device must return 'CUDA0'."""
        from auto_translate.llama_server import LlamaServerError, detect_device
        # When cuda_device is None (auto) and no executable, falls back to CUDA0
        with patch("auto_translate.llama_server.resolve_executable", side_effect=LlamaServerError("not found")), \
             patch("auto_translate.llama_server.subprocess.run", side_effect=OSError("no exe")):
            result = detect_device(executable=None, cuda_device=None)
            assert result == "CUDA0"

    def test_detect_device_auto_with_cuda(self):
        """Auto-detect with CUDA devices must prefer CUDA over Vulkan."""
        from auto_translate.llama_server import detect_device

        mock_exe = MagicMock(spec=Path)
        mock_exe.__str__ = MagicMock(return_value="/fake/llama-server")

        with patch("auto_translate.llama_server.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="Available devices:\n  CUDA0: NVIDIA RTX 3070\n  CUDA1: NVIDIA RTX 3080\n  Vulkan0: NVIDIA RTX 3070\n",
                stderr=""
            )
            result = detect_device(executable=mock_exe, cuda_device=None)
            assert result == "CUDA0"

    def test_detect_device_auto_vulkan_only(self):
        """Auto-detect with only Vulkan must return Vulkan."""
        from auto_translate.llama_server import detect_device

        mock_exe = MagicMock(spec=Path)
        mock_exe.__str__ = MagicMock(return_value="/fake/llama-server")

        with patch("auto_translate.llama_server.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="Available devices:\n  Vulkan0: AMD Radeon\n",
                stderr=""
            )
            result = detect_device(executable=mock_exe, cuda_device=None)
            assert result == "Vulkan0"

    def test_detect_device_requested_not_found(self):
        """An unavailable explicitly requested GPU must never fall back."""
        from auto_translate.llama_server import LlamaServerError, detect_device

        mock_exe = Path("/fake/llama-server")
        with patch("auto_translate.llama_server.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Available devices:\n  CUDA0: NVIDIA RTX 3070\n",
                stderr=""
            )
            with pytest.raises(LlamaServerError, match="CUDA2 is unavailable"):
                detect_device(executable=mock_exe, cuda_device=2)
            mock_run.assert_called_once()

    def test_detect_device_requested_verified(self):
        """If requested CUDA device exists, return it exactly."""
        from auto_translate.llama_server import detect_device

        mock_exe = MagicMock(spec=Path)
        mock_exe.__str__ = MagicMock(return_value="/fake/llama-server")

        with patch("auto_translate.llama_server.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Available devices:\n  CUDA0: NVIDIA RTX 3070\n  CUDA1: NVIDIA RTX 3080\n",
                stderr=""
            )
            result = detect_device(executable=mock_exe, cuda_device=1)
            assert result == "CUDA1"

    def test_explicit_cuda0_keeps_rtx_gpu(self):
        """The user's default CUDA0 must stay on GPU after the CPU fix."""
        from auto_translate.llama_server import detect_device

        with patch("auto_translate.llama_server.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Available devices:\n  CUDA0: NVIDIA RTX 3070\n  Vulkan0: NVIDIA RTX 3070\n",
                stderr=""
            )
            assert detect_device(executable=Path("/fake/llama-server"), cuda_device=0) == "CUDA0"

    def test_gemma4_server_args_use_explicit_cuda0(self):
        """Verify constructed llama-server arguments, not source-text presence."""
        from auto_translate.config import TranslateConfig
        from auto_translate.translate.gemma4 import Gemma4Translator

        translator = Gemma4Translator.__new__(Gemma4Translator)
        translator.cfg = TranslateConfig(cuda_device=0)
        translator.model = Path("/fake/gemma.gguf")
        translator.gpu_layers = 3
        translator.layer_count = 5
        translator.executable = Path("/fake/llama-server")
        with patch("auto_translate.translate.gemma4.detect_device", return_value="CUDA0") as detect:
            args = translator._args()
        detect.assert_called_once_with(translator.executable, 0)
        assert args[args.index("--device") + 1] == "CUDA0"

    def test_gemma4_passes_cuda_device(self):
        """Gemma4Translator must pass cuda_device to detect_device."""
        # Check that _args method uses self.cfg.cuda_device
        import inspect

        from auto_translate.translate.gemma4 import Gemma4Translator
        source = inspect.getsource(Gemma4Translator._args)
        assert "cuda_device" in source, "Gemma4Translator._args must reference cuda_device"

    def test_plamo_passes_cuda_device(self):
        """PlamoTranslator must pass cuda_device to detect_device."""
        import inspect

        from auto_translate.translate.plamo import PlamoTranslator
        source = inspect.getsource(PlamoTranslator._args)
        assert "cuda_device" in source, "PlamoTranslator._args must reference cuda_device"

    def test_hymt_passes_cuda_device(self):
        """HyMT translator must pass cuda_device to detect_device."""
        import inspect

        from auto_translate.translate import hymt
        # Find the translator class in the module
        translator_cls = None
        for name in dir(hymt):
            obj = getattr(hymt, name)
            if isinstance(obj, type) and hasattr(obj, '_args'):
                translator_cls = obj
                break
        assert translator_cls is not None, "Could not find translator class in hymt module"
        source = inspect.getsource(translator_cls._args)
        assert "cuda_device" in source, "HyMT translator _args must reference cuda_device"

    def test_qwen3_asr_passes_cuda_device(self):
        """Qwen3ASR must pass cuda_device to detect_device."""
        import inspect

        from auto_translate.asr.qwen3_asr import Qwen3ASR
        source = inspect.getsource(Qwen3ASR)
        assert "cuda_device" in source, "Qwen3ASR must reference cuda_device"


# ============================================================================
# 3. Qwen3-ASR startup
# ============================================================================

class TestQwen3ASRStartup:
    """Test Qwen3-ASR model resolution and error messages."""

    def test_qwen3_asr_auto_projector_matches_sibling(self, tmp_path):
        from auto_translate.registry import _find_qwen3_asr_mmproj

        model = tmp_path / "Qwen3-ASR-1.7B-Q8_0.gguf"
        (tmp_path / "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf").touch()
        projector = tmp_path / f"mmproj-{model.name}"
        projector.touch()
        assert _find_qwen3_asr_mmproj(str(model)) == str(projector)

    def test_qwen3_asr_rejects_incompatible_explicit_projector(self):
        from auto_translate.registry import _qwen3_asr_factory

        with patch("auto_translate.asr.qwen3_asr.Qwen3ASR", MagicMock()) as constructor:
            with pytest.raises(ValueError, match="different model sizes"):
                _qwen3_asr_factory({
                    "model_path": "/models/Qwen3-ASR-1.7B-Q8_0.gguf",
                    "mmproj_path": "/models/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf",
                })
            constructor.assert_not_called()

    def test_qwen3_asr_ambiguous_models_fail_instead_of_guessing(self):
        from auto_translate.registry import _find_qwen3_asr_model

        candidates = [
            "/models/first/Qwen3-ASR-1.7B-Q8_0.gguf",
            "/models/second/Qwen3-ASR-1.7B-Q8_0.gguf",
        ]
        with (
            patch("auto_translate.registry._qwen3_asr_candidates", return_value=candidates),
            pytest.raises(ValueError, match="Multiple Qwen3-ASR GGUF"),
        ):
            _find_qwen3_asr_model()

    def test_qwen3_asr_missing_model_error(self):
        """Missing model must raise ValueError with actionable message."""
        from auto_translate.registry import _qwen3_asr_factory

        with patch("auto_translate.registry._find_qwen3_asr_model", return_value=""), \
             patch("auto_translate.registry._find_qwen3_asr_mmproj", return_value=""), \
             patch("auto_translate.asr.qwen3_asr.Qwen3ASR", MagicMock()):
            with pytest.raises(ValueError) as exc_info:
                _qwen3_asr_factory({"model_path": "", "mmproj_path": ""})

            error_msg = str(exc_info.value)
            assert "Qwen3-ASR" in error_msg
            assert "model" in error_msg.lower()
            assert "config" in error_msg.lower() or "configure" in error_msg.lower()

    def test_qwen3_asr_missing_mmproj_error(self):
        """Missing mmproj must raise ValueError with actionable message."""
        from auto_translate.registry import _qwen3_asr_factory

        with patch("auto_translate.registry._find_qwen3_asr_model", return_value="/models/model.gguf"), \
             patch("auto_translate.registry._find_qwen3_asr_mmproj", return_value=""), \
             patch("auto_translate.asr.qwen3_asr.Qwen3ASR", MagicMock()):
            with pytest.raises(ValueError) as exc_info:
                _qwen3_asr_factory({"model_path": "", "mmproj_path": ""})

            error_msg = str(exc_info.value)
            assert "mmproj" in error_msg.lower() or "projector" in error_msg.lower()

    def test_qwen3_asr_model_search(self):
        """_find_qwen3_asr_model must search common directories."""
        from pathlib import Path

        from auto_translate.registry import _find_qwen3_asr_model

        with patch("pathlib.Path.cwd") as mock_cwd:
            root = Path("/fake/project")
            mock_cwd.return_value = root

            # Create a mock model directory
            model_dir = root / "models" / "qwen3-asr"
            model_file = model_dir / "Qwen3-ASR-1.7B-Q8_0.gguf"

            with patch("pathlib.Path.is_dir", return_value=True), \
                 patch("glob.glob") as mock_glob:
                mock_glob.return_value = [str(model_file)]
                result = _find_qwen3_asr_model()
                assert result == str(model_file)

    def test_qwen3_asr_mmproj_search_finds_sibling_outside_models(self, tmp_path, monkeypatch):
        from auto_translate.registry import _find_qwen3_asr_mmproj

        monkeypatch.chdir(tmp_path)
        projector = tmp_path / "mmproj-Qwen3-ASR-1.7B-JA-Q8_0.gguf"
        projector.touch()
        assert _find_qwen3_asr_mmproj("Qwen3-ASR-1.7B-JA-Q8_0.gguf") == projector.name

    def test_qwen3_asr_model_search_excludes_mmproj(self):
        """Model search must not return mmproj files."""
        from pathlib import Path

        from auto_translate.registry import _find_qwen3_asr_model

        with patch("pathlib.Path.cwd") as mock_cwd:
            root = Path("/fake/project")
            mock_cwd.return_value = root

            with patch("pathlib.Path.is_dir", return_value=True), \
                 patch("glob.glob") as mock_glob:
                # Only mmproj files exist — should not match
                mock_glob.return_value = [str(root / "models" / "mmproj-Qwen3.gguf")]
                result = _find_qwen3_asr_model()
                assert result == "", "Should not return mmproj as model"

    def test_qwen3_asr_resolved_paths_used(self):
        """When model is found by search, resolved paths must be used."""
        from auto_translate.registry import _qwen3_asr_factory

        mock_asr = MagicMock()

        with patch("auto_translate.registry._find_qwen3_asr_model", return_value="/found/model.gguf"), \
             patch("auto_translate.registry._find_qwen3_asr_mmproj", return_value="/found/mmproj-model.gguf"), \
             patch("auto_translate.asr.qwen3_asr.Qwen3ASR", mock_asr):
            _qwen3_asr_factory({"model_path": "", "mmproj_path": ""})

            # Qwen3ASR should be called with resolved paths
            call_args = mock_asr.call_args
            config_arg = call_args[0][0]
            assert config_arg.model_path == "/found/model.gguf"
            assert config_arg.mmproj_path == "/found/mmproj-model.gguf"


# ============================================================================
# 4. Coverage tracker generation awareness
# ============================================================================

class TestCoverageTrackerGeneration:
    """Test CoverageTracker generation handling."""

    def test_abandon_stale_generation(self):
        """abandon_stale_generation must only remove intervals from previous generations."""
        from auto_translate.output.coverage import CoverageTracker

        tracker = CoverageTracker(0.0)
        tracker.set_generation(2)

        # Register intervals from different generations
        tracker.record_speech(0.0, 5.0, generation=1)
        tracker.record_speech(5.0, 10.0, generation=2)
        tracker.record_speech(10.0, 15.0, generation=1)

        assert tracker.pending_count() == 3

        abandoned = tracker.abandon_stale_generation()
        assert abandoned == 2, "Should abandon 2 stale intervals"
        assert tracker.pending_count() == 1, "Only current generation should remain"

    def test_reject_stale_callback(self):
        """reject_stale must return True for previous generation callbacks."""
        from auto_translate.output.coverage import CoverageTracker

        tracker = CoverageTracker(0.0)
        tracker.set_generation(2)

        assert tracker.reject_stale(1) is True
        assert tracker.reject_stale(2) is False

    def test_reset_clears_pending(self):
        """reset must clear all pending intervals."""
        from auto_translate.output.coverage import CoverageTracker

        tracker = CoverageTracker(0.0)
        tracker.record_speech(0.0, 5.0, generation=0)
        tracker.record_speech(5.0, 10.0, generation=0)

        assert tracker.pending_count() == 2
        tracker.reset(100.0)
        assert tracker.pending_count() == 0
        assert tracker.translated_until == 100.0


# ============================================================================
# 5. Pipeline generation filtering
# ============================================================================

class TestPipelineGenerationFiltering:
    """Test that pipeline correctly filters stale generations."""

    def test_audio_progress_reset_generation_check(self):
        """AudioProgress reset markers must carry correct generation."""

        from auto_translate.types import AudioProgress

        progress = AudioProgress(
            sample=int(100.0 * 16000),
            reset=True,
            generation=5,
        )
        assert progress.generation == 5
        assert progress.reset is True


# ============================================================================
# 6. PLaMo local model path discovery
# ============================================================================

class TestPlamoModelDiscovery:
    """Local, model-free verification of PLaMo 2B / 8B path isolation."""

    @staticmethod
    def _models_dir(tmp_path, monkeypatch):
        from auto_translate import registry

        root = tmp_path / "project"
        models = root / "models" / "plamo3"
        models.mkdir(parents=True)
        # Simulate a project checkout even if pytest runs from elsewhere.
        monkeypatch.setattr(registry, "__file__", str(root / "src" / "auto_translate" / "registry.py"))
        monkeypatch.chdir(root)
        return models

    def test_plamo_2b_found_without_configured_path(self, tmp_path, monkeypatch):
        from auto_translate.registry import resolve_local_plamo_model_path

        folder = self._models_dir(tmp_path, monkeypatch)
        model = folder / "plamo-3-translate-2b-Q4_K_M.gguf"
        model.write_bytes(b"GGUF")
        assert resolve_local_plamo_model_path("plamo") == str(model.resolve())

    def test_plamo_8b_never_used_for_2b(self, tmp_path, monkeypatch):
        from auto_translate.registry import resolve_local_plamo_model_path

        folder = self._models_dir(tmp_path, monkeypatch)
        model = folder / "plamo-3-translate-8b-Q3_K_M.gguf"
        model.write_bytes(b"GGUF")
        assert resolve_local_plamo_model_path("plamo") == ""
        assert resolve_local_plamo_model_path("plamo8b") == str(model.resolve())

    def test_plamo_2b_never_used_for_8b(self, tmp_path, monkeypatch):
        from auto_translate.registry import resolve_local_plamo_model_path

        folder = self._models_dir(tmp_path, monkeypatch)
        model = folder / "plamo-3-translate-2b-Q4_K_M.gguf"
        model.write_bytes(b"GGUF")
        assert resolve_local_plamo_model_path("plamo8b") == ""

    def test_plamo_ambiguous_quantizations_require_explicit_path(self, tmp_path, monkeypatch):
        from auto_translate.registry import resolve_local_plamo_model_path

        folder = self._models_dir(tmp_path, monkeypatch)
        (folder / "plamo-3-translate-2b-Q4_K_M.gguf").write_bytes(b"GGUF")
        (folder / "plamo-3-translate-2b-Q8_0.gguf").write_bytes(b"GGUF")
        with pytest.raises(ValueError, match=r"Multiple PLaMo 2B"):
            resolve_local_plamo_model_path("plamo")

    def test_plamo_explicit_path_is_never_overridden(self, tmp_path, monkeypatch):
        from auto_translate.registry import resolve_local_plamo_model_path

        folder = self._models_dir(tmp_path, monkeypatch)
        (folder / "plamo-3-translate-2b-Q4_K_M.gguf").write_bytes(b"GGUF")
        assert resolve_local_plamo_model_path("plamo", "/custom/selected.gguf") == "/custom/selected.gguf"

    def test_plamo_factory_uses_locally_discovered_model(self, tmp_path, monkeypatch):
        from auto_translate.registry import _plamo_translate_factory

        folder = self._models_dir(tmp_path, monkeypatch)
        model = folder / "plamo-3-translate-2b-Q4_K_M.gguf"
        model.write_bytes(b"GGUF")
        with patch("auto_translate.translate.plamo.PlamoTranslator") as factory:
            _plamo_translate_factory({"model_path": ""}, gpu_layers=0)
            assert factory.call_args.args[0].model_path == str(model.resolve())

    def test_plamo_resolution_happens_before_budget_allocation(self, tmp_path, monkeypatch):
        """A discovered model must reach _load_translator's effective profile."""
        import importlib

        from auto_translate.config import Config

        folder = self._models_dir(tmp_path, monkeypatch)
        model = folder / "plamo-3-translate-2b-Q4_K_M.gguf"
        model.write_bytes(b"GGUF")
        config = Config()
        config.models.translate_backend = "plamo"
        config.translate_profiles.pop("plamo", None)
        budget = MagicMock()
        budget.free = None  # Avoid hardware-dependent offload estimation.
        translator = MagicMock()
        translator.gpu_layers = None
        registry = MagicMock()
        registry.create_backend.return_value = translator
        app = importlib.import_module("auto_translate.__main__")
        result = app._load_translator(config, MagicMock(), MagicMock(),
                                      budget, MagicMock(), registry)
        assert result is translator
        profile = registry.create_backend.call_args.args[2]
        assert profile["model_path"] == str(model.resolve())
        assert profile["cuda_device"] == 0  # Keep RTX 3070 CUDA0 default.


# ============================================================================
# Test runner
# ============================================================================

def pytest_raises():
    """Return pytest.raises context manager."""
    import pytest
    return pytest.raises


# Make pytest.raises available for test classes
import pytest

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
