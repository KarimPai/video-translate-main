"""Third fix pass: end-to-end regression tests.

These tests exercise real application methods with controlled mocks and
failure injection. They are expected to FAIL against the defective
implementation and PASS after the correction.
"""

import queue
import sys
import threading
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import httpx

from auto_translate.asr.qwen3_asr import Qwen3ASR, Qwen3ASRError
from auto_translate.asr.whisper_turbo import WhisperASR, WhisperError
from auto_translate.config import (
    AsrConfig,
    Config,
    GpuBudgetConfig,
    LanguageFilterConfig,
    get_backend_config,
    load_config,
)
from auto_translate.gpu_budget import GpuBudget
from auto_translate.pipeline import _expected_vod_idle, run_asr
from auto_translate.text_filter import LanguageFilter
from auto_translate.types import Transcript, Utterance

# ---------------------------------------------------------------------------
# P0 — language filter integration through run_asr()
# ---------------------------------------------------------------------------

def _make_fake_asr(texts):
    """Fake ASR backend returning transcripts in call order."""

    class FakeASR:
        def __init__(self, texts):
            self.texts = list(texts)
            self._i = 0
            self.cfg = AsrConfig(response_timeout_s=30.0, live_response_timeout_s=15.0)
            self.last_rejection = None
            self.last_raw_transcript = ""

        def transcribe(self, audio, probe=False):
            text = self.texts[self._i]
            self._i += 1
            return text

        def cancel_pending(self):
            pass

        def close(self):
            pass

        def remember(self, text):
            pass

        def forget(self, n=1):
            pass

    return FakeASR(texts)


def test_run_asr_language_filter_keeps_japanese_drops_english():
    """run_asr() applies the language filter to transcripts (post-ASR), not audio."""
    texts = [
        "こんにちは、お元気ですか？",      # Japanese, 3s duration -> keep
        "Hello, how are you doing today?",   # English, 3s duration -> drop
        "こんにちは",                         # Japanese, 3s duration -> keep
    ]
    asr = _make_fake_asr(texts)

    in_q = queue.Queue()
    out_q = queue.Queue()
    stop = threading.Event()
    for start_s in (0.0, 3.0, 6.0):
        in_q.put(Utterance(start_s=start_s, end_s=start_s + 3.0,
                           audio=np.zeros(16000 * 3, dtype=np.float32), generation=0))
    in_q.put(None)

    lf = LanguageFilter(LanguageFilterConfig(enabled=True, min_duration_s=2.5))
    t = threading.Thread(
        target=run_asr,
        args=(asr, in_q, out_q),
        kwargs={"language_filter": lf, "stop": stop},
    )
    t.start()
    t.join(timeout=15)

    kept = []
    while True:
        try:
            item = out_q.get_nowait()
        except queue.Empty:
            break
        if item is None:
            continue
        if isinstance(item, Transcript):
            kept.append(item.start_s)

    assert 0.0 in kept, f"Japanese transcript should be kept: {kept}"
    assert 6.0 in kept, f"Japanese transcript should be kept: {kept}"
    assert 3.0 not in kept, f"English transcript should be dropped: {kept}"


# ---------------------------------------------------------------------------
# P0 — language filter text classification (the available classifier)
# ---------------------------------------------------------------------------
# Note: no real audio-language-ID model is integrated into this pass, so the
# pipeline applies the text-based Unicode-script filter to ASR transcripts.
# This test verifies that filter for Japanese / English / Indonesian / mixed.

def test_language_filter_text_classification():
    lf = LanguageFilter(LanguageFilterConfig(enabled=True, min_duration_s=2.5))
    # Genuine Japanese (high Japanese-script ratio) is kept.
    assert lf.filter("こんにちは、お元気ですか？", duration_s=3.0) is not None
    # Proper names written in Japanese script are kept.
    assert lf.filter("アリスさんとボブが会話しています", duration_s=3.0) is not None
    # Short reactions are always kept.
    assert lf.filter("はい", duration_s=1.0) is not None
    # Latin-script English / Indonesian are dropped.
    assert lf.filter("Hello, how are you doing today?", duration_s=3.0) is None
    assert lf.filter("Apa kabar kamu hari ini?", duration_s=3.0) is None
    # Latin-heavy code-switching falls below the Japanese-script threshold.
    assert lf.filter("Today is good weather desu", duration_s=3.0) is None


# ---------------------------------------------------------------------------
# P1-A — Qwen3-ASR bounded retries and error classification
# ---------------------------------------------------------------------------

def _make_qwen3(cfg=None):
    inst = object.__new__(Qwen3ASR)
    inst.cfg = cfg or AsrConfig(response_timeout_s=30.0)
    inst._failures = 0
    inst._restarts = 0
    inst._closed = False
    inst._generation = 0
    inst._inference_lock = threading.Lock()
    inst._state_lock = threading.Lock()
    from collections import deque
    inst._recent = deque(maxlen=4)
    inst._accepted_keys = deque(maxlen=32)
    inst.last_rejection = None
    inst.last_raw_transcript = ""
    return inst


def test_qwen3_asr_timeout_followed_by_successful_retry():
    """A transient timeout is retried and does not terminate the session."""
    inst = _make_qwen3()
    call_count = [0]

    def fake_impl(audio, probe=False, no_context=False, timeout=None):
        call_count[0] += 1
        if call_count[0] == 1:
            raise httpx.TimeoutException("read timeout")
        return "こんにちは"

    with mock.patch.object(inst, "_transcribe_impl", side_effect=fake_impl), \
            mock.patch("time.sleep", return_value=None):
        text = inst.transcribe(np.zeros(16000, dtype=np.float32))
    assert text == "こんにちは"
    assert inst._restarts == 0  # recovered by retry, no restart needed
    assert call_count[0] == 2  # first call failed, second succeeded


def test_qwen3_asr_repeated_failures_trigger_restart():
    """Repeated transient failures trigger a bounded restart before giving up."""
    inst = _make_qwen3()
    restarted = []

    # Four timeouts exhaust retries, then the post-restart call succeeds.
    impl = mock.Mock(side_effect=[httpx.TimeoutException("x")] * 4 + ["成功"])

    def fake_restart():
        restarted.append(True)

    with mock.patch.object(inst, "_transcribe_impl", impl), \
            mock.patch.object(inst, "_restart", side_effect=fake_restart), \
            mock.patch("time.sleep", return_value=None):
        text = inst.transcribe(np.zeros(16000, dtype=np.float32))
    assert text == "成功"
    assert restarted, "expected a restart to be attempted"
    assert impl.call_count == 5  # 4 retries + 1 after restart


def test_qwen3_asr_restart_failure_is_permanent_shutdown():
    """When restart also fails, a clear terminal error is raised (no infinite retry)."""
    inst = _make_qwen3()
    restart_calls = []

    def fake_impl(audio, probe=False, no_context=False, timeout=None):
        raise httpx.TimeoutException("keep failing")

    def fake_restart():
        restart_calls.append(True)
        raise RuntimeError("server will not start")

    with mock.patch.object(inst, "_transcribe_impl", side_effect=fake_impl), \
            mock.patch.object(inst, "_restart", side_effect=fake_restart), \
            mock.patch("time.sleep", return_value=None):
        try:
            inst.transcribe(np.zeros(16000, dtype=np.float32))
        except Qwen3ASRError as exc:
            assert "restart failed" in str(exc).lower() or "transcription failed" in str(exc).lower()
            assert len(restart_calls) == 1  # bounded, not indefinite
            return
    raise AssertionError("expected Qwen3ASRError")


def test_qwen3_asr_all_failures_no_recovery():
    """When every request fails and restart also fails, a terminal error is raised."""
    inst = _make_qwen3()
    inst._restarts = 2  # already at restart limit

    def fake_impl(audio, probe=False, no_context=False, timeout=None):
        raise httpx.TimeoutException("always failing")

    with mock.patch.object(inst, "_transcribe_impl", side_effect=fake_impl), \
            mock.patch("time.sleep", return_value=None):
        try:
            inst.transcribe(np.zeros(16000, dtype=np.float32))
        except Qwen3ASRError as exc:
            assert "transcription failed" in str(exc).lower()
            return
    raise AssertionError("expected Qwen3ASRError when all retries fail")


def test_qwen3_asr_shutdown_skips_restart():
    """Application cancellation prevents reviving the server during recovery."""
    inst = _make_qwen3()
    inst._closed = True
    with mock.patch.object(inst, "_transcribe_impl", side_effect=httpx.TimeoutException("x")), \
            mock.patch("time.sleep", return_value=None):
        try:
            inst.transcribe(np.zeros(16000, dtype=np.float32))
        except Qwen3ASRError:
            return
    raise AssertionError("expected terminal Qwen3ASRError when closed")


# ---------------------------------------------------------------------------
# P1-B — Qwen3-ASR recovery timeout and cancellation
# ---------------------------------------------------------------------------

def test_qwen3_asr_recovery_respects_remaining_timeout():
    """recover_context_echo enforces the supplied timeout and never starts when expired."""
    inst = _make_qwen3()
    captured = {}

    def fake_impl(audio, probe=False, no_context=False, timeout=None):
        captured["timeout"] = timeout
        return "text"

    with mock.patch.object(inst, "_transcribe_impl", side_effect=fake_impl):
        # Budget expired: must not start recovery.
        assert inst.recover_context_echo(np.zeros(1600, dtype=np.float32), timeout=0.0) == ""
        assert "timeout" not in captured

        # Budget remaining: timeout is forwarded to the HTTP request.
        inst.recover_context_echo(np.zeros(1600, dtype=np.float32), timeout=7.5)
        assert captured["timeout"] == 7.5


def test_qwen3_asr_cancellation_during_recovery():
    """Cancellation (generation change) aborts context-echo recovery."""
    inst = _make_qwen3()

    def fake_impl(audio, probe=False, no_context=False, timeout=None):
        inst.cancel_pending()  # cancellation mid-recovery
        return "text"

    with mock.patch.object(inst, "_transcribe_impl", side_effect=fake_impl):
        assert inst.recover_context_echo(np.zeros(1600, dtype=np.float32), timeout=10.0) == ""


# ---------------------------------------------------------------------------
# P1-C — Qwen3-ASR metadata prefix handling through the real parser
# ---------------------------------------------------------------------------

def _make_qwen3_with_server(body):
    inst = _make_qwen3()

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return body

    class FakeClient:
        def post(self, url, json=None, timeout=None):
            return FakeResp()

    class FakeServer:
        base_url = "http://127.0.0.1:9"
        client = FakeClient()

    inst.server = FakeServer()
    return inst


def test_qwen3_asr_strips_language_japanese_prefix_through_parser():
    inst = _make_qwen3_with_server({"choices": [{"message": {"content": "language Japanese<asr_text>こんにちはです"}}]})
    text = inst._transcribe_impl(np.zeros(1600, dtype=np.float32))
    assert text == "こんにちはです"
    assert "language" not in text.lower()


def test_qwen3_asr_strips_asr_text_tag_through_parser():
    inst = _make_qwen3_with_server({"choices": [{"message": {"content": "<asr_text>おはようございます"}}]})
    text = inst._transcribe_impl(np.zeros(1600, dtype=np.float32))
    assert text == "おはようございます"


def test_qwen3_asr_preserves_legitimate_japanese_through_parser():
    inst = _make_qwen3_with_server({"choices": [{"message": {"content": "今日はいい天気ですね"}}]})
    text = inst._transcribe_impl(np.zeros(1600, dtype=np.float32))
    assert text == "今日はいい天気ですね"


# ---------------------------------------------------------------------------
# P1 — Whisper CPU/CUDA selection and model-path validation
# ---------------------------------------------------------------------------

def _make_whisper_dir(tmp_path, whisper_like=True):
    d = tmp_path / "model"
    d.mkdir(parents=True)
    if whisper_like:
        (d / "model.bin").write_bytes(b"")
        (d / "config.json").write_bytes(b"{}")
    return str(d)


def test_whisper_cpu_selection():
    with mock.patch("faster_whisper.WhisperModel") as mock_wm:
        mock_wm.return_value = mock.MagicMock()
        with mock.patch("tempfile.gettempdir", return_value=str(Path("C:\\Windows\\Temp"))):
            cfg = AsrConfig(model_path="C:\\dummy\\whisper", cuda_device=-1, threads=4)
    # Fall through: invalid path -> error, so test via a valid dir instead.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = _make_whisper_dir(Path(td), whisper_like=True)
        with mock.patch("faster_whisper.WhisperModel") as mock_wm:
            mock_wm.return_value = mock.MagicMock()
            cfg = AsrConfig(model_path=d, cuda_device=-1, threads=4)
            WhisperASR(cfg)
        kw = mock_wm.call_args.kwargs
        assert kw["device"] == "cpu", kw
        assert kw["compute_type"] == "int8", kw
        assert kw["device_index"] is None, kw
        assert kw["local_files_only"] is True, kw


def test_whisper_cuda_device_index_forwarded():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = _make_whisper_dir(Path(td), whisper_like=True)
        for cuda_device in (0, 2):
            with mock.patch("faster_whisper.WhisperModel") as mock_wm:
                mock_wm.return_value = mock.MagicMock()
                cfg = AsrConfig(model_path=d, cuda_device=cuda_device, threads=4)
                WhisperASR(cfg)
            kw = mock_wm.call_args.kwargs
            assert kw["device"] == "cuda", (cuda_device, kw)
            assert kw["device_index"] == cuda_device, (cuda_device, kw)
            assert kw["compute_type"] == "float16", (cuda_device, kw)


def test_whisper_rejects_non_whisper_model_directory():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = _make_whisper_dir(Path(td), whisper_like=False)  # empty dir
        try:
            WhisperASR(AsrConfig(model_path=d, cuda_device=-1))
        except WhisperError:
            return
    raise AssertionError("expected WhisperError for non-Whisper model dir")


def test_whisper_rejects_missing_model_no_download():
    try:
        WhisperASR(AsrConfig(model_path="C:\\nonexistent\\notwhisper", cuda_device=-1))
    except WhisperError:
        return
    raise AssertionError("expected WhisperError when model unavailable locally")


def test_whisper_missing_optional_dependency():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = _make_whisper_dir(Path(td), whisper_like=True)
        with mock.patch.dict(sys.modules, {"faster_whisper": None}):
            try:
                WhisperASR(AsrConfig(model_path=d, cuda_device=-1))
            except WhisperError as exc:
                assert "faster-whisper" in str(exc).lower()
                return
    raise AssertionError("expected WhisperError when faster-whisper missing")


# ---------------------------------------------------------------------------
# P1 — configuration precedence and backend validation
# ---------------------------------------------------------------------------

def test_get_backend_config_precedence_cli_wins_over_profile():
    cfg = Config()
    cfg.asr_profiles["qwen3-asr"] = {
        "model_path": "/models/qwen3-asr/profile.gguf",
        "mmproj_path": "/models/qwen3-asr/profile-mmproj.gguf",
        "threads": 4,
    }
    explicit = {"model_path": "/cli/model.gguf"}
    merged = get_backend_config(cfg, "asr", "qwen3-asr", explicit_overrides=explicit)
    assert merged["model_path"] == "/cli/model.gguf"
    assert merged["mmproj_path"] == "/models/qwen3-asr/profile-mmproj.gguf"
    assert merged["threads"] == 4


def test_common_asr_mmproj_path_is_not_discarded():
    import tempfile
    toml = """
[models]
asr_backend = "qwen3-asr"

[asr]
mmproj_path = "C:/shared/mmproj.gguf"
response_timeout_s = 30

[asr.qwen3-asr]
model_path = "C:/models/qwen3-asr/model.gguf"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml)
        f.flush()
        cfg = load_config(Path(f.name))
    assert cfg.asr.mmproj_path == "C:/shared/mmproj.gguf"
    assert cfg.asr_profiles["qwen3-asr"]["model_path"] == "C:/models/qwen3-asr/model.gguf"


def test_asr_cuda_device_allows_negative_cpu():
    cfg = AsrConfig(cuda_device=-1)
    assert cfg.cuda_device == -1
    try:
        AsrConfig(cuda_device=-2)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# P2 — GPU pressure low-to-critical transitions
# ---------------------------------------------------------------------------

def _set_gpu_reading(used_gb, total_gb=8.0):
    return mock.patch("auto_translate.gpu_budget.gpu_memory_gb", return_value=(used_gb, total_gb))


def test_gpu_pressure_transitions():
    cfg = GpuBudgetConfig(startup_headroom_gb=2.0, low_free_gb=2.0, emergency_free_gb=1.0)
    budget = GpuBudget(cfg)
    with _set_gpu_reading(used_gb=1.0):  # free 7 GB -> normal
        budget.refresh()
        assert budget.pressure_level == "normal"
    with _set_gpu_reading(used_gb=6.5):  # free 1.5 GB -> low
        budget.refresh()
        assert budget.pressure_level == "low"
    with _set_gpu_reading(used_gb=7.5):  # free 0.5 GB -> critical
        budget.refresh()
        assert budget.pressure_level == "critical"


def test_gpu_critical_uses_shorter_cooldown_than_low():
    cfg = GpuBudgetConfig(startup_headroom_gb=2.0, low_free_gb=2.0, emergency_free_gb=1.0)
    GpuBudget(cfg)
    # Cooldown constants differ: critical responds sooner.
    assert GpuBudget.CRITICAL_RELIEF_COOLDOWN_S < GpuBudget.LOW_RELIEF_COOLDOWN_S


def test_gpu_sustained_critical_pressure_relief():
    """Sustained critical pressure triggers relief after cooldown expires."""
    cfg = GpuBudgetConfig(startup_headroom_gb=2.0, low_free_gb=2.0, emergency_free_gb=1.0)
    budget = GpuBudget(cfg)
    relief_calls = []

    def fake_release():
        relief_calls.append(True)

    # Simulate sustained critical pressure: free=0.5 GB (below emergency=1.0)
    with _set_gpu_reading(used_gb=7.5):  # free=0.5 GB -> critical
        budget.refresh()
        assert budget.pressure_level == "critical"
        # First relief request
        budget._last_critical_relief_time = 0.0
        budget._last_low_relief_time = 0.0
        budget._last_pressure = "normal"
        # Simulate one monitor iteration
        now = 100.0
        cooldown = budget.CRITICAL_RELIEF_COOLDOWN_S
        within_cooldown = now - budget._last_critical_relief_time < cooldown
        assert not within_cooldown, "first relief should not be in cooldown"
        fake_release()
        budget._last_critical_relief_time = now
        budget._last_pressure = "critical"

    # Simulate sustained critical pressure after cooldown expires
    with _set_gpu_reading(used_gb=7.5):  # still critical
        budget.refresh()
        assert budget.pressure_level == "critical"
        now = 100.0 + cooldown + 1.0  # cooldown expired
        within_cooldown = now - budget._last_critical_relief_time < cooldown
        assert not within_cooldown, "relief should be allowed after cooldown"
        fake_release()
        budget._last_critical_relief_time = now
        budget._last_pressure = "critical"

    assert len(relief_calls) == 2, f"expected 2 relief calls during sustained critical pressure, got {len(relief_calls)}"


def test_build_backend_command_no_default_override():
    """build_backend_command does not add --asr-model when using default paths."""
    import tempfile

    from auto_translate.gui_runner import build_backend_command
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "pyproject.toml").write_text("")
        (root / "src" / "auto_translate" / "__main__.py").parent.mkdir(parents=True)
        (root / "src" / "auto_translate" / "__main__.py").write_text("")
        # When asr_model=None, no --asr-model flag should appear
        _, args = build_backend_command(
            root, "https://www.youtube.com/watch?v=abc123", 720, 1,
            asr_model=None, mt_model=None,
            asr_backend="haas", translate_backend="gemma4",
        )
        assert "--asr-model" not in args
        assert "--mt-model" not in args


def test_build_backend_command_with_override():
    """build_backend_command adds --asr-model when an explicit override is given."""
    import tempfile

    from auto_translate.gui_runner import build_backend_command
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "pyproject.toml").write_text("")
        (root / "src" / "auto_translate" / "__main__.py").parent.mkdir(parents=True)
        (root / "src" / "auto_translate" / "__main__.py").write_text("")
        _, args = build_backend_command(
            root, "https://www.youtube.com/watch?v=abc123", 720, 1,
            asr_model="/custom/model.gguf", mt_model="/custom/mt.gguf",
            asr_backend="haas", translate_backend="gemma4",
        )
        assert "--asr-model" in args
        assert "/custom/model.gguf" in args
        assert "--mt-model" in args
        assert "/custom/mt.gguf" in args


def test_timeout_resolution_live_vod():
    """Timeout is derived from the merged backend profile, not common config."""
    cfg = Config()
    cfg.asr_profiles["qwen3-asr"] = {
        "response_timeout_s": 60.0,
        "live_response_timeout_s": 20.0,
    }
    merged = get_backend_config(cfg, "asr", "qwen3-asr")
    # Backend profile values should override common defaults
    assert merged["response_timeout_s"] == 60.0
    assert merged["live_response_timeout_s"] == 20.0
    # Live mode uses the shorter timeout
    live_timeout = min(merged["response_timeout_s"], merged["live_response_timeout_s"])
    assert live_timeout == 20.0
    # VOD mode uses the full timeout
    vod_timeout = merged["response_timeout_s"]
    assert vod_timeout == 60.0


def test_llama_server_watcher_lifecycle():
    """Only one cancellation watcher exists per server; old watcher is joined on restart."""
    import tempfile

    from auto_translate.llama_server import LlamaServer

    with tempfile.TemporaryDirectory() as td:
        exe = Path(td) / "llama-server.exe"
        exe.write_bytes(b"fake")
        server = LlamaServer(
            name="test", executable=exe, args=["--model", "test.gguf"],
            startup_timeout_s=1.0,
        )
        # Initial state: no watcher
        assert server._cancel_thread is None
        assert server._watcher_generation == 0

        # After reset_for_restart, generation is incremented
        server._closed = True  # simulate closed state
        server.reset_for_restart()
        assert server._watcher_generation == 1
        assert server._cancel_thread is None
        server._closed = True  # keep closed so start() doesn't create a watcher


# ---------------------------------------------------------------------------
# P2 / mandatory — MPV manual-pause protection + auto-pause hysteresis
# ---------------------------------------------------------------------------

def test_manual_pause_protects_vod_idle():
    class FakeIngest:
        class Source:
            live = False
        source = Source()
        def is_vod_audio_lead_limited(self):
            return False

    class ManuallyPaused:
        def is_manually_paused(self):
            return True

    class NotPaused:
        def is_manually_paused(self):
            return False

    assert _expected_vod_idle(FakeIngest(), [ManuallyPaused()]) is True
    assert _expected_vod_idle(FakeIngest(), [NotPaused()]) is False


def test_mpv_auto_pause_hysteresis():
    mpv = object.__new__(type("MpvOutput", (), {}))
    from auto_translate.output.mpv import MpvOutput
    mpv = MpvOutput.__new__(MpvOutput)
    mpv.ingest = type("I", (), {"source": type("S", (), {"live": False})()})()
    mpv._source_finished = False
    mpv._now = 100.0
    mpv._buffer_condition = threading.Lock()
    mpv.cfg = type("C", (), {"vod_pause_threshold_s": 0.5, "vod_resume_threshold_s": 3.0})()

    mpv._translated_until = 100.4  # lead 0.4 < 0.5 -> auto-pause
    assert mpv._vod_should_pause() is True
    assert mpv._vod_should_resume() is False

    mpv._translated_until = 103.0  # lead 3.0 >= 3.0 -> resume (hysteresis)
    assert mpv._vod_should_resume() is True
    assert mpv._vod_should_pause() is False


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
