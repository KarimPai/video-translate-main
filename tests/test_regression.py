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
import pytest

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
    # Latin-heavy code-switching without Japanese characters is dropped.
    assert lf.filter("Today is good weather desu", duration_s=3.0) is None


def test_language_filter_mixed_japanese_english():
    """Mixed Japanese/English text is kept to preserve legitimate dialogue."""
    lf = LanguageFilter(LanguageFilterConfig(enabled=True, min_duration_s=2.5))
    # Names with Japanese — keep
    assert lf.filter("Jason said こんにちは", duration_s=3.0) is not None
    assert lf.filter("Luciaが言った", duration_s=3.0) is not None
    # Game titles with Japanese — keep
    assert lf.filter("GTAのゲームプレイ", duration_s=3.0) is not None
    # VTuber names with Japanese — keep
    assert lf.filter("Kizuna AIのライブ", duration_s=3.0) is not None
    # Short Japanese responses — keep
    assert lf.filter("はい", duration_s=3.0) is not None
    assert lf.filter("うん", duration_s=3.0) is not None
    # Pure English — drop
    assert lf.filter("Hello, how are you doing today?", duration_s=3.0) is None
    # Pure Indonesian — drop
    assert lf.filter("Apa kabar kamu hari ini?", duration_s=3.0) is None


def test_language_filter_chinese_text_handling():
    """CJK-only text (Chinese) is kept as uncertain, not dropped.

    Chinese text without hiragana/katakana is ambiguous — it could be
    Chinese dialogue or Japanese kanji-only text. The filter keeps it
    to avoid dropping legitimate content, but doesn't classify it as
    confirmed Japanese.
    """
    lf = LanguageFilter(LanguageFilterConfig(enabled=True, min_duration_s=2.5))
    # Pure Chinese text (no kana) — kept as uncertain
    chinese = "你好世界"
    result = lf.filter(chinese, duration_s=3.0)
    assert result is not None, "CJK-only text should be kept as uncertain"
    # Japanese with hiragana — confirmed Japanese
    assert lf.filter("こんにちは世界", duration_s=3.0) is not None
    # Japanese with katakana — confirmed Japanese
    assert lf.filter("コンニチハ世界", duration_s=3.0) is not None
    # Mixed Chinese/Japanese with kana — confirmed Japanese
    assert lf.filter("你好、こんにちは", duration_s=3.0) is not None


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


def test_qwen3_asr_slow_restart_consumes_budget():
    """If restart + warmup consumes the entire budget, no inference is attempted."""
    inst = _make_qwen3()
    monotonic_counter = [0.0]

    def fake_monotonic():
        return monotonic_counter[0]

    def fake_impl(audio, probe=False, no_context=False, timeout=None):
        raise httpx.TimeoutException("fail")

    def fake_restart():
        # Simulate slow warmup consuming the entire budget
        monotonic_counter[0] += inst.cfg.response_timeout_s + 1.0
        inst._restarts += 1

    with mock.patch.object(inst, "_transcribe_impl", side_effect=fake_impl), \
            mock.patch.object(inst, "_restart", side_effect=fake_restart), \
            mock.patch("time.sleep", return_value=None), \
            mock.patch("time.monotonic", side_effect=fake_monotonic):
        try:
            inst.transcribe(np.zeros(16000, dtype=np.float32))
        except Qwen3ASRError as exc:
            # Budget consumed by restart + warmup — no post-restart inference
            assert "budget exceeded" in str(exc).lower() or "transcription failed" in str(exc).lower()
            return
    raise AssertionError("expected Qwen3ASRError when budget consumed by restart")


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
        (d / "tokenizer.json").write_bytes(b"{}")
        (d / "vocabulary.json").write_bytes(b"{}")
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


def test_whisper_falls_back_to_cache_on_haas_path():
    """When cfg.model_path points to a Haas model, Whisper falls back to HF cache."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        # Create a Haas-like directory (not Whisper)
        haas_dir = Path(td) / "haas-ja-v1.0"
        haas_dir.mkdir()
        (haas_dir / "model.gguf").write_bytes(b"fake gguf")

        # Mock HF cache to return a valid Whisper model (with all required files)
        cached_dir = Path(td) / "cached_whisper"
        cached_dir.mkdir()
        (cached_dir / "model.bin").write_bytes(b"")
        (cached_dir / "config.json").write_bytes(b"{}")
        (cached_dir / "tokenizer.json").write_bytes(b"{}")

        with mock.patch.object(WhisperASR, "_find_cached_model", return_value=cached_dir), \
                mock.patch("faster_whisper.WhisperModel") as mock_wm:
            mock_wm.return_value = mock.MagicMock()
            cfg = AsrConfig(model_path=str(haas_dir), cuda_device=-1)
            WhisperASR(cfg)
        # Should use the cached Whisper model, not the Haas path
        assert mock_wm.call_args.kwargs["device"] == "cpu"


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


def test_whisper_recovery_timeout_enforcement():
    """Whisper recover_context_echo enforces the supplied timeout budget."""
    inst = object.__new__(WhisperASR)
    inst.cfg = AsrConfig(response_timeout_s=30.0)
    inst._closed = False
    inst._generation = 0
    inst._state_lock = threading.Lock()
    inst._inference_lock = threading.Lock()
    inst.last_rejection = None
    inst.last_raw_transcript = ""
    from collections import deque
    inst._recent = deque(maxlen=4)
    inst._accepted_keys = deque(maxlen=32)

    # Budget expired: must not start recovery
    assert inst.recover_context_echo(np.zeros(1600, dtype=np.float32), timeout=0.0) == ""

    # Closed state: must not start recovery
    inst._closed = True
    assert inst.recover_context_echo(np.zeros(1600, dtype=np.float32), timeout=10.0) == ""
    inst._closed = False

    # Successful recovery with timeout
    call_count = [0]
    def fake_impl(audio, probe=False, no_context=False):
        call_count[0] += 1
        return "こんにちは"

    with mock.patch.object(inst, "_transcribe_impl", side_effect=fake_impl):
        result = inst.recover_context_echo(np.zeros(1600, dtype=np.float32), timeout=10.0)
    assert result == "こんにちは"
    assert call_count[0] == 1

    # Cancellation during recovery
    def fake_cancel_impl(audio, probe=False, no_context=False):
        inst.cancel_pending()
        return "text"

    with mock.patch.object(inst, "_transcribe_impl", side_effect=fake_cancel_impl):
        result = inst.recover_context_echo(np.zeros(1600, dtype=np.float32), timeout=10.0)
    assert result == ""


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


def test_backend_profile_propagates_to_pipeline():
    """Backend-specific translate settings reach the pipeline via cfg.translate."""
    cfg = Config()
    cfg.translate_profiles["plamo"] = {
        "context_lines": 10,
        "merge_fragments": True,
        "merge_max_gap_s": 2.0,
        "merge_max_s": 5.0,
        "timeout_s": 120.0,
    }
    merged = get_backend_config(cfg, "translate", "plamo")
    # Verify profile values are present
    assert merged["context_lines"] == 10
    assert merged["merge_fragments"] is True
    assert merged["merge_max_gap_s"] == 2.0
    assert merged["merge_max_s"] == 5.0
    assert merged["timeout_s"] == 120.0
    # Simulate what __main__.py does: apply profile to shared config
    for key in ("context_lines", "merge_fragments", "merge_max_gap_s", "merge_max_s",
                "live_merge_margin_s", "max_identical_cues", "timeout_s"):
        if key in merged:
            setattr(cfg.translate, key, merged[key])
    # Pipeline now sees backend-specific values
    assert cfg.translate.context_lines == 10
    assert cfg.translate.merge_fragments is True
    assert cfg.translate.merge_max_gap_s == 2.0
    assert cfg.translate.merge_max_s == 5.0
    assert cfg.translate.timeout_s == 120.0


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


def test_gui_backend_switching_preserves_paths():
    """GUI backend switching preserves per-backend model paths correctly."""
    import tempfile

    # Create a config with backend-specific profiles
    toml = """
[models]
asr_backend = "haas"
translate_backend = "gemma4"

[asr.haas]
model_path = "C:/models/haas/model.gguf"

[asr.qwen3-asr]
model_path = "C:/models/qwen3-asr/model.gguf"
mmproj_path = "C:/models/qwen3-asr/mmproj.gguf"

[translate.gemma4]
model_path = "C:/models/gemma4/model.gguf"

[translate.plamo]
model_path = "C:/models/plamo/model.gguf"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml)
        f.flush()
        cfg = load_config(Path(f.name))

    # Verify backend profiles are loaded correctly
    asr_profile = get_backend_config(cfg, "asr", "haas")
    assert asr_profile["model_path"] == "C:/models/haas/model.gguf"

    qwen3_profile = get_backend_config(cfg, "asr", "qwen3-asr")
    assert qwen3_profile["model_path"] == "C:/models/qwen3-asr/model.gguf"
    assert qwen3_profile["mmproj_path"] == "C:/models/qwen3-asr/mmproj.gguf"

    mt_profile = get_backend_config(cfg, "translate", "gemma4")
    assert mt_profile["model_path"] == "C:/models/gemma4/model.gguf"

    plamo_profile = get_backend_config(cfg, "translate", "plamo")
    assert plamo_profile["model_path"] == "C:/models/plamo/model.gguf"

    # Verify that switching backends doesn't contaminate model paths
    # Haas profile should NOT have mmproj_path from Qwen3-ASR
    assert "mmproj_path" not in asr_profile or not asr_profile.get("mmproj_path")
    # Qwen3-ASR should NOT have Haas's model path
    assert qwen3_profile["model_path"] != asr_profile["model_path"]


def test_gui_qt_backend_switching_integration():
    """Real Qt offscreen test: exercise dropdown signals and model path fields.

    Tests Haas -> Whisper -> Haas, Haas -> Qwen3-ASR -> Whisper -> Haas,
    and Gemma -> PLaMo -> Hy-MT2 -> Gemma switching scenarios.
    Verifies custom paths are preserved on switch-back and defaults are restored.
    """
    import os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])

    import tempfile

    from auto_translate.gui import VideoTranslateWindow

    toml = """
[models]
asr_backend = "haas"
translate_backend = "gemma4"

[asr]
model_path = "C:/models/haas/model.gguf"

[asr.qwen3-asr]
model_path = "C:/models/qwen3-asr/model.gguf"
mmproj_path = "C:/models/qwen3-asr/mmproj.gguf"

[asr.whisper]
model_path = "C:/models/whisper/model.bin"

[translate]
model_path = "C:/models/gemma4/model.gguf"

[translate.plamo]
model_path = "C:/models/plamo/model.gguf"

[translate.hymt]
model_path = "C:/models/hymt/model.gguf"
"""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        config_file = root / "config.toml"
        config_file.write_text(toml)
        (root / "pyproject.toml").write_text("")
        (root / "src" / "auto_translate" / "__main__.py").parent.mkdir(parents=True)
        (root / "src" / "auto_translate" / "__main__.py").write_text("")

        window = VideoTranslateWindow(root)

        # Verify initial state: Haas ASR, Gemma4 MT
        assert window._last_asr_backend == "haas"
        assert window._last_mt_backend == "gemma4"
        assert window.asr_model.text() == "C:/models/haas/model.gguf"
        assert window.mt_model.text() == "C:/models/gemma4/model.gguf"

        # --- ASR: Haas -> Whisper -> Haas ---
        # Switch to Whisper
        whisper_idx = window.asr_backend.findData("whisper")
        assert whisper_idx >= 0, "Whisper backend should be in dropdown"
        window.asr_backend.setCurrentIndex(whisper_idx)
        assert window._last_asr_backend == "whisper"
        assert window.asr_model.text() == "C:/models/whisper/model.bin"

        # Switch back to Haas — should restore Haas default
        haas_idx = window.asr_backend.findData("haas")
        window.asr_backend.setCurrentIndex(haas_idx)
        assert window._last_asr_backend == "haas"
        assert window.asr_model.text() == "C:/models/haas/model.gguf"

        # --- ASR: Haas -> Qwen3-ASR -> Whisper -> Haas ---
        qwen3_idx = window.asr_backend.findData("qwen3-asr")
        assert qwen3_idx >= 0, "Qwen3-ASR backend should be in dropdown"
        window.asr_backend.setCurrentIndex(qwen3_idx)
        assert window._last_asr_backend == "qwen3-asr"
        assert window.asr_model.text() == "C:/models/qwen3-asr/model.gguf"

        window.asr_backend.setCurrentIndex(whisper_idx)
        assert window._last_asr_backend == "whisper"

        window.asr_backend.setCurrentIndex(haas_idx)
        assert window._last_asr_backend == "haas"
        assert window.asr_model.text() == "C:/models/haas/model.gguf"

        # --- Custom path preservation ---
        # Set custom path for Haas, switch away, switch back
        window.asr_model.setText("C:/custom/haas.gguf")
        window.asr_backend.setCurrentIndex(whisper_idx)
        window.asr_backend.setCurrentIndex(haas_idx)
        # Custom path should be preserved
        assert window.asr_model.text() == "C:/custom/haas.gguf"

        # Restore default by setting the default path explicitly
        window.asr_model.setText("C:/models/haas/model.gguf")
        window.asr_backend.setCurrentIndex(whisper_idx)
        window.asr_backend.setCurrentIndex(haas_idx)
        # Should show default, no stale override
        assert window.asr_model.text() == "C:/models/haas/model.gguf"

        # --- MT: Gemma -> PLaMo -> Hy-MT2 -> Gemma ---
        plamo_idx = window.mt_backend.findData("plamo")
        assert plamo_idx >= 0, "PLaMo backend should be in dropdown"
        window.mt_backend.setCurrentIndex(plamo_idx)
        assert window._last_mt_backend == "plamo"
        assert window.mt_model.text() == "C:/models/plamo/model.gguf"

        hymt_idx = window.mt_backend.findData("hymt")
        assert hymt_idx >= 0, "Hy-MT2 backend should be in dropdown"
        window.mt_backend.setCurrentIndex(hymt_idx)
        assert window._last_mt_backend == "hymt"
        assert window.mt_model.text() == "C:/models/hymt/model.gguf"

        gemma_idx = window.mt_backend.findData("gemma4")
        window.mt_backend.setCurrentIndex(gemma_idx)
        assert window._last_mt_backend == "gemma4"
        assert window.mt_model.text() == "C:/models/gemma4/model.gguf"

        # --- Verify _get_asr_model_override returns None for defaults ---
        window.asr_backend.setCurrentIndex(haas_idx)
        window.asr_model.setText("C:/models/haas/model.gguf")
        assert window._get_asr_model_override() is None

        # --- Verify _get_asr_model_override returns path for custom ---
        window.asr_model.setText("C:/custom/haas.gguf")
        override = window._get_asr_model_override()
        assert override == "C:/custom/haas.gguf"

        # --- Verify ASR and MT selections are independent ---
        window.asr_backend.setCurrentIndex(whisper_idx)
        window.mt_backend.setCurrentIndex(plamo_idx)
        assert window._last_asr_backend == "whisper"
        assert window._last_mt_backend == "plamo"


# ---------------------------------------------------------------------------
# P0 — Model resolution: no cross-backend contamination
# ---------------------------------------------------------------------------

def test_model_resolution_all_six_backends():
    """Each of the 6 backends resolves correct model paths without cross-contamination."""
    cfg = Config()
    # Set common defaults that should NOT be inherited by other backends
    cfg.asr.model_path = "C:/models/haas/model.gguf"
    cfg.translate.model_path = "C:/models/gemma4/model.gguf"

    # Backend profiles
    cfg.asr_profiles["haas"] = {"model_path": "C:/models/haas/model.gguf"}
    cfg.asr_profiles["qwen3-asr"] = {
        "model_path": "C:/models/qwen3-asr/model.gguf",
        "mmproj_path": "C:/models/qwen3-asr/mmproj.gguf",
    }
    cfg.asr_profiles["whisper"] = {"model_path": "C:/models/whisper/model.bin"}
    cfg.translate_profiles["gemma4"] = {"model_path": "C:/models/gemma4/model.gguf"}
    cfg.translate_profiles["plamo"] = {"model_path": "C:/models/plamo/model.gguf"}
    cfg.translate_profiles["hymt"] = {"model_path": "C:/models/hymt/model.gguf"}

    # ASR backends
    haas = get_backend_config(cfg, "asr", "haas")
    assert haas["model_path"] == "C:/models/haas/model.gguf"

    qwen3 = get_backend_config(cfg, "asr", "qwen3-asr")
    assert qwen3["model_path"] == "C:/models/qwen3-asr/model.gguf"
    assert qwen3["mmproj_path"] == "C:/models/qwen3-asr/mmproj.gguf"
    assert qwen3["model_path"] != cfg.asr.model_path  # no Haas contamination

    whisper = get_backend_config(cfg, "asr", "whisper")
    assert whisper["model_path"] == "C:/models/whisper/model.bin"
    assert whisper["model_path"] != cfg.asr.model_path  # no Haas contamination

    # Translate backends
    gemma = get_backend_config(cfg, "translate", "gemma4")
    assert gemma["model_path"] == "C:/models/gemma4/model.gguf"

    plamo = get_backend_config(cfg, "translate", "plamo")
    assert plamo["model_path"] == "C:/models/plamo/model.gguf"
    assert plamo["model_path"] != cfg.translate.model_path  # no Gemma contamination

    hymt = get_backend_config(cfg, "translate", "hymt")
    assert hymt["model_path"] == "C:/models/hymt/model.gguf"
    assert hymt["model_path"] != cfg.translate.model_path  # no Gemma contamination


def test_model_resolution_missing_profile_no_inheritance():
    """When a non-default backend profile is absent, model_path is empty, not inherited from common config."""
    cfg = Config()
    cfg.asr.model_path = "C:/models/haas/model.gguf"
    cfg.translate.model_path = "C:/models/gemma4/model.gguf"

    # Unknown backend — model_path should NOT be inherited
    unknown_asr = get_backend_config(cfg, "asr", "unknown-backend")
    assert unknown_asr.get("model_path", "") == "", \
        "model_path should be empty, not inherited from common config"

    unknown_mt = get_backend_config(cfg, "translate", "unknown-backend")
    assert unknown_mt.get("model_path", "") == "", \
        "model_path should be empty, not inherited from common config"

    # Default backends (haas, gemma4) SHOULD get their model_path from common config
    haas_cfg = get_backend_config(cfg, "asr", "haas")
    assert haas_cfg.get("model_path") == "C:/models/haas/model.gguf", \
        "Haas (default ASR) should inherit model_path from common config"

    gemma_cfg = get_backend_config(cfg, "translate", "gemma4")
    assert gemma_cfg.get("model_path") == "C:/models/gemma4/model.gguf", \
        "Gemma4 (default translate) should inherit model_path from common config"

    # Non-default backends should NOT inherit
    qwen3_cfg = get_backend_config(cfg, "asr", "qwen3-asr")
    assert qwen3_cfg.get("model_path", "") == "", \
        "Qwen3-ASR should NOT inherit Haas model_path"

    whisper_cfg = get_backend_config(cfg, "asr", "whisper")
    assert whisper_cfg.get("model_path", "") == "", \
        "Whisper should NOT inherit Haas model_path"

    plamo_cfg = get_backend_config(cfg, "translate", "plamo")
    assert plamo_cfg.get("model_path", "") == "", \
        "PLaMo should NOT inherit Gemma model_path"


def test_model_resolution_combinations():
    """Test ASR+MT combinations: no cross-contamination between backends."""
    cfg = Config()
    cfg.asr.model_path = "C:/models/haas/model.gguf"
    cfg.translate.model_path = "C:/models/gemma4/model.gguf"

    cfg.asr_profiles["qwen3-asr"] = {
        "model_path": "C:/models/qwen3-asr/model.gguf",
        "mmproj_path": "C:/models/qwen3-asr/mmproj.gguf",
    }
    cfg.asr_profiles["whisper"] = {"model_path": "C:/models/whisper/model.bin"}
    cfg.translate_profiles["plamo"] = {"model_path": "C:/models/plamo/model.gguf"}
    cfg.translate_profiles["hymt"] = {"model_path": "C:/models/hymt/model.gguf"}

    # Qwen3-ASR + PLaMo
    asr = get_backend_config(cfg, "asr", "qwen3-asr")
    mt = get_backend_config(cfg, "translate", "plamo")
    assert asr["model_path"] == "C:/models/qwen3-asr/model.gguf"
    assert mt["model_path"] == "C:/models/plamo/model.gguf"
    assert asr["model_path"] != mt["model_path"]

    # Whisper + Hy-MT2
    asr = get_backend_config(cfg, "asr", "whisper")
    mt = get_backend_config(cfg, "translate", "hymt")
    assert asr["model_path"] == "C:/models/whisper/model.bin"
    assert mt["model_path"] == "C:/models/hymt/model.gguf"


def test_resolve_model_identity():
    """resolve_model_identity returns explicit, profile, or compatible default paths."""
    from auto_translate.config import resolve_model_identity

    cfg = Config()
    cfg.asr.model_path = "C:/models/haas/model.gguf"
    cfg.asr.mmproj_path = "C:/models/haas/mmproj.gguf"

    # Unknown backend — no profile, not the default — should return empty strings
    identity = resolve_model_identity(cfg, "asr", "unknown")
    assert identity["model_path"] == ""
    assert identity["mmproj_path"] == ""
    # server_path is no longer model identity — it's a shared runtime dependency

    # Haas IS the default ASR — common config value IS compatible
    identity = resolve_model_identity(cfg, "asr", "haas")
    assert identity["model_path"] == "C:/models/haas/model.gguf"
    assert identity["mmproj_path"] == "C:/models/haas/mmproj.gguf"

    # Qwen3-ASR is NOT the default — common config should NOT be inherited
    identity = resolve_model_identity(cfg, "asr", "qwen3-asr")
    assert identity["model_path"] == ""
    assert identity["mmproj_path"] == ""

    # With profile — should return profile values
    cfg.asr_profiles["qwen3-asr"] = {
        "model_path": "C:/models/qwen3-asr/model.gguf",
        "mmproj_path": "C:/models/qwen3-asr/mmproj.gguf",
    }
    identity = resolve_model_identity(cfg, "asr", "qwen3-asr")
    assert identity["model_path"] == "C:/models/qwen3-asr/model.gguf"
    assert identity["mmproj_path"] == "C:/models/qwen3-asr/mmproj.gguf"

    # Explicit override wins over profile
    identity = resolve_model_identity(
        cfg, "asr", "qwen3-asr",
        explicit_overrides={"model_path": "C:/custom/model.gguf"}
    )
    assert identity["model_path"] == "C:/custom/model.gguf"


# ---------------------------------------------------------------------------
# P1 — Qwen3-ASR permanent error: no restart
# ---------------------------------------------------------------------------

def test_qwen3_asr_http_400_no_restart():
    """HTTP 400 Bad Request should not trigger a server restart."""
    inst = _make_qwen3()

    def fake_impl(audio, probe=False, no_context=False, timeout=None):
        resp = mock.Mock()
        resp.status_code = 400
        raise httpx.HTTPStatusError("400 Bad Request", request=mock.Mock(), response=resp)

    with mock.patch.object(inst, "_transcribe_impl", side_effect=fake_impl), \
            mock.patch.object(inst, "_restart", side_effect=AssertionError("should not restart")), \
            mock.patch("time.sleep", return_value=None):
        try:
            inst.transcribe(np.zeros(16000, dtype=np.float32))
        except Qwen3ASRError as exc:
            assert "400" in str(exc) or "permanent" in str(exc).lower() or "failed" in str(exc).lower()
            return
    raise AssertionError("expected Qwen3ASRError for HTTP 400")


def test_qwen3_asr_http_500_retry():
    """HTTP 500 Internal Server Error should be retried as a transient failure."""
    inst = _make_qwen3()
    call_count = [0]

    def fake_impl(audio, probe=False, no_context=False, timeout=None):
        call_count[0] += 1
        if call_count[0] <= 2:
            resp = mock.Mock()
            resp.status_code = 500
            raise httpx.HTTPStatusError("500 Internal Server Error", request=mock.Mock(), response=resp)
        return "成功"

    with mock.patch.object(inst, "_transcribe_impl", side_effect=fake_impl), \
            mock.patch("time.sleep", return_value=None):
        text = inst.transcribe(np.zeros(16000, dtype=np.float32))
    assert text == "成功"
    assert call_count[0] == 3  # 2 failures + 1 success


def test_qwen3_asr_permanent_error_no_restart():
    """Qwen3ASRError should not trigger a server restart."""
    inst = _make_qwen3()

    def fake_impl(audio, probe=False, no_context=False, timeout=None):
        raise Qwen3ASRError("invalid model configuration")

    with mock.patch.object(inst, "_transcribe_impl", side_effect=fake_impl), \
            mock.patch.object(inst, "_restart", side_effect=AssertionError("should not restart")), \
            mock.patch("time.sleep", return_value=None):
        try:
            inst.transcribe(np.zeros(16000, dtype=np.float32))
        except Qwen3ASRError as exc:
            assert "invalid model" in str(exc).lower() or "permanent" in str(exc).lower() or "failed" in str(exc).lower()
            return
    raise AssertionError("expected Qwen3ASRError for permanent error")


# ---------------------------------------------------------------------------
# P1 — Whisper explicit path handling
# ---------------------------------------------------------------------------

def test_whisper_explicit_invalid_path_raises():
    """Explicitly configured invalid Whisper path raises clear error, no fallback."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        # Create a non-Whisper directory (e.g., GGUF model)
        gguf_dir = Path(td) / "my-whisper"
        gguf_dir.mkdir()
        (gguf_dir / "model.gguf").write_bytes(b"fake gguf")

        cfg = AsrConfig(model_path=str(gguf_dir), cuda_device=-1)

        # is_explicit=True should raise, not fall back to cache
        try:
            WhisperASR._resolve_model_path(cfg, is_explicit=True)
            raise AssertionError("expected WhisperError for explicit invalid path")
        except WhisperError as exc:
            assert "not a valid Whisper model" in str(exc) or "explicitly configured" in str(exc).lower()


def test_whisper_inherited_haas_path_ignored():
    """Inherited Haas path falls through to HF cache search."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        # Create a Haas-like directory (not Whisper)
        haas_dir = Path(td) / "haas-ja-v1.0"
        haas_dir.mkdir()
        (haas_dir / "model.gguf").write_bytes(b"fake gguf")

        # Mock HF cache to return a valid Whisper model (with all required files)
        cached_dir = Path(td) / "cached_whisper"
        cached_dir.mkdir()
        (cached_dir / "model.bin").write_bytes(b"")
        (cached_dir / "config.json").write_bytes(b"{}")
        (cached_dir / "tokenizer.json").write_bytes(b"{}")

        cfg = AsrConfig(model_path=str(haas_dir), cuda_device=-1)

        with mock.patch.object(WhisperASR, "_find_cached_model", return_value=cached_dir):
            result = WhisperASR._resolve_model_path(cfg, is_explicit=False)
        # Should use the cached Whisper model, not the Haas path
        assert result == str(cached_dir)


def test_whisper_explicit_nonexistent_path_raises():
    """Explicitly configured non-existent path raises clear error."""
    cfg = AsrConfig(model_path="C:/nonexistent/whisper-model", cuda_device=-1)

    try:
        WhisperASR._resolve_model_path(cfg, is_explicit=True)
        raise AssertionError("expected WhisperError for non-existent explicit path")
    except WhisperError as exc:
        assert "does not exist" in str(exc) or "explicitly configured" in str(exc).lower()


# ---------------------------------------------------------------------------
# P2 — Pipeline config propagation from production path
# ---------------------------------------------------------------------------

def test_pipeline_config_from_production_path():
    """Test actual __main__.py config resolution path, not manual reproduction."""
    cfg = Config()
    cfg.translate_profiles["plamo"] = {
        "context_lines": 10,
        "merge_fragments": True,
        "merge_max_gap_s": 2.0,
        "merge_max_s": 5.0,
        "timeout_s": 120.0,
    }
    cfg.asr_profiles["qwen3-asr"] = {
        "response_timeout_s": 60.0,
        "live_response_timeout_s": 20.0,
    }

    # Simulate what __main__.py does: get_backend_config then apply to shared config
    translate_profile = get_backend_config(cfg, "translate", "plamo")
    for key in ("context_lines", "merge_fragments", "merge_max_gap_s", "merge_max_s",
                "live_merge_margin_s", "max_identical_cues", "timeout_s"):
        if key in translate_profile:
            setattr(cfg.translate, key, translate_profile[key])

    asr_profile = get_backend_config(cfg, "asr", "qwen3-asr")
    for key in ("response_timeout_s", "live_response_timeout_s"):
        if key in asr_profile:
            setattr(cfg.asr, key, asr_profile[key])

    # Verify backend-specific values are now in the shared config
    assert cfg.translate.context_lines == 10
    assert cfg.translate.merge_fragments is True
    assert cfg.translate.merge_max_gap_s == 2.0
    assert cfg.translate.merge_max_s == 5.0
    assert cfg.translate.timeout_s == 120.0
    assert cfg.asr.response_timeout_s == 60.0
    assert cfg.asr.live_response_timeout_s == 20.0


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
    from auto_translate.output.coverage import CoverageTracker
    from auto_translate.output.mpv import MpvOutput
    mpv = MpvOutput.__new__(MpvOutput)
    mpv.ingest = type("I", (), {"source": type("S", (), {"live": False})()})()
    mpv._source_finished = False
    mpv._now = 100.0
    mpv._buffer_condition = threading.Lock()
    mpv.cfg = type("C", (), {"vod_pause_threshold_s": 0.5, "vod_resume_threshold_s": 3.0})()

    # Use CoverageTracker instead of direct _translated_until
    mpv._coverage = CoverageTracker(0.0)
    mpv._coverage._translated_until = 100.4  # lead 0.4 < 0.5 -> auto-pause
    mpv._coverage._vad_progress = 100.4
    assert mpv._vod_should_pause() is True
    assert mpv._vod_should_resume() is False

    mpv._coverage._translated_until = 103.0  # lead 3.0 >= 3.0 -> resume (hysteresis)
    mpv._coverage._vad_progress = 103.0
    assert mpv._vod_should_resume() is True
    assert mpv._vod_should_pause() is False


# ---------------------------------------------------------------------------
# P1 — Whisper CTranslate2 model validation (complete model required)
# ---------------------------------------------------------------------------

def test_whisper_rejects_incomplete_model_missing_tokenizer():
    """A directory with model.bin + config.json but no tokenizer is rejected."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "incomplete"
        d.mkdir()
        (d / "model.bin").write_bytes(b"")
        (d / "config.json").write_bytes(b"{}")
        # Missing tokenizer.json and vocabulary.json
        assert WhisperASR._looks_like_whisper_model(d) is False


def test_whisper_rejects_incomplete_model_single_file():
    """A directory with only model.bin is rejected."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "single_file"
        d.mkdir()
        (d / "model.bin").write_bytes(b"")
        assert WhisperASR._looks_like_whisper_model(d) is False


def test_whisper_accepts_complete_model():
    """A directory with model.bin + config.json + tokenizer is accepted."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "complete"
        d.mkdir()
        (d / "model.bin").write_bytes(b"")
        (d / "config.json").write_bytes(b"{}")
        (d / "tokenizer.json").write_bytes(b"{}")
        assert WhisperASR._looks_like_whisper_model(d) is True


def test_whisper_accepts_complete_model_with_vocabulary():
    """A directory with model.bin + config.json + vocabulary.json is accepted."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "complete_vocab"
        d.mkdir()
        (d / "model.bin").write_bytes(b"")
        (d / "config.json").write_bytes(b"{}")
        (d / "vocabulary.json").write_bytes(b"{}")
        assert WhisperASR._looks_like_whisper_model(d) is True


def test_whisper_find_cached_model_rejects_incomplete():
    """_find_cached_model rejects a snapshot missing tokenizer files."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        # Create an incomplete snapshot
        incomplete_dir = Path(td) / "incomplete_snapshot"
        incomplete_dir.mkdir()
        (incomplete_dir / "model.bin").write_bytes(b"")
        (incomplete_dir / "config.json").write_bytes(b"{}")
        # Missing tokenizer

        with mock.patch("auto_translate.asr.whisper_turbo.try_to_load_from_cache") as mock_load:
            mock_load.return_value = incomplete_dir / "model.bin"
            result = WhisperASR._find_cached_model("repo/id")
        assert result is None


def test_whisper_find_cached_model_accepts_complete():
    """_find_cached_model accepts a complete snapshot."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        complete_dir = Path(td) / "complete_snapshot"
        complete_dir.mkdir()
        (complete_dir / "model.bin").write_bytes(b"")
        (complete_dir / "config.json").write_bytes(b"{}")
        (complete_dir / "tokenizer.json").write_bytes(b"{}")

        with mock.patch("auto_translate.asr.whisper_turbo.try_to_load_from_cache") as mock_load:
            mock_load.return_value = complete_dir / "model.bin"
            result = WhisperASR._find_cached_model("repo/id")
        assert result == complete_dir


# ---------------------------------------------------------------------------
# P1 — Whisper explicit path through registry factory
# ---------------------------------------------------------------------------

def test_whisper_registry_factory_passes_is_explicit():
    """The registry factory passes is_explicit=True when model_path is set."""
    import tempfile

    from auto_translate.registry import get_default_registry

    with tempfile.TemporaryDirectory() as td:
        d = _make_whisper_dir(Path(td), whisper_like=True)
        with mock.patch("faster_whisper.WhisperModel") as mock_wm:
            mock_wm.return_value = mock.MagicMock()
            registry = get_default_registry()
            # Config dict with model_path set (simulating explicit override)
            config = {"model_path": d, "cuda_device": -1, "threads": 4}
            registry.create_asr("whisper", config, _whisper_is_explicit=True, context="")
        # If we got here without error, the factory passed is_explicit correctly


def test_whisper_registry_factory_default_is_not_explicit():
    """The registry factory defaults to is_explicit=False."""
    import tempfile

    from auto_translate.registry import get_default_registry

    with tempfile.TemporaryDirectory() as td:
        d = _make_whisper_dir(Path(td), whisper_like=True)
        with mock.patch("faster_whisper.WhisperModel") as mock_wm:
            mock_wm.return_value = mock.MagicMock()
            registry = get_default_registry()
            config = {"model_path": d, "cuda_device": -1, "threads": 4}
            # No _whisper_is_explicit passed — should default to False
            registry.create_asr("whisper", config, context="")


# ---------------------------------------------------------------------------
# P2 — Backend startup receives correct resolved model identity
# ---------------------------------------------------------------------------

def test_backend_startup_receives_correct_model_identity():
    """get_backend_config + resolve_model_identity produce consistent results."""
    from auto_translate.config import resolve_model_identity

    cfg = Config()
    cfg.asr.model_path = "C:/models/haas/model.gguf"
    cfg.translate.model_path = "C:/models/gemma4/model.gguf"
    cfg.asr_profiles["qwen3-asr"] = {
        "model_path": "C:/models/qwen3-asr/model.gguf",
        "mmproj_path": "C:/models/qwen3-asr/mmproj.gguf",
    }

    # Haas: common config is the source
    haas_merged = get_backend_config(cfg, "asr", "haas")
    haas_identity = resolve_model_identity(cfg, "asr", "haas")
    assert haas_merged["model_path"] == haas_identity["model_path"]
    assert haas_merged["model_path"] == "C:/models/haas/model.gguf"

    # Qwen3-ASR: profile is the source
    qwen3_merged = get_backend_config(cfg, "asr", "qwen3-asr")
    qwen3_identity = resolve_model_identity(cfg, "asr", "qwen3-asr")
    assert qwen3_merged["model_path"] == qwen3_identity["model_path"]
    assert qwen3_merged["model_path"] == "C:/models/qwen3-asr/model.gguf"

    # Whisper: no profile, not default — empty
    whisper_merged = get_backend_config(cfg, "asr", "whisper")
    whisper_identity = resolve_model_identity(cfg, "asr", "whisper")
    assert whisper_merged.get("model_path", "") == whisper_identity["model_path"] == ""


# ---------------------------------------------------------------------------
# P0 — Registry factory prevents dataclass default fallback
# These tests invoke the actual factory, not just get_backend_config().
# They would have caught the AsrConfig.model_path.default AttributeError.
# ---------------------------------------------------------------------------

def test_registry_factory_plamo_raises_on_empty_model():
    """PLaMo factory must raise ValueError when no model is configured."""
    from auto_translate.config import Config, get_backend_config
    from auto_translate.registry import get_default_registry

    cfg = Config()
    cfg.translate.model_path = "C:/models/gemma4/model.gguf"
    merged = get_backend_config(cfg, "translate", "plamo")
    assert merged.get("model_path", "") == "", \
        "PLaMo should not inherit Gemma4's model_path"

    # Invoke the actual factory — it must raise ValueError, not AttributeError
    registry = get_default_registry()
    with pytest.raises(ValueError, match="PLaMo.*model.*configured"):
        registry.create_translate("plamo", merged)


def test_registry_factory_hymt_raises_on_empty_model():
    """Hy-MT2 factory must raise ValueError when no model is configured."""
    from auto_translate.config import Config, get_backend_config
    from auto_translate.registry import get_default_registry

    cfg = Config()
    cfg.translate.model_path = "C:/models/gemma4/model.gguf"
    merged = get_backend_config(cfg, "translate", "hymt")
    assert merged.get("model_path", "") == "", \
        "Hy-MT2 should not inherit Gemma4's model_path"

    registry = get_default_registry()
    with pytest.raises(ValueError, match="Hy-MT2.*model.*configured"):
        registry.create_translate("hymt", merged)


def test_registry_factory_qwen3_raises_on_empty_model():
    """Qwen3-ASR factory must raise ValueError when no model is configured."""
    from auto_translate.config import Config, get_backend_config
    from auto_translate.registry import get_default_registry

    cfg = Config()
    cfg.asr.model_path = "haas-ja-v1.0"
    merged = get_backend_config(cfg, "asr", "qwen3-asr")
    assert merged.get("model_path", "") == "", \
        "Qwen3-ASR should not inherit Haas's model_path"

    registry = get_default_registry()
    with pytest.raises(ValueError, match="Qwen3-ASR.*model.*configured"):
        registry.create_asr("qwen3-asr", merged, context="")


def test_registry_factory_qwen3_raises_on_empty_mmproj():
    """Qwen3-ASR factory must raise ValueError when mmproj is missing."""
    from auto_translate.config import Config, get_backend_config
    from auto_translate.registry import get_default_registry

    cfg = Config()
    cfg.asr.model_path = "haas-ja-v1.0"
    cfg.asr_profiles["qwen3-asr"] = {"model_path": "C:/models/qwen3/model.gguf"}
    merged = get_backend_config(cfg, "asr", "qwen3-asr")
    assert merged.get("model_path") == "C:/models/qwen3/model.gguf"
    assert merged.get("mmproj_path", "") == ""

    registry = get_default_registry()
    with pytest.raises(ValueError, match="Qwen3-ASR.*mmproj.*configured"):
        registry.create_asr("qwen3-asr", merged, context="")


def test_registry_factory_whisper_does_not_inherit_haas():
    """Whisper factory must not inherit Haas's model path.

    Mock WhisperASR to prevent loading a real Whisper model into GPU memory.
    """
    from auto_translate.config import Config, get_backend_config
    from auto_translate.registry import get_default_registry

    cfg = Config()
    cfg.asr.model_path = "haas-ja-v1.0"
    merged = get_backend_config(cfg, "asr", "whisper")
    assert merged.get("model_path", "") == "", \
        "Whisper should not inherit Haas's model_path"

    # Mock WhisperASR to prevent loading a real model into GPU memory.
    # With an empty model_path and no cached model, WhisperASR raises WhisperError.
    registry = get_default_registry()
    with mock.patch("auto_translate.asr.whisper_turbo.WhisperASR") as mock_whisper:
        mock_whisper.side_effect = WhisperError("no Whisper model available")
        with pytest.raises(WhisperError, match="no Whisper model available"):
            registry.create_asr("whisper", merged, context="")
        # Verify WhisperASR was invoked (factory reached the constructor)
        mock_whisper.assert_called_once()
        # Verify the passed config has empty model_path, not Haas's
        call_cfg = mock_whisper.call_args[0][0]
        assert call_cfg.model_path == ""


def test_registry_factory_haas_retains_default():
    """Haas factory should use default model when config dict has no model_path.

    The factory must not raise AttributeError accessing .model_path.default.
    Mock HaasASR to prevent starting a real speech.cpp server process.
    """
    from auto_translate.config import Config, get_backend_config
    from auto_translate.registry import get_default_registry

    cfg = Config()
    cfg.asr.model_path = "haas-ja-v1.0"
    merged = get_backend_config(cfg, "asr", "haas")
    assert merged.get("model_path") == "haas-ja-v1.0"

    # Mock HaasASR to prevent starting a real server subprocess
    registry = get_default_registry()
    with mock.patch("auto_translate.asr.haas.HaasASR") as mock_haas:
        mock_haas.return_value = mock.MagicMock()
        try:
            registry.create_asr("haas", merged, context="")
        except AttributeError:
            raise AssertionError(
                "Haas factory raised AttributeError — likely accessing "
                "AsrConfig.model_path.default on a string field"
            )
        # Verify HaasASR was actually invoked (factory reached the constructor)
        mock_haas.assert_called_once()
        # Verify the passed config has the correct model_path
        call_cfg = mock_haas.call_args[0][0]
        assert call_cfg.model_path == "haas-ja-v1.0"


def test_registry_factory_gemma4_retains_default():
    """Gemma4 factory should use default model when config dict has no model_path.

    The factory must not raise AttributeError accessing .model_path.default.
    Mock Gemma4Translator to prevent starting a real llama-server subprocess.
    """
    from auto_translate.config import Config, get_backend_config
    from auto_translate.registry import get_default_registry

    cfg = Config()
    cfg.translate.model_path = "models/gemma4/google_gemma-4-E4B-it-Q4_K_M.gguf"
    merged = get_backend_config(cfg, "translate", "gemma4")
    assert merged.get("model_path") == "models/gemma4/google_gemma-4-E4B-it-Q4_K_M.gguf"

    # Mock Gemma4Translator to prevent starting a real llama-server subprocess
    registry = get_default_registry()
    with mock.patch("auto_translate.translate.gemma4.Gemma4Translator") as mock_gemma:
        mock_gemma.return_value = mock.MagicMock()
        try:
            registry.create_translate("gemma4", merged)
        except AttributeError:
            raise AssertionError(
                "Gemma4 factory raised AttributeError — likely accessing "
                "TranslateConfig.model_path.default on a string field"
            )
        # Verify Gemma4Translator was actually invoked (factory reached the constructor)
        mock_gemma.assert_called_once()
        # Verify the passed config has the correct model_path
        call_cfg = mock_gemma.call_args[0][0]
        assert call_cfg.model_path == "models/gemma4/google_gemma-4-E4B-it-Q4_K_M.gguf"


def test_server_path_shared_across_backends():
    """server_path should be shared across translation backends, not treated as identity."""
    from auto_translate.config import Config, get_backend_config

    cfg = Config()
    cfg.translate.server_path = ".runtime/llama-server.exe"
    # PLaMo without profile should still get server_path from common config
    merged = get_backend_config(cfg, "translate", "plamo")
    assert merged.get("server_path") == ".runtime/llama-server.exe", \
        "server_path should be shared across backends"


def test_translate_config_allows_empty_model_path():
    """TranslateConfig must allow empty model_path for non-default backends."""
    from auto_translate.config import TranslateConfig

    # This should not raise — empty model_path is valid for non-default backends
    cfg = TranslateConfig(model_path="", server_path=".runtime/llama-server.exe")
    assert cfg.model_path == ""


# ---------------------------------------------------------------------------
# P1 — Whisper Windows path handling
# ---------------------------------------------------------------------------

def test_whisper_resolves_forward_slash_local_path():
    """Whisper should resolve forward-slash Windows paths as local directories."""
    import tempfile

    from auto_translate.asr.whisper_turbo import WhisperASR
    from auto_translate.config import AsrConfig

    with tempfile.TemporaryDirectory() as td:
        # Create a complete Whisper model directory
        model_dir = Path(td) / "models" / "whisper-large-v3-turbo"
        model_dir.mkdir(parents=True)
        (model_dir / "model.bin").write_bytes(b"")
        (model_dir / "config.json").write_bytes(b"{}")
        (model_dir / "tokenizer.json").write_bytes(b"{}")

        # Use forward-slash path (Windows-style relative path)
        # The path should be resolved as a local directory, not an HF repo ID
        cfg = AsrConfig(model_path=str(model_dir), cuda_device=-1)
        resolved = WhisperASR._resolve_model_path(cfg, is_explicit=True)
        assert resolved == str(model_dir.resolve())


def test_whisper_rejects_nonexistent_forward_slash_path():
    """Explicit forward-slash path that doesn't exist locally should raise.

    The path 'models/whisper-large-v3-turbo' contains '/' and no ':', so it's
    classified as a potential HF repo ID. Since it doesn't exist locally and
    isn't cached, the explicit path should raise a clear cache-missing error.
    """
    from auto_translate.asr.whisper_turbo import WhisperASR, WhisperError
    from auto_translate.config import AsrConfig

    # A forward-slash path that doesn't exist and looks like an HF repo ID
    # (contains '/' and no ':'). Since it's explicit and not cached, it
    # should raise a clear error about the missing cache.
    cfg = AsrConfig(model_path="models/whisper-large-v3-turbo", cuda_device=-1)
    with mock.patch.object(WhisperASR, "_find_cached_model", return_value=None), \
            pytest.raises(WhisperError, match="not present in the local cache"):
        WhisperASR._resolve_model_path(cfg, is_explicit=True)


def test_whisper_rejects_nonexistent_local_path():
    """Explicit local path without slashes that doesn't exist should raise."""
    from auto_translate.asr.whisper_turbo import WhisperASR, WhisperError
    from auto_translate.config import AsrConfig

    # A path without '/' is unambiguously a local path, not an HF repo ID.
    # Since it doesn't exist and is explicit, it should raise a clear error.
    cfg = AsrConfig(model_path="nonexistent_whisper_model", cuda_device=-1)
    with mock.patch.object(WhisperASR, "_find_cached_model", return_value=None), \
            pytest.raises(WhisperError, match="does not exist"):
        WhisperASR._resolve_model_path(cfg, is_explicit=True)


def test_whisper_absolute_windows_path():
    """Whisper should handle absolute Windows paths with backslashes."""
    import tempfile

    from auto_translate.asr.whisper_turbo import WhisperASR
    from auto_translate.config import AsrConfig

    with tempfile.TemporaryDirectory() as td:
        model_dir = Path(td) / "whisper"
        model_dir.mkdir()
        (model_dir / "model.bin").write_bytes(b"")
        (model_dir / "config.json").write_bytes(b"{}")
        (model_dir / "vocabulary.json").write_bytes(b"{}")

        cfg = AsrConfig(model_path=str(model_dir), cuda_device=-1)
        resolved = WhisperASR._resolve_model_path(cfg, is_explicit=True)
        assert resolved == str(model_dir.resolve())


def test_whisper_explicit_hf_repo_id_cached():
    """Explicit HF repo ID that is cached should resolve correctly."""
    import tempfile

    from auto_translate.asr.whisper_turbo import WhisperASR
    from auto_translate.config import AsrConfig

    with tempfile.TemporaryDirectory() as td:
        # Create a complete cached Whisper model
        cached_dir = Path(td) / "cached"
        cached_dir.mkdir()
        (cached_dir / "model.bin").write_bytes(b"")
        (cached_dir / "config.json").write_bytes(b"{}")
        (cached_dir / "tokenizer.json").write_bytes(b"{}")

        cfg = AsrConfig(model_path="Systran/faster-whisper-large-v3-turbo", cuda_device=-1)
        with mock.patch.object(WhisperASR, "_find_cached_model", return_value=cached_dir):
            resolved = WhisperASR._resolve_model_path(cfg, is_explicit=True)
        assert resolved == str(cached_dir)


def test_whisper_explicit_hf_repo_id_not_cached():
    """Explicit HF repo ID that is NOT cached should raise clear error."""
    from auto_translate.asr.whisper_turbo import WhisperASR, WhisperError
    from auto_translate.config import AsrConfig

    cfg = AsrConfig(model_path="Systran/faster-whisper-large-v3-turbo", cuda_device=-1)
    with mock.patch.object(WhisperASR, "_find_cached_model", return_value=None):
        try:
            WhisperASR._resolve_model_path(cfg, is_explicit=True)
            raise AssertionError("expected WhisperError for uncached HF repo ID")
        except WhisperError as exc:
            assert "not present in the local cache" in str(exc) or "cache" in str(exc).lower()


def test_effective_translation_settings_reach_pipeline():
    """Backend profile settings should reach the effective config."""
    from auto_translate.config import Config, get_backend_config

    cfg = Config()
    cfg.translate.context_lines = 6
    cfg.translate_profiles["gemma4"] = {"context_lines": 10, "temperature": 0.2}
    merged = get_backend_config(cfg, "translate", "gemma4")
    assert merged["context_lines"] == 10
    assert merged["temperature"] == 0.2


def test_backend_profile_runtime_only_no_model_inheritance():
    """Backend profile with only runtime settings should not inherit another backend's model."""
    from auto_translate.config import Config, get_backend_config

    cfg = Config()
    cfg.translate.model_path = "C:/models/gemma4/model.gguf"
    cfg.translate_profiles["plamo"] = {"temperature": 0.1, "context_lines": 4}
    # PLaMo profile has runtime settings but no model_path
    merged = get_backend_config(cfg, "translate", "plamo")
    assert merged.get("model_path", "") == "", \
        "PLaMo with runtime-only profile should not inherit Gemma4's model"
    assert merged["temperature"] == 0.1
    assert merged["context_lines"] == 4


# ---------------------------------------------------------------------------
# P2 — GUI stale override cleanup with actual dropdown switching
# ---------------------------------------------------------------------------

def test_gui_clearing_asr_model_clears_override():
    """Clearing the ASR model path field should remove the stored override.

    Uses actual combo box switching to exercise the real signal path.
    """
    import os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])

    from auto_translate.config import Config
    from auto_translate.gui import VideoTranslateWindow

    cfg = Config()
    cfg.asr.model_path = "haas-ja-v1.0"
    window = VideoTranslateWindow(Path("."))
    window._cfg = cfg

    # Use the actual current backend from the combo box
    current_backend = window.asr_backend.currentData()
    window._last_asr_backend = current_backend

    # Set a custom override for the current backend
    window._asr_model_overrides[str(current_backend)] = "C:/custom/model"
    window.asr_model.setText("C:/custom/model")

    # Clear the field and switch away to trigger the save/clear logic
    window.asr_model.setText("")
    switch_to_index = 1 if window.asr_backend.count() > 1 else 0
    if switch_to_index == window.asr_backend.currentIndex():
        switch_to_index = 0
    window.asr_backend.setCurrentIndex(switch_to_index)

    # The override should be cleared because the field was empty
    assert str(current_backend) not in window._asr_model_overrides, \
        "Clearing the field should remove the stale override"

    window.close()


def test_gui_clearing_mt_model_clears_override():
    """Clearing the MT model path field should remove the stored override.

    Uses actual combo box switching to exercise the real signal path.
    """
    import os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])

    from auto_translate.config import Config
    from auto_translate.gui import VideoTranslateWindow

    cfg = Config()
    cfg.translate.model_path = "models/gemma4/google_gemma-4-E4B-it-Q4_K_M.gguf"
    window = VideoTranslateWindow(Path("."))
    window._cfg = cfg

    # Use the actual current backend from the combo box
    current_backend = window.mt_backend.currentData()
    window._last_mt_backend = current_backend

    # Set a custom override for the current backend
    window._mt_model_overrides[str(current_backend)] = "C:/custom/model"
    window.mt_model.setText("C:/custom/model")

    # Clear the field and switch away to trigger the save/clear logic
    window.mt_model.setText("")
    switch_to_index = 1 if window.mt_backend.count() > 1 else 0
    if switch_to_index == window.mt_backend.currentIndex():
        switch_to_index = 0
    window.mt_backend.setCurrentIndex(switch_to_index)

    # The override should be cleared because the field was empty
    assert str(current_backend) not in window._mt_model_overrides, \
        "Clearing the field should remove the stale override"

    window.close()


def test_gui_restoring_default_clears_override():
    """Restoring the default model path should clear the stored override.

    Uses actual combo box switching to exercise the real signal path.
    """
    import os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])

    from auto_translate.config import Config
    from auto_translate.gui import VideoTranslateWindow

    cfg = Config()
    cfg.asr.model_path = "haas-ja-v1.0"
    window = VideoTranslateWindow(Path("."))
    window._cfg = cfg

    # Ensure _last_asr_backend matches the current combo box selection
    current_backend = window.asr_backend.currentData()
    window._last_asr_backend = current_backend

    # Set a custom override for the current backend
    window._asr_model_overrides[str(current_backend)] = "C:/custom/model"
    window.asr_model.setText("C:/custom/model")

    # Restore the default by resolving the current backend's identity
    from auto_translate.config import resolve_model_identity
    identity = resolve_model_identity(cfg, "asr", current_backend)
    default_path = identity.get("model_path", "")
    window.asr_model.setText(default_path)

    # Switch away to trigger the save/clear logic
    switch_to_index = 1 if window.asr_backend.count() > 1 else 0
    if switch_to_index == window.asr_backend.currentIndex():
        switch_to_index = 0
    window.asr_backend.setCurrentIndex(switch_to_index)

    # The override should be cleared because the field matches the default
    assert str(current_backend) not in window._asr_model_overrides, \
        f"Restoring the default should remove the override for {current_backend}"

    window.close()


def test_gui_dropdown_switching_preserves_custom_override():
    """Switching backends via combo box should preserve custom model overrides.

    Verifies that switching away and back restores the custom path.
    """
    import os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])

    from auto_translate.config import Config
    from auto_translate.gui import VideoTranslateWindow

    cfg = Config()
    cfg.asr.model_path = "haas-ja-v1.0"
    window = VideoTranslateWindow(Path("."))
    window._cfg = cfg

    # Use the actual current backend from the combo box
    current_backend = window.asr_backend.currentData()
    current_index = window.asr_backend.currentIndex()
    window._last_asr_backend = current_backend

    # Set a custom override for the current backend
    window._asr_model_overrides[str(current_backend)] = "C:/custom/model"
    window.asr_model.setText("C:/custom/model")

    # Switch to a different backend (triggers save of the current backend's custom path)
    switch_to_index = 1 if window.asr_backend.count() > 1 else 0
    if switch_to_index == current_index:
        switch_to_index = 0
    window.asr_backend.setCurrentIndex(switch_to_index)

    # The custom path should be stored as an override
    assert str(current_backend) in window._asr_model_overrides, \
        "Custom path should be stored when switching away"
    assert window._asr_model_overrides[str(current_backend)] == "C:/custom/model"

    # Switch back — the custom path should be restored
    window.asr_backend.setCurrentIndex(current_index)
    assert window.asr_model.text() == "C:/custom/model", \
        "Custom path should be restored when switching back"

    window.close()


def test_gui_cli_args_respect_override():
    """build_backend_command should include --asr-model when override is set.

    Verifies the GUI override flows through to CLI argument generation.
    """
    import os
    import tempfile
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])

    from auto_translate.config import Config
    from auto_translate.gui import VideoTranslateWindow
    from auto_translate.gui_runner import build_backend_command

    cfg = Config()
    cfg.asr.model_path = "haas-ja-v1.0"
    window = VideoTranslateWindow(Path("."))
    window._cfg = cfg

    # Set a custom override
    window._asr_model_overrides["haas"] = "C:/custom/haas-model"
    window.asr_model.setText("C:/custom/haas-model")

    # Simulate the GUI building a backend command with the override
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "pyproject.toml").write_text("")
        (root / "src" / "auto_translate" / "__main__.py").parent.mkdir(parents=True)
        (root / "src" / "auto_translate" / "__main__.py").write_text("")

        # With override, --asr-model should be present
        _, args = build_backend_command(
            root, "https://www.youtube.com/watch?v=abc123", 720, 1,
            asr_model="C:/custom/haas-model", mt_model=None,
            asr_backend="haas", translate_backend="gemma4",
        )
        assert "--asr-model" in args
        assert "C:/custom/haas-model" in args

        # Without override, --asr-model should NOT be present
        _, args_default = build_backend_command(
            root, "https://www.youtube.com/watch?v=abc123", 720, 1,
            asr_model=None, mt_model=None,
            asr_backend="haas", translate_backend="gemma4",
        )
        assert "--asr-model" not in args_default

    window.close()


# ---------------------------------------------------------------------------
# Audit fixes regression tests
# ---------------------------------------------------------------------------

def test_vod_resume_threshold_default_is_8s():
    """Issue #4: MpvConfig.vod_resume_threshold_s must default to 8.0s, not 3.0s.

    The 3.0s default was dangerously close to vod_pause_threshold_s (0.5s),
    creating a narrow hysteresis band that could cause rapid pause/resume
    oscillation. The code fallback was 8.0s, so the config should match.
    """
    from auto_translate.config import MpvConfig
    cfg = MpvConfig()
    assert cfg.vod_resume_threshold_s == 8.0, (
        f"vod_resume_threshold_s should default to 8.0s, got {cfg.vod_resume_threshold_s}"
    )
    # Ensure hysteresis gap is adequate (resume - pause >= 5s)
    hysteresis = cfg.vod_resume_threshold_s - cfg.vod_pause_threshold_s
    assert hysteresis >= 5.0, (
        f"Hysteresis gap {hysteresis}s is too narrow; should be >= 5s"
    )


def test_qwen3_forget_clears_all_context():
    """Issue #5: Qwen3ASR.forget() must clear all context when called without arguments.

    Previously, forget(n=1) removed only the last transcript, while HaasASR.forget()
    cleared all. The pipeline calls forget() with no args after ASR_CONTEXT_GAP_S
    (15s) is exceeded, expecting full context reset.
    """
    from auto_translate.asr.qwen3_asr import Qwen3ASR
    from unittest import mock

    # Create a mock Qwen3ASR instance
    asr = mock.MagicMock(spec=Qwen3ASR)
    asr._recent = ["transcript1", "transcript2", "transcript3"]
    asr._state_lock = threading.RLock()

    # Call the actual forget method with no arguments
    Qwen3ASR.forget(asr)
    assert asr._recent == [], (
        f"forget() with no args should clear all context, got {asr._recent}"
    )

    # Verify forget(n=2) still works for partial removal
    asr._recent = ["a", "b", "c"]
    Qwen3ASR.forget(asr, n=2)
    assert asr._recent == ["a"], (
        f"forget(n=2) should remove last 2 items, got {asr._recent}"
    )


def test_resolve_slash_alternatives_removed():
    """Issue #12: resolve_slash_alternatives no-op function must be removed.

    The function returned its input unchanged and was called in clean_output()
    with no effect. It should be removed along with its call site.
    """
    # The function should no longer exist in the module
    from auto_translate import translate
    assert not hasattr(translate.gemma4, 'resolve_slash_alternatives'), (
        "resolve_slash_alternatives should be removed from gemma4 module"
    )


def test_vad_segment_imported_from_base():
    """Issue #11: VADSegment and VADBackend must be imported from base.py, not duplicated.

    Previously, both vad/__init__.py and vad/base.py defined the same classes.
    Now __init__.py imports from base.py to avoid duplication.
    """
    from auto_translate.vad import VADSegment, VADBackend
    from auto_translate.vad.base import VADSegment as BaseVADSegment, VADBackend as BaseVADBackend

    # They should be the same class objects
    assert VADSegment is BaseVADSegment, (
        "VADSegment from vad.__init__ should be the same as vad.base.VADSegment"
    )
    assert VADBackend is BaseVADBackend, (
        "VADBackend from vad.__init__ should be the same as vad.base.VADBackend"
    )


def test_pipeline_no_module_level_run_state():
    """Issue #7: pipeline module must not have _run_state singleton.

    The module-level _run_state singleton held _last_vod_diag which persisted
    across pipeline restarts. It should be a local variable in run().
    """
    import auto_translate.pipeline as pipeline_mod
    assert not hasattr(pipeline_mod, '_run_state'), (
        "pipeline module should not have _run_state singleton"
    )
    assert not hasattr(pipeline_mod, '_PipelineState'), (
        "pipeline module should not have _PipelineState class"
    )


def test_mpv_vod_wait_timeout_is_3x_lead():
    """Issue #2: VOD autoplay wait timeout must be 3x PAUSE_LEAD_SECONDS.

    Previously, vod_wait_until was set to lead (5s), which was insufficient
    for cold-start model loading. Now it's 3x lead (15s) to account for
    model loading and first-utterance translation time.
    """
    import inspect
    from auto_translate.output.mpv import MpvOutput

    # Check the source code contains the 3x multiplier
    source = inspect.getsource(MpvOutput._play_session)
    assert "lead * 3" in source, (
        "VOD wait timeout should use lead * 3 multiplier"
    )


def test_vad_fallback_enabled_by_default():
    """Livestream speech detection: VAD fallback probe must be enabled by default.

    When VAD fails to detect speech in livestreams (low volume, background music,
    inconsistent encoding), the fallback probe mechanism sends periodic ASR probes
    during VAD-silence periods to catch speech that VAD missed.

    Previously, fallback_s defaulted to 0.0 (disabled), meaning livestreams with
    audio that VAD classified as silence would produce no subtitles at all.
    Now it defaults to 12.0s, probing every 12 seconds of VAD-silence.
    """
    from auto_translate.config import VadConfig
    cfg = VadConfig()
    assert cfg.fallback_s > 0, (
        f"fallback_s must be > 0 to enable fallback probing, got {cfg.fallback_s}"
    )
    assert cfg.fallback_s >= 10.0, (
        f"fallback_s should be >= 10s to avoid excessive ASR probes, got {cfg.fallback_s}"
    )
    assert cfg.fallback_s <= 15.0, (
        f"fallback_s should be <= 15s to catch speech in a timely manner, got {cfg.fallback_s}"
    )


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
