"""GpuBudget monitor resilience: a failed sample must not end monitoring."""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from auto_translate import gpu_budget as gb


def test_monitor_survives_refresh_exception(monkeypatch):
    monkeypatch.setattr(gb, "gpu_memory_gb", lambda: (2.0, 8.0))
    cfg = SimpleNamespace(sample_interval_s=0.01, emergency_free_gb=0.3,
                          low_free_gb=0.8, startup_headroom_gb=0.2)
    budget = gb.GpuBudget(cfg)
    calls = {"n": 0}
    recovered = threading.Event()
    real_refresh = budget.refresh

    def flaky_refresh():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated NVML/diagnostic failure")
        real_refresh()
        if calls["n"] >= 3:
            recovered.set()

    budget.refresh = flaky_refresh
    budget.start(lambda: None)
    try:
        assert recovered.wait(3.0), "GPU sampler stopped after a failed iteration"
    finally:
        budget.close()
    assert budget.thread is not None and not budget.thread.is_alive()


def test_critical_pressure_relaxes_to_low_before_normal():
    budget = gb.GpuBudget.__new__(gb.GpuBudget)
    budget.cfg = SimpleNamespace(emergency_free_gb=0.25, low_free_gb=0.8)
    budget._last_pressure = "critical"
    budget.free = 0.5

    assert budget.pressure_level == "low"


def test_rejected_relief_is_not_counted_as_a_request(monkeypatch):
    monkeypatch.setattr(gb, "gpu_memory_gb", lambda: (7.5, 8.0))
    cfg = SimpleNamespace(sample_interval_s=0.01, emergency_free_gb=0.3,
                          low_free_gb=0.8, startup_headroom_gb=0.2)
    budget = gb.GpuBudget(cfg)
    attempted = threading.Event()

    def reject_relief():
        attempted.set()
        return False

    budget.start(reject_relief)
    try:
        assert attempted.wait(2.0)
    finally:
        budget.close()

    assert budget._last_low_relief_time == 0.0