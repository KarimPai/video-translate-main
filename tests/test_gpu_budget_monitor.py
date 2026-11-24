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