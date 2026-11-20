"""GPU memory observations and headroom-based optional-model admission.

No artificial device-wide VRAM ceiling: Windows/NVIDIA own allocation limits.
Use actual NVIDIA memory.total and memory.used to keep modest reserve for MPV
and transient CUDA allocations; never abort playback due to a fixed cap.
"""
import importlib
import logging
import math
import subprocess
import threading

log = logging.getLogger(__name__)
_MIB_TO_GB = 1048576 / 1e9
_NVML_LOCK = threading.Lock()
_SAMPLE_LOCK = threading.RLock()
_NVML_UNTRIED = object()
_NVML = _NVML_UNTRIED
_NVML_HANDLE = None
_LAST_METRICS = {}


def _nvml_metrics():
    """One persistent NVML connection; unsupported per-device fields stay unknown.

    NVML reports dedicated device memory. In particular, a WDDM driver may not
    expose a particular measurement; zero must never stand in for missing data.
    Initialization failures are cached to avoid repeated DLL loading each poll.
    """
    global _NVML, _NVML_HANDLE
    with _NVML_LOCK:
        if _NVML is _NVML_UNTRIED:
            try:
                module = importlib.import_module("pynvml")
                module.nvmlInit()
                handle = module.nvmlDeviceGetHandleByIndex(0)
                _NVML, _NVML_HANDLE = module, handle
            except Exception:  # noqa: BLE001
                _NVML = None
        if _NVML is None:
            return None
        metrics = {"source": "nvml", "used_gb": None, "total_gb": None, "free_gb": None,
                   "utilization_pct": None, "temperature_c": None, "power_w": None}
        try:
            memory = _NVML.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
            used, total = memory.used / 1e9, memory.total / 1e9
            if math.isfinite(used) and math.isfinite(total) and 0 <= used <= total and total > 0:
                metrics.update(used_gb=used, total_gb=total, free_gb=total - used)
        except Exception:  # noqa: S110, BLE001
            pass
        for field, getter in (
            ("utilization_pct", lambda: _NVML.nvmlDeviceGetUtilizationRates(_NVML_HANDLE).gpu),
            ("temperature_c", lambda: _NVML.nvmlDeviceGetTemperature(
                _NVML_HANDLE, _NVML.NVML_TEMPERATURE_GPU)),
            ("power_w", lambda: _NVML.nvmlDeviceGetPowerUsage(_NVML_HANDLE) / 1000),
        ):
            try:
                value = float(getter())
                if math.isfinite(value) and value >= 0 and (field != "utilization_pct" or value <= 100):
                    metrics[field] = value
            except Exception:  # noqa: S110, BLE001
                pass
        return metrics


def gpu_memory_gb():
    """Return (used, total) decimal GB, or (None, None) on unavailable data."""
    with _SAMPLE_LOCK:
        return _read_gpu_memory_gb()


def _read_gpu_memory_gb():
    global _LAST_METRICS
    measured = _nvml_metrics()
    if measured is not None and measured["used_gb"] is not None:
        _LAST_METRICS = measured
        return measured["used_gb"], measured["total_gb"]
    _LAST_METRICS = measured or {"source": "unavailable", "utilization_pct": None,
                                 "temperature_c": None, "power_w": None}
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--id=0", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=2, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        fields = [field.strip() for field in output.splitlines()[0].split(",")]
        if len(fields) != 2:
            return None, None
        used, total = (int(x) * _MIB_TO_GB for x in fields)
        if not 0 <= used <= total or total <= 0:
            return None, None
        _LAST_METRICS.update(source="nvidia-smi", used_gb=used, total_gb=total,
                             free_gb=total - used)
        return used, total
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None, None


def used_gb():
    """Compatibility utility for callers interested only in used device memory."""
    return gpu_memory_gb()[0]


class GpuBudget:
    def __init__(self, cfg):
        self.cfg = cfg
        self.used = None
        self.total = None
        self.free = None
        self.peak_used_gb = None
        self.minimum_free_gb = None
        self.stop = threading.Event()
        self.thread = None
        self._last_pressure = "normal"
        self.metrics = {}
        self.refresh()

    def refresh(self):
        # Admission and background sampling may coincide; keep their readings
        # together without ever placing a GPU call in snapshot()/the GUI.
        with _SAMPLE_LOCK:
            used, total = gpu_memory_gb()
            self.used, self.total = used, total
            self.free = None if used is None else max(0., total - used)
            if self.used is not None:
                self.peak_used_gb = max(self.used, self.peak_used_gb or 0.)
                self.minimum_free_gb = min(
                    self.free, self.minimum_free_gb if self.minimum_free_gb is not None else self.free)
            self.metrics = {**_LAST_METRICS, "used_gb": self.used,
                            "total_gb": self.total, "free_gb": self.free,
                            "peak_used_gb": self.peak_used_gb,
                            "minimum_free_gb": self.minimum_free_gb}
        return self.used

    def snapshot(self):
        """Return the latest sample without GPU calls or GUI-thread subprocesses."""
        return {**self.metrics, "pressure": self.pressure_level}

    @property
    def pressure_level(self):
        if self.free is None:
            return "unknown"
        if self.free <= self.cfg.emergency_free_gb:
            return "critical"
        if self.free <= self.cfg.low_free_gb:
            return "low"
        return "normal"

    def admits(self, reserve):
        """Is there room for an estimated *additional* allocation plus slack?"""
        self.refresh()
        return (reserve >= 0 and self.free is not None
                and self.free >= reserve + self.cfg.startup_headroom_gb)

    def start(self, release):
        def monitor():
            while not self.stop.is_set():
                self.refresh()
                level = self.pressure_level
                from .runtime_diagnostics import record_event, update_metrics
                update_metrics(gpu=self.snapshot())
                log.info("GPU 0 used=%s GB total=%s GB free=%s GB; pressure=%s (no hard software cap)",
                         None if self.used is None else round(self.used, 3),
                         None if self.total is None else round(self.total, 3),
                         None if self.free is None else round(self.free, 3), level)
                if level in ("low", "critical"):
                    if level != self._last_pressure:
                        log.warning("GPU free-memory pressure=%s; keeping ASR/translation playback alive",
                                    level)
                        record_event("gpu_memory_pressure", **self.snapshot())
                    try:
                        release()
                    except Exception:
                        # Optional-model cleanup must not kill the sampler.
                        log.exception("GPU pressure resource release failed")
                self._last_pressure = level
                self.stop.wait(self.cfg.sample_interval_s)

        self.thread = threading.Thread(target=monitor, daemon=True, name="gpu-budget")
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(4)
