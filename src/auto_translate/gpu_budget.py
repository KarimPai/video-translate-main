"""GPU memory observations and headroom-based optional-model admission.

No artificial device-wide VRAM ceiling: Windows/NVIDIA own allocation limits.
Use actual NVIDIA memory.total and memory.used to keep modest reserve for MPV
and transient CUDA/Vulkan allocations; never abort playback due to a fixed cap.
"""
import logging
import subprocess
import threading

log = logging.getLogger(__name__)
_MIB_TO_GB = 1048576 / 1e9


def gpu_memory_gb():
    """Return (used, total) decimal GB, or (None, None) on unavailable data."""
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
        self.stop = threading.Event()
        self.thread = None
        self._last_pressure = "normal"
        self.refresh()

    def refresh(self):
        used, total = gpu_memory_gb()
        self.used, self.total = used, total
        self.free = None if used is None else max(0., total - used)
        return self.used

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
                log.info("GPU 0 used=%s GB total=%s GB free=%s GB; pressure=%s (no hard software cap)",
                         None if self.used is None else round(self.used, 3),
                         None if self.total is None else round(self.total, 3),
                         None if self.free is None else round(self.free, 3), level)
                if level in ("low", "critical"):
                    if level != self._last_pressure:
                        log.warning("GPU free-memory pressure=%s; keeping Haas/PLaMo playback alive",
                                    level)
                    release()
                self._last_pressure = level
                self.stop.wait(self.cfg.sample_interval_s)

        self.thread = threading.Thread(target=monitor, daemon=True, name="gpu-budget")
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(4)
