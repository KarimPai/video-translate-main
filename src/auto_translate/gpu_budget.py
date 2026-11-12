"""Device-wide NVIDIA memory policy. nvidia-smi reports MiB, policy uses decimal GB."""
import logging
import subprocess
import threading

log = logging.getLogger(__name__)


def used_gb():
    try:
        value = subprocess.check_output(
            ["nvidia-smi", "--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=2, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return int(value.strip()) * 1048576 / 1e9
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class GpuBudget:
    def __init__(self, cfg):
        self.cfg = cfg
        self.used = used_gb()
        self.stop = threading.Event()
        self.thread = None

    def refresh(self):
        self.used = used_gb()
        return self.used

    def admits(self, reserve):
        self.refresh()
        return self.used is not None and self.used + reserve <= self.cfg.target_gb

    def start(self, release):
        def monitor():
            while not self.stop.is_set():
                self.used = used_gb()
                log.info("GPU 0 total used GB=%s target=%.2f", self.used, self.cfg.target_gb)
                if self.used is None or self.used >= self.cfg.emergency_threshold_gb:
                    release()
                self.stop.wait(2)
        self.thread = threading.Thread(target=monitor, daemon=True, name="gpu-budget")
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(4)
