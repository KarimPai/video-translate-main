import logging
import threading
from dataclasses import replace

from ..config import AsrConfig
from .base import ASR
from .haas import HaasASR

log = logging.getLogger(__name__)


class PrimaryASR:
    """Replace a timed-out primary on the next cue without reusing its pipe."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.worker = HaasASR(cfg)
        self.last_stop = None
        self.closed = False
        self.lock = threading.Lock()
        self.failures = 0
        self.last_failed = False

    def transcribe(self, audio, probe=False):
        with self.lock:
            if self.closed:
                raise RuntimeError("primary ASR closed")
            if self.worker is None:
                self.worker = HaasASR(self.cfg)
            worker = self.worker
        try:
            result = worker.transcribe(audio, probe)
        except (TimeoutError, OSError, RuntimeError) as exc:
            with self.lock:
                if self.worker is worker:
                    self.worker = None
                    worker.close()
            self.failures += 1
            self.last_stop = None
            self.last_failed = True
            if self.closed or self.failures >= 3:
                raise
            # A transient Vulkan timeout should not immediately force slow
            # CPU inference; restart that device once before falling back.
            if self.failures >= 2 and self.cfg.haas_device.lower() != "cpu":
                self.cfg = replace(self.cfg, haas_device="cpu")
                log.warning("Repeated Haas failures; switching next decode to CPU: %s", exc)
            else:
                log.warning("Haas worker failed; restarting next cue on %s: %s",
                            self.cfg.haas_device, exc)
            return ""
        self.failures = 0
        self.last_failed = False
        self.last_stop = worker.last_stop
        return result

    def close(self):
        with self.lock:
            self.closed = True
            if self.worker:
                self.worker.close()
                self.worker = None


def create_asr(cfg: AsrConfig) -> ASR:
    return PrimaryASR(cfg)
