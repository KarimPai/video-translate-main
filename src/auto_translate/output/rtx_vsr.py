"""Isolated opt-in NVIDIA RTX Video Super Resolution integration for MPV.

MPV can request the NVIDIA scaling extension but only the NVIDIA driver/app
can confirm actual RTX video enhancement. This does not touch audio or ASR.
"""
from dataclasses import dataclass

VSR_LABEL = "@haas-rtx-vsr"
VSR_FILTER = "@haas-rtx-vsr:d3d11vpp=scale=3:scaling-mode=nvidia"


def launch_options(enabled: bool, hwdec: str, platform: str) -> list[str]:
    """Leave the normal player flags intact unless opt-in prerequisites match."""
    if not enabled or platform != "win32" or hwdec != "auto-safe":
        return []
    return ["--vo=gpu-next", "--gpu-api=d3d11", "--gpu-context=d3d11"]


def source_eligible(video_params: object) -> bool:
    """Only apply to source frames whose height can be verified <=480p."""
    if not isinstance(video_params, dict):
        return False
    try:
        h = int(video_params.get("h", 0))
    except (ValueError, TypeError, OverflowError):
        return False
    return 0 < h <= 480


def filter_present(filters: object) -> bool:
    """Require MPV to report our own labeled filter."""
    return isinstance(filters, list) and any(
        isinstance(item, dict)
        and str(item.get("label", "")).lstrip("@") == VSR_LABEL[1:]
        for item in filters
    )


def _nonnegative_counter(value: object) -> int | None:
    try:
        count = int(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return count if count >= 0 else None


@dataclass
class VsrPerformanceGuard:
    """Observe relative drop counts, ignoring manual pause and cache stalls."""
    last_at: float | None = None
    dropped: int | None = None
    delayed: int | None = None

    def sample(
        self, *, now: float, frame_drops: object, delayed_frames: object,
        hwdec: object, paused: bool, caching: bool,
    ) -> str | None:
        if paused or caching:
            self.last_at = None
            self.dropped = None
            self.delayed = None
            return None
        if hwdec == "no":
            return "hardware decode fell back to software"

        dropped = _nonnegative_counter(frame_drops)
        delayed = _nonnegative_counter(delayed_frames)
        if dropped is None or delayed is None:
            self.last_at = None
            return None

        if (self.last_at is None or self.dropped is None or self.delayed is None
                or dropped < self.dropped or delayed < self.delayed):
            self.last_at, self.dropped, self.delayed = now, dropped, delayed
            return None

        interval = now - self.last_at
        if interval < 25.0:
            return None
        delta_dropped = dropped - self.dropped
        delta_delayed = delayed - self.delayed
        self.last_at, self.dropped, self.delayed = now, dropped, delayed

        if delta_dropped >= 15 or delta_delayed >= 30:
            return (
                f"frame delivery regression over {interval:.0f}s: "
                f"{delta_dropped} dropped and {delta_delayed} delayed frames"
            )
        return None
