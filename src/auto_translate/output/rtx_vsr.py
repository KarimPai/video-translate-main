"""Isolated, optional RTX Video Super Resolution MPV integration helpers.

MPV requests the NVIDIA extension; actual AI engagement is controlled by the
NVIDIA driver and can only be confirmed with its RTX Video status indicator.
"""

from dataclasses import dataclass

VSR_LABEL = "@whisper-rtx-vsr"
VSR_FILTER = "@whisper-rtx-vsr:d3d11vpp=scale=3:scaling-mode=nvidia"
CHECK_INTERVAL_S = 30.0


def launch_options(enabled: bool, hwdec: str, platform: str) -> list[str]:
    """Leave the previous MPV renderer untouched unless VSR is requested."""
    if not enabled or platform != "win32" or hwdec != "auto-safe":
        return []
    return ["--vo=gpu-next", "--gpu-api=d3d11", "--gpu-context=d3d11"]


def source_eligible(video_params: object) -> bool:
    """Never upscale a source above 480p (including quality fallback cases)."""
    if not isinstance(video_params, dict):
        return False
    try:
        height = int(video_params.get("h", 0))
    except (ValueError, TypeError, OverflowError):
        return False
    return 0 < height <= 480


def filter_present(filters: object) -> bool:
    """Check MPV's current filter list, not the requested state alone."""
    return isinstance(filters, list) and any(
        isinstance(item, dict)
        and str(item.get("label", "")).lstrip("@") == VSR_LABEL[1:]
        for item in filters
    )


def _nonnegative_counter(value: object) -> int | None:
    try:
        count = int(value)  # mpv may return numeric strings or native integers
    except (ValueError, TypeError, OverflowError):
        return None
    return count if count >= 0 else None


@dataclass
class VsrPerformanceGuard:
    """Detect sustained drops over sample windows, ignoring cache/user pauses."""

    last_at: float | None = None
    dropped: int | None = None
    delayed: int | None = None

    def sample(
        self,
        *,
        now: float,
        frame_drops: object,
        delayed_frames: object,
        hwdec: object,
        paused: bool,
        caching: bool,
    ) -> str | None:
        if paused or caching:
            self.last_at = None
            self.dropped = None
            self.delayed = None
            return None
        if hwdec == "no":
            return "hardware decoding fell back to software"

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

        # 15 dropped or 30 late frames in ~30 seconds is excessive for the
        # 480p-to-1440p experiment and warrants reverting the optional filter.
        if delta_dropped >= 15 or delta_delayed >= 30:
            return (
                f"video performance regression over {interval:.0f}s: "
                f"{delta_dropped} dropped and {delta_delayed} delayed frames"
            )
        return None
