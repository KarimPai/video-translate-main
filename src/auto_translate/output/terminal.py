"""Terminal output for direct Granite Speech audio-to-English translation."""

from ..types import Subtitle


def fmt_ts(seconds: float) -> str:
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}:{minutes:02d}:{seconds:04.1f}" if hours else f"{minutes:02d}:{seconds:04.1f}"


class TerminalOutput:
    def __init__(self, *, simple: bool = False) -> None:
        self.simple = simple

    def show(self, sub: Subtitle) -> None:
        if not sub.en:
            return
        if self.simple:
            print(f"EN: {sub.en}", flush=True)
        else:
            decode_s = max(0.0, sub.timings.asr_end - sub.timings.asr_start)
            print(f"[{fmt_ts(sub.start_s)}–{fmt_ts(sub.end_s)}] "
                  f"{sub.en}  (Granite {decode_s:.2f}s)", flush=True)

    def drain(self) -> None:
        pass

    def close(self) -> None:
        pass
