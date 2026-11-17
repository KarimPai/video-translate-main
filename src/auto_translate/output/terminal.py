from dataclasses import replace

from ..subtitle_clean import sanitize_english_output
from ..types import Subtitle


def fmt_ts(seconds: float) -> str:
    """12.4 → '00:12.4', 3725.0 → '1:02:05.0'."""
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h}:{m:02d}:{s:04.1f}" if h else f"{m:02d}:{s:04.1f}"


class TerminalOutput:
    def __init__(self, *, simple: bool = False) -> None:
        self.simple = simple

    def show(self, sub: Subtitle) -> None:
        if sub.en:
            cleaned = sanitize_english_output(sub.en)
            if not cleaned:
                return
            sub = replace(sub, en=cleaned)
        if self.simple:
            # Send both transcript lines in a single flushed write. Windows
            # console I/O is synchronous; avoid two flushes per utterance.
            lines = []
            if sub.ja:
                lines.append(f"JP: {sub.ja}")
            if sub.en:
                lines.append(f"EN: {sub.en}")
            if lines:
                print("\n".join(lines), flush=True)
        else:
            print_subtitle(sub)

    def drain(self) -> None:
        pass

    def close(self) -> None:
        pass


def print_subtitle(sub: Subtitle) -> None:
    if sub.en:
        cleaned = sanitize_english_output(sub.en)
        if not cleaned:
            return
        sub = replace(sub, en=cleaned)
    tm = sub.timings
    lat = (f"vad {tm.segmented - tm.audio_read:.2f}s · wait {tm.asr_start - tm.segmented:.2f}s · "
           f"asr {tm.asr_end - tm.asr_start:.2f}s · mt {tm.mt_end - tm.mt_start:.2f}s")
    stamp = f"[{fmt_ts(sub.start_s)}–{fmt_ts(sub.end_s)}]"
    print(f"{stamp} {sub.ja}    ({lat})", flush=True)
    if sub.en:
        print(f"{' ' * len(stamp)} → {sub.en}", flush=True)
