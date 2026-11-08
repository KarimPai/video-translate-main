from ..types import Subtitle


def _speaker_label(sub: Subtitle) -> str:
    """Display anonymous voice IDs in terminal diagnostics, not MPV."""
    return (
        f" [{chr(65 + sub.speaker_id)}]"
        if sub.speaker_id is not None and 0 <= sub.speaker_id < 26
        else ""
    )


def fmt_ts(seconds: float) -> str:
    """12.4 → '00:12.4', 3725.0 → '1:02:05.0'."""
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h}:{m:02d}:{s:04.1f}" if h else f"{m:02d}:{s:04.1f}"


class TerminalOutput:
    def __init__(self, *, simple: bool = False) -> None:
        self.simple = simple

    def show(self, sub: Subtitle) -> None:
        if self.simple:
            # Send both transcript lines in a single flushed write. Windows
            # console I/O is synchronous; avoid two flushes per utterance.
            lines = []
            speaker = _speaker_label(sub)
            if sub.ja:
                lines.append(f"JP{speaker}: {sub.ja}")
            if sub.en:
                lines.append(f"EN{speaker}: {sub.en}")
            if lines:
                print("\n".join(lines), flush=True)
        else:
            print_subtitle(sub)

    def drain(self) -> None:
        pass

    def close(self) -> None:
        pass


def print_subtitle(sub: Subtitle) -> None:
    tm = sub.timings
    lat = (f"vad {tm.segmented - tm.audio_read:.2f}s · wait {tm.asr_start - tm.segmented:.2f}s · "
           f"asr {tm.asr_end - tm.asr_start:.2f}s · mt {tm.mt_end - tm.mt_start:.2f}s")
    stamp = f"[{fmt_ts(sub.start_s)}–{fmt_ts(sub.end_s)}]"
    speaker = _speaker_label(sub)
    print(f"{stamp}{speaker} {sub.ja}    ({lat})", flush=True)
    if sub.en:
        print(f"{' ' * len(stamp)} →{speaker} {sub.en}", flush=True)
