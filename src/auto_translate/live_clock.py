"""Media timestamps from the shared HLS epoch. Wall time only bounds poll gaps."""

from dataclasses import dataclass
import logging
import math
import re

log = logging.getLogger(__name__)


class Discontinuity(RuntimeError):
    pass


# asettb immediately before ashowinfo fixes the time base to 1/output_rate.
# Use the integer PTS: pts_time is rounded severely for multi-hour inputs.
_SEGMENT = re.compile(rb"Opening '[^']*/segment_(\d+)\.m4s' for reading")
_FRAME = re.compile(rb"\bn:\s*(\d+)\s+pts:\s*(\S+).*?\brate:\s*(\d+)\s+nb_samples:\s*(\d+)")


def pcm_frames(stderr, stdout, rate, channels):
    """Pair post-resample frame records with exactly their packed PCM bytes."""
    expected = None
    pending = bytearray()
    pending_samples = 0
    number = 0
    last_segment = None
    for line in stderr:
        if b"Failed to open segment" in line or b"skipping" in line and b"segments ahead" in line:
            raise Discontinuity("local HLS reader missed a media segment")
        if b"Parsed_ashowinfo_" not in line:
            segment = _SEGMENT.search(line)
            if segment:
                sequence = int(segment.group(1))
                if last_segment is not None and sequence not in (last_segment, last_segment + 1):
                    raise Discontinuity(f"local HLS sequence jumped {last_segment} -> {sequence}")
                if last_segment is None or sequence % 30 == 0:
                    log.info("live audio opened segment=%d", sequence)
                last_segment = sequence
            if b"failed" in line.lower() or b"error" in line.lower():
                log.warning("live audio: %s", line.decode(errors="replace").strip())
            continue
        # ashowinfo also emits side-data messages with the same logger prefix.
        # Only frame records have an index and correspond to bytes on stdout.
        if not re.search(rb"\b(?:n|pts|nb_samples):", line):
            continue
        match = _FRAME.search(line)
        if not match:
            raise Discontinuity(f"malformed live audio frame {number}: {line[:240]!r}")
        index, raw_pts, actual_rate, count = match.groups()
        count = int(count)
        if int(index) != number or int(actual_rate) != rate or not 0 < count <= rate:
            raise Discontinuity(f"invalid live audio frame metadata at frame {number}")
        size = count * channels * 2
        data = stdout.read(size)
        if len(data) != size:
            raise Discontinuity("truncated timestamped PCM frame")
        number += 1
        if raw_pts in (b"NOPTS", b"N/A", b"-9223372036854775808"):
            pending_samples += count
            if pending_samples > rate // 2:
                raise Discontinuity("live audio PTS unavailable for over 500 ms")
            pending.extend(data)
            continue
        if not re.fullmatch(rb"-?\d+", raw_pts):
            raise Discontinuity(f"unrecognized live audio PTS {raw_pts[:48]!r}")
        pts = int(raw_pts)
        if expected is not None and abs(pts - (expected + pending_samples)) > 1:
            raise Discontinuity(f"audio PTS gap: expected={expected + pending_samples} observed={pts} rate={rate}")
        if pending_samples:
            if expected is not None:
                # Both surrounding timestamps agree with the exact PCM duration.
                yield expected, bytes(pending)
            # A leading untimed interval has no verified start. Discard its PCM.
            pending.clear()
            pending_samples = 0
        yield pts, data
        expected = pts + count
    if pending_samples:
        raise Discontinuity("audio ended before missing PTS could be verified")



def pcm_batches(stderr, stdout, rate, channels, target_samples):
    """Batch validated adjacent frames for FireRed's streaming GPU forward pass."""
    buffer = bytearray()
    origin = 0
    for pts, data in pcm_frames(stderr, stdout, rate, channels):
        if not buffer:
            origin = pts
        buffer.extend(data)
        if len(buffer) >= target_samples * channels * 2:
            yield origin, bytes(buffer)
            buffer.clear()
    if buffer:
        yield origin, bytes(buffer)


@dataclass
class LiveMediaClock:
    epoch_id: int
    last_pos: float | None = None
    last_poll: float = 0.0
    verified: bool = False
    mapping_error: float | None = None
    initial_error: float | None = None

    def observe(self, position, audio_pts, span, sampled_at):
        """Independent player audio/video agreement inside the decoded media span."""
        self.verified = False
        self.mapping_error = None
        if position is None or audio_pts is None or span is None:
            return None
        if not all(math.isfinite(p) for p in (position, audio_pts)):
            return None
        self.mapping_error = position - audio_pts
        if abs(self.mapping_error) > 0.5:
            return None
        if self.last_pos is not None:
            step = audio_pts - self.last_pos
            if step < -0.5 or step > max(0, sampled_at - self.last_poll) + 2:
                raise Discontinuity(f"player PTS jumped {self.last_pos:.3f} -> {audio_pts:.3f}")
        if not span[0] - 0.1 <= audio_pts <= span[1] + 0.1:
            return None
        self.last_pos = audio_pts
        self.last_poll = sampled_at
        self.verified = True
        if self.initial_error is None:
            self.initial_error = self.mapping_error
        return audio_pts
