"""Optional, explicit VOD timing refinement using WhisperX Japanese alignment.

Install the ``refinement`` extra separately. Stop playback/translation and unload
their models before calling this function on an 8 GB GPU; CPU is the default.
WhisperX may download its Japanese alignment model on the first explicit call.
Audio is decoded into RAM once, so long VODs require additional host memory.
Alignment refines Japanese cue timing, not translation quality or English word
timing. This module does not import WhisperX or PyTorch at application startup.
"""
import gc
import logging
import math
import threading
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path

from .subtitle_files import validate_subtitles
from .types import SAMPLE_RATE, Subtitle

log = logging.getLogger(__name__)


class RefinementCancelled(RuntimeError):
    """The caller cancelled refinement; no partial result should be applied."""


def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise RefinementCancelled("Subtitle timing refinement cancelled")


def refine_subtitle_timings(
    media_path: str | Path,
    subtitles: Iterable[Subtitle],
    *,
    device: str = "cpu",
    cancel: threading.Event | None = None,
    is_live: bool = False,
    model_name: str | None = None,
) -> list[Subtitle]:
    """Return independent cues aligned to local VOD audio; never persist edits.

    Only existing Japanese text is aligned, one cue at a time. Unalignable cues
    keep their original timing. Source timestamps are absolute media seconds;
    provide the matching complete local media file, not a differently offset
    clip. Cancellation is cooperative between decode/model/alignment calls and
    discards the whole pending result; a native model call cannot be interrupted.
    """
    if is_live:
        raise ValueError("Timing refinement is available for local VOD files only")
    _check_cancel(cancel)
    path = Path(media_path)
    if not path.is_file():
        raise ValueError("Timing refinement requires an existing local media file")
    if device != "cpu" and device != "cuda" and not device.startswith("cuda:"):
        raise ValueError("Refinement device must be cpu or cuda")
    cues = validate_subtitles(subtitles)
    if not cues:
        return []
    if not any(cue.ja.strip() for cue in cues):
        raise ValueError("Japanese transcript text is required for timing refinement")
    try:
        import whisperx
    except ImportError as exc:
        raise RuntimeError(
            "Timing refinement requires WhisperX; install the 'refinement' extra "
            "in a separate compatible environment (uv sync --extra refinement)."
        ) from exc

    model = metadata = audio = None
    try:
        audio = whisperx.load_audio(str(path))
        _check_cancel(cancel)
        duration = len(audio) / SAMPLE_RATE
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Media contains no decodable audio")
        if any(cue.end_s > duration + 0.001 for cue in cues):
            raise ValueError("Subtitle timestamps exceed the media duration; select the matching VOD")
        kwargs = {"language_code": "ja", "device": device}
        if model_name:
            kwargs["model_name"] = model_name
        model, metadata = whisperx.load_align_model(**kwargs)
        _check_cancel(cancel)
        refined = []
        for cue in cues:
            _check_cancel(cancel)
            if not cue.ja.strip():
                refined.append(deepcopy(cue))
                continue
            result = whisperx.align(
                [{"start": cue.start_s, "end": cue.end_s, "text": cue.ja}],
                model, metadata, audio, device, return_char_alignments=False,
            )
            _check_cancel(cancel)
            # Words without valid character alignment (e.g. unknown glyphs) have
            # no timestamps. Require at least one real aligned word; the segment
            # start/end alone can simply be WhisperX's unchanged fallback.
            words = [
                word for segment in result.get("segments", [])
                for word in segment.get("words", [])
                if _valid_word(word, cue)
            ]
            updated = deepcopy(cue)
            if words:
                start = min(float(word["start"]) for word in words)
                end = max(float(word["end"]) for word in words)
                # Missing words at either edge must not truncate valid speech.
                aligned_text = "".join(str(word.get("word", "")) for word in words)
                if "".join(aligned_text.split()) == "".join(cue.ja.split()) and end > start:
                    updated.start_s = start
                    updated.end_s = end
                    updated.speech_start_s = start
                else:
                    log.info("Keeping cue %.3f timing: incomplete Japanese alignment", cue.start_s)
            else:
                log.info("Keeping cue %.3f timing: no aligned Japanese words", cue.start_s)
            refined.append(updated)
        _check_cancel(cancel)
        return validate_subtitles(refined)
    finally:
        # Do not retain an alignment model after this explicit offline operation.
        model = metadata = audio = None
        gc.collect()


def _valid_word(word: dict, cue: Subtitle) -> bool:
    try:
        start, end = float(word["start"]), float(word["end"])
        return (math.isfinite(start) and math.isfinite(end)
                and cue.start_s <= start < end <= cue.end_s)
    except (KeyError, TypeError, ValueError):
        return False
