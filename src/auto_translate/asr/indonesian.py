"""Optional CPU Whisper language gate and Indonesian-audio -> English captions.

Nemotron 3.5 does not support Indonesian (id-ID). Do not guess Indonesian from
Japanese/English model output: use independent Whisper multilingual language ID.

Both the encoder and decoder run on CPU/int8, so no extra CUDA/Vulkan VRAM is
reserved. The model must be downloaded explicitly before first run.
"""
from dataclasses import dataclass
import logging
import re
import time

import numpy as np

from ..text_filter import is_usable_transcript

log = logging.getLogger(__name__)

_JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_OUTRO = re.compile(
    r"^(?:thank you for watching|please subscribe(?: to (?:the|my) channel)?"
    r"|subtitles by.{0,60})[.!?\s]*$", re.I
)


@dataclass(frozen=True)
class IndonesianResult:
    """Accepted Indonesian speech; text is English, possibly empty on failure."""

    english: str
    confidence: float
    elapsed_s: float
    reason: str = "indonesian_whisper"


class IndonesianWhisper:
    """Persistent CPU CTranslate2 Whisper small; no extra GPU model."""

    def __init__(self, model_name="small", cpu_threads=4):
        from faster_whisper import WhisperModel

        log.info("Loading multilingual Whisper %s on CPU int8 for Indonesian ID/EN",
                 model_name)
        started = time.monotonic()
        # Prevent a surprise model download while opening an MPV stream.
        # Download once using the setup command in the README.
        self.model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
            cpu_threads=cpu_threads,
            num_workers=1,
            local_files_only=True,
        )
        log.info("Indonesian CPU Whisper ready in %.1fs", time.monotonic() - started)

    def transcribe_if_indonesian(
        self, audio, *, min_probability=0.65,
        min_margin=0.20, min_duration_s=1.2,
    ) -> IndonesianResult | None:
        """Conservative independent ID gate; return None for JA/EN/uncertain."""
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        duration = len(audio) / 16000
        if duration < min_duration_s or audio.ndim != 1 or not np.isfinite(audio).all():
            return None
        started = time.monotonic()
        # Detection happens when transcribe() returns its TranscriptionInfo.
        # Do not iterate these one-token provisional decode segments.
        _, info = self.model.transcribe(
            audio,
            language=None,
            task="transcribe",
            beam_size=1,
            best_of=1,
            max_new_tokens=1,
            without_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=False,
            temperature=0.0,
        )
        language = str(getattr(info, "language", "")).lower()
        probability = float(getattr(info, "language_probability", 0))
        alternatives = getattr(info, "all_language_probs", None)
        if alternatives:
            competitors = [
                float(score) for code, score in alternatives
                if str(code).lower() != "id"
            ]
            runner_up = max(competitors, default=0.0)
            gap = probability - runner_up
        else:
            # With no alternatives, require considerably stronger confidence.
            gap = max(0.0, probability - 0.15)

        if language == "id" or probability >= 0.2 and any(
                code == "id" and score >= 0.2
                for code, score in (alternatives or [])):
            log.info(
                "Indonesian Whisper LID candidate: predicted=%s confidence=%.3f "
                "margin=%.3f audio=%.2fs", language, probability, gap, duration,
            )
        if language != "id" or probability < min_probability or gap < min_margin:
            return None

        segments, _ = self.model.transcribe(
            audio,
            language="id",
            task="translate",
            beam_size=4,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
        en = " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()
        if (not en or not is_usable_transcript(en) or _JAPANESE.search(en)
                or _OUTRO.fullmatch(en)):
            en = ""
        elapsed = time.monotonic() - started
        log.info("Indonesian Whisper route audio=%.2fs confidence=%.3f "
                 "cpu_elapsed=%.3fs English=%r",
                 duration, probability, elapsed, en[:160])
        return IndonesianResult(en, probability, elapsed,
                                "indonesian_whisper" if en else "indonesian_no_english")
