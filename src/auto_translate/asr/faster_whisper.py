
import logging
import numpy as np

from .. import cuda_dlls
from ..config import AsrConfig

cuda_dlls.setup()

from faster_whisper import WhisperModel

log = logging.getLogger(__name__)


class FasterWhisperASR:
    def __init__(self, cfg: AsrConfig) -> None:
        self.cfg = cfg
        self.model = WhisperModel(
            cfg.model,
            device=cfg.device,
            compute_type=cfg.compute_type,
        )

    def transcribe(self, audio: np.ndarray, probe: bool = False) -> str:
        segments, _ = self.model.transcribe(
            audio,
            language="ja",
            beam_size=self.cfg.beam_size,
            initial_prompt=self.cfg.initial_prompt,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,
            no_speech_threshold=self.cfg.no_speech_threshold,
            log_prob_threshold=self.cfg.log_prob_threshold,
            repetition_penalty=1.08,
            no_repeat_ngram_size=3,
            compression_ratio_threshold=2.0,
            temperature=0.0,
            hotwords="ホロライブ, 星街すいせい, 兎田ぺこら, 宝鐘マリン, さくらみこ, 白上フブキ, 博衣こより, 獅白ぼたん, ホロメン, スパチャ",
        )

        parts = []

        for seg in segments:
            text = seg.text.strip()

            log.info(
                "ASR RAW: %r | no_speech=%.3f | logprob=%.3f | compression=%.3f",
                text,
                seg.no_speech_prob,
                seg.avg_logprob,
                seg.compression_ratio,
            )

            # Reject unreliable speech and likely hallucinations
            if (
                seg.no_speech_prob > self.cfg.no_speech_threshold
                or seg.avg_logprob < self.cfg.log_prob_threshold
                or seg.compression_ratio > 2.0
            ):
                log.info("Discarding unreliable ASR segment: %s", text)
                continue

            # Additional filtering for probe transcription
            if probe and (
                seg.no_speech_prob > self.cfg.probe_no_speech_max
                or seg.avg_logprob < self.cfg.probe_log_prob_min
            ):
                log.debug("Discarding probe segment: %s", text)
                continue

            # Reject configured unwanted phrases
            if any(phrase in text for phrase in self.cfg.blocklist):
                log.debug("Discarding blocked ASR text: %s", text)
                continue

            parts.append(text)

        return "".join(parts)
