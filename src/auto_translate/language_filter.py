"""Conservative acoustic language gate for Japanese-only transcription.

Classify original 16 kHz PCM BEFORE Japanese ASR. Never derive speech language
from Japanese-forced transcripts; that text is unreliable for code-switching.

An uncertain result is passed through to protect Japanese dialogue.
"""
import logging
import time
from dataclasses import dataclass

import numpy as np

from .config import LanguageFilterConfig
from .types import SAMPLE_RATE

log = logging.getLogger(__name__)

_MODEL_REPO = "krut42/voice-lid-ecapa-voxlingua107"
_MODEL_FILE = "lid-ecapa-voxlingua107.int8.onnx"
_LABEL_FILE = "lid-labels.txt"
_HOLOLIVE_FOREIGN = frozenset(("en", "id"))


def radio_band_limited(audio: np.ndarray) -> bool:
    """Detect telephone/game-radio speech with almost no energy above 4 kHz."""
    clip = np.asarray(audio, dtype=np.float32)
    if clip.size < SAMPLE_RATE // 4:
        return False
    power = np.abs(np.fft.rfft((clip - clip.mean()) * np.hanning(clip.size))) ** 2
    freqs = np.fft.rfftfreq(clip.size, 1 / SAMPLE_RATE)
    speech = float(power[(freqs >= 300) & (freqs < 7500)].sum())
    if speech <= 1e-6 * clip.size:
        return False
    return float(power[(freqs >= 4000) & (freqs < 7500)].sum()) / speech < 0.02


@dataclass(frozen=True)
class LanguageDecision:
    language: str
    probability: float
    japanese_probability: float
    drop: bool
    reason: str


class LanguageFilter:
    """CPU-only ECAPA spoken-language ID, without another ASR/translator."""

    def __init__(self, cfg: LanguageFilterConfig):
        from huggingface_hub import hf_hub_download
        import onnxruntime as ort

        self.cfg = cfg
        self.retry_at = 0.0
        log.info(
            "Initializing CPU-only VoxLingua107 language filter (first run "
            "downloads a ~27 MB INT8 ONNX model and its language labels)"
        )
        model_path = hf_hub_download(repo_id=_MODEL_REPO, filename=_MODEL_FILE)
        label_path = hf_hub_download(repo_id=_MODEL_REPO, filename=_LABEL_FILE)
        with open(label_path, "r", encoding="utf-8-sig") as file:
            self.labels = [line.strip().lower() for line in file if line.strip()]
        if len(self.labels) != 107 or "ja" not in self.labels or not {"en", "id"}.issubset(self.labels):
            raise RuntimeError("VoxLingua107 label set is invalid or incomplete")

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        model_inputs = {item.name for item in self.session.get_inputs()}
        model_outputs = {item.name for item in self.session.get_outputs()}
        if "wav" not in model_inputs or "logp" not in model_outputs:
            raise RuntimeError("Unexpected VoxLingua107 ONNX input/output schema")
        self.japanese_idx = self.labels.index("ja")
        log.info("CPU VoxLingua107 language filter ready; preserving Japanese and uncertain speech")

    def _predict(self, audio: np.ndarray) -> tuple[str, float, float, float]:
        # The model takes raw mono 16k float32 PCM and internally extracts FBANKs.
        x = np.ascontiguousarray(audio, dtype=np.float32).reshape(1, -1)
        logp = np.asarray(
            self.session.run(["logp"], {"wav": x})[0], dtype=np.float32
        ).reshape(-1)
        if logp.shape != (len(self.labels),) or not np.isfinite(logp).all():
            raise RuntimeError("VoxLingua107 returned invalid class probabilities")
        # Output is log-softmax, so exp gives normalized model probabilities.
        probs = np.exp(logp)
        top = int(np.argmax(probs))
        second = float(np.partition(probs, -2)[-2])
        return (
            self.labels[top],
            float(probs[top]),
            float(probs[self.japanese_idx]),
            float(probs[top]) - second,
        )

    def classify(self, audio: np.ndarray) -> LanguageDecision:
        if time.monotonic() < self.retry_at:
            return LanguageDecision("unknown", 0., 0., False, "unavailable")
        samples = len(audio)
        duration = samples / SAMPLE_RATE
        if duration < self.cfg.min_duration_s:
            return LanguageDecision("unknown", 0., 0., False, "too_short")
        if not np.isfinite(audio).all():
            return LanguageDecision("unknown", 0., 0., False, "invalid_audio")

        # Inspect both ends of long VAD utterances; mixed Japanese/English in a
        # single long segment must NOT be dropped solely on its dominant language.
        window = max(1, int(self.cfg.max_window_s * SAMPLE_RATE))
        if samples >= int(self.cfg.min_duration_s * 2 * SAMPLE_RATE):
            # Inspect independent beginning/end halves even for medium 5-8s
            # cues, where a speaker may switch English -> Japanese mid-phrase.
            half_window = min(window, samples // 2)
            windows = [audio[:half_window], audio[-half_window:]]
        else:
            windows = [audio[:window]]

        try:
            predicted = [self._predict(clip) for clip in windows]
        except Exception as exc:
            self.retry_at = time.monotonic() + 5.0
            log.warning(
                "Language filter retry in 5s after CPU inference error; "
                "preserving original Japanese pipeline: %s", exc,
            )
            return LanguageDecision("unknown", 0., 0., False, "inference_error")

        # Require ALL windows to be strongly non-Japanese, and only when
        # classifying non-Japanese speech. This avoids
        # suppressing Japanese speech inside code-switched utterances.
        # Band-limited radio Japanese is often misread as an arbitrary
        # language, so radio audio may only be dropped as English/Indonesian.
        radio = radio_band_limited(audio)
        strong = [
            (lang != "ja" and lang != "unknown"
             and (lang in _HOLOLIVE_FOREIGN or not radio)
             and confidence >= self.cfg.drop_probability
             and japanese <= self.cfg.max_japanese_probability
             and margin >= self.cfg.min_top_margin)
            for lang, confidence, japanese, margin in predicted
        ]
        first = predicted[0]
        if all(strong):
            return LanguageDecision(
                first[0], min(p[1] for p in predicted),
                max(p[2] for p in predicted), True, "confident_non_japanese",
            )
        return LanguageDecision(
            first[0], first[1],
            max(p[2] for p in predicted), False, "japanese_or_uncertain",
        )
