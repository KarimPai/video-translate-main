"""Japanese-only ReazonSpeech k2 v2 (Zipformer RNN-T) using sherpa-onnx.

This branch replaces the Haas worker, not the existing VAD, translation,
MPV, timeline, or subtitle cache. ONNX inference runs on CPU by default,
leaving VRAM for FireRedVAD and the local PLaMo llama.cpp server.

Official model: https://huggingface.co/reazon-research/reazonspeech-k2-v2
Sherpa usage: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/zipformer-transducer-models.html
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..config import AsrConfig
from ..text_filter import rejection_reason
from ..types import SAMPLE_RATE

log = logging.getLogger(__name__)

# Official INT8 encoder and joiner + FP32 decoder combination.
_FILES = (
    "encoder-epoch-99-avg-1.int8.onnx",
    "decoder-epoch-99-avg-1.onnx",
    "joiner-epoch-99-avg-1.int8.onnx",
    "tokens.txt",
)
_MIN_BYTES = (120_000_000, 8_000_000, 1_000_000, 20_000)


def _ensure_models(cfg: AsrConfig) -> Path:
    """Download only the four required files; verify nontrivial file sizes."""
    destination = Path(cfg.model_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    missing = [
        name for name, minimum in zip(_FILES, _MIN_BYTES)
        if not (destination / name).is_file()
        or (destination / name).stat().st_size < minimum
    ]
    if missing:
        log.info(
            "Downloading ReazonSpeech k2 v2 INT8 model (%d missing files) "
            "from Hugging Face; one-time internet connection required",
            len(missing),
        )
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=cfg.model_id,
                revision=cfg.model_revision,
                allow_patterns=list(_FILES),
                local_dir=str(destination),
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not download ReazonSpeech k2 v2: " + str(exc)
                + ". Verify Hugging Face connectivity or manually place the"
                " official ONNX files in " + str(destination)
            ) from exc

    bad = [
        name for name, minimum in zip(_FILES, _MIN_BYTES)
        if not (destination / name).is_file()
        or (destination / name).stat().st_size < minimum
    ]
    if bad:
        raise RuntimeError(
            "Missing or partial ReazonSpeech model files in "
            + str(destination) + ": " + ", ".join(bad)
        )
    return destination


class ReazonK2ASR:
    """Reusable sherpa offline Zipformer recognizer with existing ASR API."""

    def __init__(
        self, cfg: AsrConfig, *, recognizer: object | None = None,
    ) -> None:
        self.cfg = cfg
        if cfg.provider != "cpu":
            raise ValueError(
                "ReazonSpeech branch uses provider='cpu' so GPU VRAM stays"
                " available for FireRedVAD and PLaMo."
            )
        if cfg.num_threads < 1:
            raise ValueError("ReazonSpeech num_threads must be >= 1")
        if cfg.max_audio_s <= 0 or cfg.max_audio_s > 30.0:
            raise ValueError("ReazonSpeech model supports at most ~30s per clip")

        if recognizer is not None:
            self._recognizer = recognizer
            return

        destination = _ensure_models(cfg)
        try:
            import sherpa_onnx

            self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(destination / _FILES[0]),
                decoder=str(destination / _FILES[1]),
                joiner=str(destination / _FILES[2]),
                tokens=str(destination / _FILES[3]),
                num_threads=cfg.num_threads,
                sample_rate=SAMPLE_RATE,
                feature_dim=80,
                decoding_method="greedy_search",
                provider="cpu",
            )
        except Exception as exc:
            raise RuntimeError(
                "ReazonSpeech k2 v2 failed to initialize via sherpa-onnx. "
                "Check that uv sync installed a working Windows x64 wheel "
                "and models were fully downloaded: " + str(exc)
            ) from exc

        log.info(
            "ReazonSpeech k2 v2 ASR ready (Zipformer INT8, CPU threads=%d, %s)",
            cfg.num_threads, destination,
        )

    def transcribe(self, audio: np.ndarray, probe: bool = False) -> str:
        """Return Japanese text; never call PLaMo for silence or bad PCM."""
        # Probes were designed for Haas and increase false-positive risk.
        # VAD-negative probe transcription is disabled for this experiment.
        if probe:
            return ""
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return ""
        if samples.size > int(self.cfg.max_audio_s * SAMPLE_RATE):
            raise ValueError(
                "ReazonSpeech audio segment exceeds max_audio_s; "
                "reduce VAD max_utterance_s"
            )
        if not np.isfinite(samples).all():
            raise ValueError("Non-finite ReazonSpeech PCM input")
        # Model input: contiguous 16-kHz mono float samples in [-1,1].
        samples = np.ascontiguousarray(np.clip(samples, -1.0, 1.0))
        if float(np.max(np.abs(samples))) < 1.0 / 32768:
            return ""

        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        self._recognizer.decode_stream(stream)
        result = getattr(stream, "result", None)
        text = (getattr(result, "text", "") or "").strip()
        if not text:
            return ""
        if any(phrase and phrase in text for phrase in self.cfg.blocklist):
            log.debug("ReazonSpeech blocklist match: %r", text)
            return ""
        reason = rejection_reason(text)
        if reason is not None:
            log.debug("ReazonSpeech unusable transcript (%s): %r", reason, text)
            return ""
        return text

    def close(self) -> None:
        """Release CPU ONNX sessions on shutdown."""
        self._recognizer = None
