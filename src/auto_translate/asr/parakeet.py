import logging
from pathlib import Path

import numpy as np

from .. import cuda_dlls
from ..config import AsrConfig

log = logging.getLogger(__name__)


class ParakeetASR:
    def __init__(self, cfg: AsrConfig) -> None:
        self.cfg = cfg

        # Make CUDA DLLs from installed NVIDIA wheels available.
        cuda_dlls.setup()

        import sherpa_onnx

        model_dir = Path(cfg.parakeet_model)
        model = model_dir / "model.int8.onnx"
        tokens = model_dir / "tokens.txt"

        if not model.is_file() or not tokens.is_file():
            raise FileNotFoundError(
                f"Parakeet model files not found in: {model_dir}"
            )

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=str(model),
            tokens=str(tokens),
            num_threads=cfg.parakeet_threads,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            provider="cuda",
            debug=True,
        )

        log.info("Parakeet ASR initialized: %s", model_dir)

    def transcribe(self, audio: np.ndarray, probe: bool = False) -> str:
        audio = np.ascontiguousarray(audio, dtype=np.float32)

        stream = self.recognizer.create_stream()
        stream.accept_waveform(16000, audio)
        self.recognizer.decode_stream(stream)

        text = stream.result.text.strip()

        if not text:
            return ""

        if any(phrase in text for phrase in self.cfg.blocklist):
            log.debug("Discarding blocked Parakeet text: %s", text)
            return ""

        return text