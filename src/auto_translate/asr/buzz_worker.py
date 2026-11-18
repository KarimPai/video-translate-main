"""Isolated CUDA interpreter for the pinned Japanese BuzzASR checkpoint."""
import base64
import json
import sys

import numpy as np


def main():
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    path, device = sys.argv[1:]
    processor = WhisperProcessor.from_pretrained(path, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.float16, local_files_only=True
    ).to(device).eval()
    print(json.dumps({"ready": True}), flush=True)
    with torch.inference_mode():
        for line in sys.stdin.buffer:
            audio = np.frombuffer(base64.b64decode(line), dtype="<f4")
            pieces = []
            # Each short-form Whisper input covers at most 30 seconds.
            for offset in range(0, len(audio), 30 * 16000):
                chunk = audio[offset:offset + 30 * 16000]
                features = processor(chunk, sampling_rate=16000, return_tensors="pt").input_features
                ids = model.generate(features.to(device, dtype=torch.float16), num_beams=1,
                                     no_repeat_ngram_size=3, repetition_penalty=1.2)
                pieces.extend(processor.batch_decode(ids, skip_special_tokens=True))
            print(json.dumps({"text": "".join(pieces)}), flush=True)


if __name__ == "__main__":
    main()
