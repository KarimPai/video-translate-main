"""Isolated CUDA interpreter for the pinned Japanese BuzzASR checkpoint."""
import base64
import json
import os
import sys
import traceback

import numpy as np


def main():
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    path, device = sys.argv[1:]
    processor = WhisperProcessor.from_pretrained(path, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        path, dtype=torch.float16, local_files_only=True, attn_implementation="sdpa"
    ).to(device).eval()

    def decode(audio, **options):
        batch = processor(audio, sampling_rate=16000, return_attention_mask=True,
                          return_tensors="pt")
        ids = model.generate(batch.input_features.to(device, dtype=torch.float16),
                             attention_mask=batch.attention_mask.to(device),
                             num_beams=1, no_repeat_ngram_size=3, repetition_penalty=1.2,
                             **options)
        return processor.batch_decode(ids, skip_special_tokens=True)

    # Avoid automatic cuDNN/memory-efficient attention dispatch on Windows CUDA.
    # Flash SDPA keeps encoder workspace small; math covers unsupported shapes.
    with torch.inference_mode(), sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
        # Allocate encoder/decoder workspace before the parent budgets MT offload.
        # Do not empty the CUDA allocator cache after this warmup.
        decode(np.zeros(16000, dtype=np.float32), min_new_tokens=8, max_new_tokens=8)
        torch.cuda.synchronize(device)
        print(json.dumps({"ready": True}), flush=True)
        for line in sys.stdin.buffer:
            audio = np.frombuffer(base64.b64decode(line), dtype="<f4")
            pieces = []
            # Each short-form Whisper input covers at most 30 seconds.
            for offset in range(0, len(audio), 30 * 16000):
                chunk = audio[offset:offset + 30 * 16000]
                pieces.extend(decode(chunk))
            print(json.dumps({"text": "".join(pieces)}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # A poisoned CUDA context can hang during interpreter teardown. Report
        # the cause before exiting without invoking CUDA destructors.
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
