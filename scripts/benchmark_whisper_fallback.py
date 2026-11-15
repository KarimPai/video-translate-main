"""Compare single-temperature and fallback Japanese decodes on local annotated clips.

Usage: python scripts/benchmark_whisper_fallback.py manifest.csv local_model_dir
CSV: clip,reference_ja,tags. Include speech, silence (empty reference),
repeated speech and previously hallucinated audio. Results are JSON lines.
"""
import argparse
import csv
import json
import time
from importlib.metadata import version
from pathlib import Path

from benchmark_quality import distance
from auto_translate.asr.whisper_turbo import _decode
from auto_translate.cuda_dlls import setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error("provide an existing converted Whisper model directory")
    setup()
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
    import ctranslate2
    model = WhisperModel(str(args.model), device="cuda", compute_type="int8_float16",
                         cpu_threads=4, num_workers=1, local_files_only=True)
    with args.manifest.open(encoding="utf-8-sig", newline="") as source:
        for index, row in enumerate(csv.DictReader(source)):
            audio = decode_audio(str(args.manifest.parent / row["clip"]), sampling_rate=16000)
            # Alternate order to distribute warm-cache effects.
            configs = [0.0, (0.0, 0.2)]
            for temperature in configs[::(-1 if index % 2 else 1)]:
                ctranslate2.set_random_seed(0)
                started = time.perf_counter()
                text = _decode(model, audio, 6, temperature=temperature)
                elapsed = time.perf_counter() - started
                reference = row["reference_ja"]
                print(json.dumps(dict(clip=row["clip"], tags=row.get("tags"), reference=reference,
                                      temperature=temperature, output=text, seconds=elapsed,
                                      char_edits=distance(reference, text), reference_chars=len(reference),
                                      false_speech=bool(text and not reference),
                                      faster_whisper=version("faster-whisper")), ensure_ascii=False))


if __name__ == "__main__":
    main()
