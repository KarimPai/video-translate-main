"""Replay identical clips through this checkout (or a pinned baseline worktree)."""
import argparse
import hashlib
import importlib.metadata
import itertools
import json
import queue
import subprocess
import sys
import time
import unicodedata
import wave
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np


def wav(path):
    with wave.open(str(path), "rb") as src:
        if (src.getframerate(), src.getnchannels(), src.getsampwidth()) != (16000, 1, 2):
            raise ValueError(f"{path} requires 16 kHz mono signed 16-bit PCM")
        return np.frombuffer(src.readframes(src.getnframes()), "<i2").astype("f") / 32768


def fixture():
    # Known speech intervals, a close pair, brief reply and sustained speech.
    p = np.full(3800, .02)
    for start, end in [(100, 154), (205, 225), (302, 375), (420, 502), (605, 3410)]:
        p[start:end] = .9
    phase = np.arange(len(p) * 160) * (2 * np.pi * 220 / 16000)
    return (np.sin(phase) * np.repeat(p, 160) * .08).astype("f"), p


class RecordedScores:
    def __init__(self, scores):
        self.scores = scores

    def reset(self):
        self.offset = 0

    def __call__(self, audio):
        start = self.offset
        self.offset += len(audio) // 160
        return self.scores[start:self.offset]


def character_error(reference, prediction):
    def chars(text):
        return [c for c in unicodedata.normalize("NFKC", text)
                if not c.isspace() and not unicodedata.category(c).startswith("P")]
    a, b = chars(reference), chars(prediction)
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(row[j] + 1, new[-1] + 1, row[j-1] + (x != y)))
        row = new
    return row[-1] / len(a) if a else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest", type=Path)
    inputs.add_argument("--synthetic", action="store_true")
    parser.add_argument("--checkout", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--baseline", action="store_true", help="use checkout defaults only")
    parser.add_argument("--japanese", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.checkout.resolve() / "src"))
    from auto_translate import cuda_dlls
    cuda_dlls.setup()
    from auto_translate.config import Config
    from auto_translate.types import AudioChunk
    from auto_translate.vad import Segmenter
    from auto_translate.pipeline import run_whisper
    from auto_translate.whisper_v3 import WhisperLargeV3

    revision = subprocess.check_output(
        ["git", "-C", str(args.checkout), "rev-parse", "HEAD"], text=True).strip()
    cfg = Config()
    model = splitter = None
    if args.synthetic:
        clips = [{"wav": "synthetic"}]
    else:
        model = WhisperLargeV3(cfg.whisper)
        if hasattr(cfg, "speaker") and cfg.speaker.enabled:
            from auto_translate.speaker import SherpaSplitter
            splitter = SherpaSplitter(cfg.speaker)
        clips = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    profiles = [cfg.vad]
    if not args.baseline:
        for pad, silence, hard in itertools.product((80, 300), (400, 500), (12, 16, 20)):
            profiles.append(replace(cfg.vad, speech_pad_ms=pad, min_silence_ms=silence,
                                    max_utterance_s=hard - cfg.vad.max_split_extension_s))

    with args.output.open("w", encoding="utf-8") as out:
        for clip in clips:
            scores = None
            if args.synthetic:
                audio, scores = fixture()
            else:
                audio = wav(args.manifest.parent / clip["wav"])
            cfg.whisper.hololive_detected_names = clip.get("names", [])
            digest = hashlib.sha256(audio.tobytes()).hexdigest()
            for profile in profiles:
                mocked = (patch("auto_translate.vad.StreamingFireRedVAD",
                                return_value=RecordedScores(scores))
                          if scores is not None else nullcontext())
                with mocked:
                    segmenter = Segmenter(profile)
                record = {"revision": revision, "clip": clip["wav"], "pcm_sha256": digest,
                          "pad_ms": profile.speech_pad_ms, "silence_ms": profile.min_silence_ms,
                          "hard_s": profile.max_utterance_s + profile.max_split_extension_s,
                          "sherpa": splitter is not None, "segments": [], "cues": []}
                started = time.perf_counter()
                source, sink = queue.Queue(), queue.Queue()
                ready = {}
                parts = []
                for idx in range(0, len(audio), 1600):
                    chunk = audio[idx:idx+1600]
                    window_end = (idx + len(chunk)) / 16000
                    items = segmenter.feed(AudioChunk(idx, chunk, time.monotonic()))
                    if idx + len(chunk) == len(audio):
                        items.extend(segmenter.flush())
                    for item in items:
                        for part in splitter.split(item) if splitter else [item]:
                            record["segments"].append([part.start_s, part.end_s, window_end])
                            ready[part.start_s] = window_end
                            source.put(part)
                            parts.append(part)
                source.put(None)
                if model:
                    run_whisper(model, source, sink)
                    while (cue := sink.get()) is not None:
                        record["cues"].append({
                            "start": cue.start_s, "end": cue.end_s, "en": cue.en,
                            "endpoint_plus_decode_s": ready[cue.start_s] - cue.end_s
                            + cue.timings.asr_end - cue.timings.asr_start})
                record["processing_s"] = time.perf_counter() - started
                record["audio_s"] = len(audio) / 16000
                if model:
                    record["real_time_factor"] = record["processing_s"] / record["audio_s"]
                    record["versions"] = {name: importlib.metadata.version(name)
                                          for name in ("faster-whisper", "ctranslate2", "fireredvad")}
                    try:
                        value = subprocess.check_output(
                            ["nvidia-smi", "-i", "0", "--query-gpu=memory.used",
                             "--format=csv,noheader,nounits"], text=True, timeout=3)
                        record["device_used_mib_after_decode"] = int(value.strip())
                    except (OSError, ValueError, subprocess.SubprocessError):
                        record["device_used_mib_after_decode"] = None
                    if clip.get("kind") == "non-speech":
                        record["non_speech_subtitles"] = len(record["cues"])
                    if clip.get("kind") == "utterance":
                        record["utterance_omitted"] = not bool(record["cues"])
                    if "en" in clip:
                        record["reference_en"] = clip["en"]
                    if args.japanese:
                        ja = []
                        for part in parts:
                            segs, _ = model._model.transcribe(
                                part.audio, language="ja", task="transcribe", beam_size=5,
                                temperature=0., condition_on_previous_text=False, vad_filter=False)
                            ja.extend(s.text for s in segs)
                        record["diagnostic_ja"] = "".join(ja)
                        if clip.get("ja"):
                            record["cer"] = character_error(clip["ja"], record["diagnostic_ja"])
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()


if __name__ == "__main__":
    main()
