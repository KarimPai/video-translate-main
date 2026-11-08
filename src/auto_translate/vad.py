"""FireRed VAD segmenter: AudioChunk stream → Utterance on each pause (or at max length).

Loads the FireRed streaming model from Hugging Face on first use and keeps its
recurrent model state across calls so it works on a live stream.

Audio rejected by VAD isn't necessarily discarded: every `fallback_s` it can send a
"probe" utterance to Qwen ASR, helping retain speech mixed with loud music or singing.
"""

import logging
import queue
import time
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np

from .config import VadConfig
from .types import SAMPLE_RATE, AudioChunk, AudioProgress, Utterance

log = logging.getLogger(__name__)

FRAME = 160  # FireRed streaming VAD uses a 10 ms frame shift (16 kHz)


class StreamingFireRedVAD:
    """FireRedStreamVad adapter returning one speech probability per 10 ms."""

    WINDOW_SAMPLES = 400  # 25 ms model window

    def __init__(self, speech_threshold: float) -> None:
        from huggingface_hub import snapshot_download
        import torch
        from fireredvad import FireRedStreamVad, FireRedStreamVadConfig

        if not torch.cuda.is_available():
            raise RuntimeError(
                "FireRedVAD requires CUDA, but torch.cuda.is_available() is false. "
                "Install a CUDA-enabled PyTorch build and verify the NVIDIA driver."
            )

        assets = Path(__file__).resolve().parents[2] / "assets" / "fireredvad"
        model_dir = assets / "Stream-VAD"
        if not (model_dir / "cmvn.ark").is_file() or not (model_dir / "model.pth.tar").is_file():
            log.info("Downloading FireRedVAD streaming model from Hugging Face...")
            snapshot_download(
                repo_id="FireRedTeam/FireRedVAD",
                allow_patterns=["Stream-VAD/*"],
                local_dir=str(assets),
            )

        config = FireRedStreamVadConfig(
            use_gpu=True,
            smooth_window_size=1,
            speech_threshold=speech_threshold,
            pad_start_frame=1,
            min_speech_frame=1,
            max_speech_frame=2000,
            min_silence_frame=1,
            chunk_max_frame=30000,
        )
        self._stream = FireRedStreamVad.from_pretrained(str(model_dir), config)
        self._history = np.zeros(self.WINDOW_SAMPLES - FRAME, dtype=np.int16)
        self.device = torch.cuda.get_device_name(0)
        log.info("FireRed VAD initialized on CUDA (%s)", self.device)

    def reset(self) -> None:
        """Reset streaming caches at a VOD seek without reloading model weights."""
        self._stream.reset()
        self._history.fill(0)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Accept float32 mono audio and return one raw speech probability per 10 ms."""
        pcm = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16, copy=False)
        # FireRed's detect_chunk batches all 10 ms frames into one model forward
        # pass. Keep the preceding 15 ms so each 25 ms fbank window is continuous
        # across AudioChunk boundaries.
        window = np.concatenate((self._history, pcm))
        results = self._stream.detect_chunk(window)
        self._history = window[-(self.WINDOW_SAMPLES - FRAME):].copy()
        probs = np.asarray([result.raw_prob for result in results], dtype=np.float32)
        expected = len(pcm) // FRAME
        if len(probs) != expected:
            raise RuntimeError(
                f"FireRed returned {len(probs)} frames for {expected} audio frames"
            )
        return probs


def _ms_to_frames(ms: float) -> int:
    return max(1, round(ms * SAMPLE_RATE / 1000 / FRAME))


class Segmenter:
    """State machine over 10 ms frames: idle → speech → (pause | max length) → emit."""

    def __init__(self, cfg: VadConfig) -> None:
        self.cfg = cfg
        self.min_silence = _ms_to_frames(cfg.min_silence_ms)
        self.pad = _ms_to_frames(cfg.speech_pad_ms)
        self.min_speech = _ms_to_frames(cfg.min_speech_ms)
        self.max_frames = _ms_to_frames(cfg.max_utterance_s * 1000)
        self.split_search = _ms_to_frames(2000)  # look this far back for a quiet split point
        self.fallback_frames = _ms_to_frames(cfg.fallback_s * 1000) if cfg.fallback_s > 0 else 0
        self.fallback_min = _ms_to_frames(cfg.fallback_min_s * 1000)
        self.fallback_min_rms = 10 ** (cfg.fallback_min_db / 20)
        # Keep one FireRed model instance for the lifetime of the segmenter.
        # A seek only resets recurrent state; it must not reload model weights.
        self.vad = StreamingFireRedVAD(cfg.threshold)
        self._reset(0, generation=0)

    def _reset(self, next_sample: int, generation: int) -> None:
        """Fresh stream state, starting at sample `next_sample` (0 = take it from the first chunk)."""
        self._generation = generation
        self.vad.reset()
        self._leftover = np.zeros(0, dtype=np.float32)  # samples not yet a whole frame
        self._next_frame_sample = next_sample  # sample index of the next frame to process
        self._preroll: deque[np.ndarray] = deque(maxlen=self.pad)  # recent non-speech frames
        self._idle: list[np.ndarray] = []  # non-speech frames since the last utterance (fallback)
        self._idle_start = next_sample
        self._frames: list[np.ndarray] = []  # current utterance
        self._probs: list[float] = []
        self._start_sample = 0
        self._speech_start_sample = 0
        self._silence = 0  # consecutive silent frames at the end of the utterance
        self._in_speech = False
        self._read_at = 0.0
        self._out: list[Utterance] = []
        self._pending_progress = next_sample
        self._last_progress_sample = next_sample

    def feed(self, chunk: AudioChunk) -> list[Utterance]:
        if chunk.reset:
            self._reset(chunk.start_sample, generation=chunk.generation)
            return []

        if chunk.generation != self._generation:
            return []
        self._read_at = chunk.read_at
        if self._next_frame_sample == 0:
            self._next_frame_sample = chunk.start_sample  # VOD may start mid-video
            self._idle_start = chunk.start_sample
        audio = np.concatenate([self._leftover, chunk.pcm])
        whole = len(audio) // FRAME * FRAME
        self._leftover = audio[whole:]
        if whole:
            probs = self.vad(audio[:whole])
            for i, prob in enumerate(probs):
                self._step(audio[i * FRAME:(i + 1) * FRAME], float(prob))
                self._next_frame_sample += FRAME
        out, self._out = self._out, []
        return out

    def flush(self) -> list[Utterance]:
        """End of stream: emit whatever speech is in progress."""
        if self._in_speech:
            self._emit(len(self._frames), keep_rest=False)
        self._emit_probe(len(self._idle))
        out, self._out = self._out, []
        return out

    def advance_progress(self, sample: int, *, force: bool = False) -> AudioProgress | None:
        """Advance only past audio that cannot belong to a future utterance."""
        self._pending_progress = max(self._pending_progress, sample)
        return self.release_progress(force=force)

    def release_progress(self, force: bool = False) -> AudioProgress | None:
        if force:
            safe_sample = self._next_frame_sample
        elif self._in_speech:
            safe_sample = self._start_sample
        else:
            safe_sample = self._next_frame_sample - len(self._preroll) * FRAME

        # Idle frames may still produce a fallback ASR probe.
        if not force and not self._in_speech and self.fallback_frames and self._idle:
            safe_sample = min(safe_sample, self._idle_start)

        target = min(self._pending_progress, safe_sample)

        if target <= self._last_progress_sample:
            return None

        self._last_progress_sample = target
        return AudioProgress(target, generation=self._generation)


    def _step(self, frame: np.ndarray, prob: float) -> None:
        cfg = self.cfg
        if not self._in_speech:
            if prob >= cfg.threshold:
                # Idle audio before the preroll goes to Qwen ASR; the preroll joins the utterance.
                self._emit_probe(len(self._idle) - len(self._preroll))
                self._in_speech = True
                self._frames = [*self._preroll, frame]
                self._probs = [0.0] * len(self._preroll) + [prob]
                self._start_sample = self._next_frame_sample - len(self._preroll) * FRAME
                # Keep pre-roll in the ASR audio, but timestamp subtitles from
                # the first frame that actually crossed the speech threshold.
                self._speech_start_sample = self._next_frame_sample
                self._silence = 0
            else:
                self._preroll.append(frame)
                self._add_idle(frame)
            return

        self._frames.append(frame)
        self._probs.append(prob)
        self._silence = self._silence + 1 if prob < cfg.neg_threshold else 0

        if self._silence >= self.min_silence:
            # Pause found: keep `pad` frames of trailing silence, the rest becomes preroll.
            n = min(len(self._frames), len(self._frames) - self._silence + self.pad)
            self._emit(n, keep_rest=False)
        elif len(self._frames) >= self.max_frames:
            # Too long: split at the quietest frame in the last ~2 s, keep the rest going.
            lo = len(self._frames) - self.split_search
            cut = lo + int(np.argmin(self._probs[lo:])) + 1
            self._emit(cut, keep_rest=True)

    def _emit(self, n: int, keep_rest: bool) -> None:
        frames, probs = self._frames[:n], self._probs[:n]
        rest_frames, rest_probs = self._frames[n:], self._probs[n:]
        start = self._start_sample
        end = start + n * FRAME

        speech_frames = sum(p >= self.cfg.threshold for p in probs)
        if speech_frames >= self.min_speech:
            utt = Utterance(
                start / SAMPLE_RATE,
                end / SAMPLE_RATE,
                np.concatenate(frames),
                generation=self._generation,
                speech_start_s=self._speech_start_sample / SAMPLE_RATE,
            )
            # Estimate when the utterance's last sample arrived: the current chunk's read time,
            # minus the audio processed after it (the trailing silence we waited for).
            processed_end = self._next_frame_sample + FRAME
            utt.timings.audio_read = self._read_at - (processed_end - end) / SAMPLE_RATE
            utt.timings.segmented = time.monotonic()
            self._out.append(utt)

        if keep_rest:
            self._frames, self._probs = rest_frames, rest_probs
            self._start_sample = end
            self._speech_start_sample = end + next(
                (i for i, p in enumerate(rest_probs) if p >= self.cfg.threshold),
                0,
            ) * FRAME
        else:
            self._in_speech = False
            self._frames, self._probs = [], []
            self._preroll.clear()
            self._preroll.extend(rest_frames)  # deque(maxlen) keeps only the last `pad`
            self._idle, self._idle_start = list(rest_frames), end

    def _add_idle(self, frame: np.ndarray) -> None:
        if not self.fallback_frames:
            return
        if not self._idle:
            self._idle_start = self._next_frame_sample
        self._idle.append(frame)
        if len(self._idle) >= self.fallback_frames:
            # Cut at the quietest frame in the last ~2 s, so a word is less likely to be split.
            lo = max(self.fallback_min, len(self._idle) - self.split_search)
            rms = [float(np.sqrt(np.mean(f * f))) for f in self._idle[lo:]]
            self._emit_probe(lo + int(np.argmin(rms)) + 1)

    def _emit_probe(self, n: int) -> None:
        """Send the first `n` idle frames to Qwen ASR as a probe (if long and loud enough)."""
        frames, self._idle = self._idle[:max(n, 0)], self._idle[max(n, 0):]
        start = self._idle_start
        self._idle_start += len(frames) * FRAME
        if not self.fallback_frames or len(frames) < self.fallback_min:
            return
        audio = np.concatenate(frames)
        if np.sqrt(np.mean(audio * audio)) < self.fallback_min_rms:
            return  # (near) silence
        end = start + len(audio)
        utt = Utterance(
            start / SAMPLE_RATE,
            end / SAMPLE_RATE,
            audio,
            probe=True,
            generation=self._generation,
        )
        utt.timings.audio_read = self._read_at - (self._next_frame_sample + FRAME - end) / SAMPLE_RATE
        utt.timings.segmented = time.monotonic()
        self._out.append(utt)


def run_segmenter(
    cfg: VadConfig,
    in_q: "queue.Queue[AudioChunk | AudioProgress | None]",
    out_q: "queue.Queue[Utterance | AudioProgress | None]",
    get_generation: Callable[[], int] | None = None,
) -> None:
    """Thread body."""
    try:
        seg = Segmenter(cfg)
    except Exception:
        log.exception("Could not initialize the FireRed VAD segmenter")
        out_q.put(None)
        return

    awaiting_first_utterance = False

    while True:
        item = in_q.get()

        if item is None:
            for utt in seg.flush():
                out_q.put(utt)

            progress = seg.advance_progress(seg._next_frame_sample, force=True)

            if progress is not None:
                out_q.put(progress)

            out_q.put(None)
            break

        if get_generation is not None and item.generation != get_generation():
            continue

        if isinstance(item, AudioProgress):
            if item.generation != seg._generation:
                continue

            progress = seg.advance_progress(item.sample)
            if progress is not None:
                out_q.put(progress)
            continue

        if item.reset:
            log.info(
                "VAD seek reset: generation=%d position=%.2fs",
                item.generation,
                item.start_sample / SAMPLE_RATE,
            )
            seg.feed(item)
            awaiting_first_utterance = True
            out_q.put(
                AudioProgress(
                    item.start_sample,
                    reset=True,
                    generation=item.generation,
                )
            )
            continue

        for utt in seg.feed(item):
            if awaiting_first_utterance:
                log.info(
                    "first VAD utterance after seek: generation=%d range=%.2f-%.2fs",
                    utt.generation,
                    utt.start_s,
                    utt.end_s,
                )
                awaiting_first_utterance = False
            out_q.put(utt)

        progress = seg.advance_progress(seg._next_frame_sample)
        if progress is not None:
            out_q.put(progress)
