"""Silero VAD segmenter: AudioChunk stream to timestamped ASR utterances.

Loads the Silero VAD ONNX model on first use and keeps its LSTM recurrent
state across calls so it works on a live stream.

VAD-negative probes are disabled by default to avoid hallucinating speech
from music, laughter and game sound effects. Optional probe mode is retained.
"""

# Re-export VAD interface types from base.py
from .base import VADBackend, VADSegment

__all__ = ["VADBackend", "VADSegment"]

import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable

import numpy as np

from ..config import VadConfig
from ..types import SAMPLE_RATE, AudioChunk, AudioProgress, SourceEOF, Utterance

log = logging.getLogger(__name__)

FRAME = 512  # Silero VAD uses a 32 ms frame shift (16 kHz)


class StreamingSileroVAD:
    """Silero VAD ONNX adapter returning one speech probability per 32 ms."""

    def __init__(self, speech_threshold: float, device: str = "cpu", model_dir: str = "") -> None:
        from silero_vad_lite import SileroVAD

        if device not in ("cpu",):
            raise ValueError("SileroVAD device must be cpu")

        self._model = SileroVAD(SAMPLE_RATE)
        self.device = "CPU"
        log.info("Silero VAD initialized on %s", self.device)

    def reset(self) -> None:
        """Reset LSTM state at a VOD seek without reloading model weights."""
        self._model.reset()

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Accept float32 mono audio and return one raw speech probability per 32 ms."""
        # Process audio in 512-sample (32ms) chunks, preserving LSTM state.
        probs: list[float] = []
        for i in range(0, len(audio), FRAME):
            chunk = audio[i:i + FRAME]
            if len(chunk) < FRAME:
                # Pad short final chunk with silence.
                chunk = np.pad(chunk, (0, FRAME - len(chunk)), mode='constant')
            prob = self._model.process(chunk)
            probs.append(float(prob))
        return np.asarray(probs, dtype=np.float32)


def _ms_to_frames(ms: float) -> int:
    return max(1, round(ms * SAMPLE_RATE / 1000 / FRAME))


def _select_longform_cut(
    frames: list[np.ndarray],
    probs: list[float],
    *,
    search_frames: int,
    pause_frames: int,
    context_frames: int,
    negative_threshold: float,
    hard: bool,
) -> int | None:
    """Prefer sustained low-VAD/low-energy pauses; never discard or repeat PCM.

    A long utterance may extend briefly beyond its soft cap to find a pause.
    On the hard cap, choose the least energetic 70ms neighborhood instead.
    The returned cut always leaves context on both sides in the source audio.
    """
    count = len(frames)
    lo = max(context_frames, count - search_frames)
    hi = count - context_frames
    if lo >= hi:
        # Even a tiny custom max_utterance_s must have a finite hard limit.
        return max(1, count // 2) if hard and count >= 2 else None
    # Only the search window, percentile lookback and convolution halo
    # contribute to the cut. Keep offsets in the original frame timeline.
    offset = max(0, lo - 20)
    window = np.asarray(frames[offset:hi + 3], dtype=np.float32)
    rms = np.sqrt(np.mean(window * window, axis=1))
    probs = probs[offset:hi + 3]
    lo -= offset
    hi -= offset
    baseline = float(np.percentile(rms[max(0, lo - 20):hi], 75))
    if baseline > 1e-5:
        quiet = ((np.asarray(probs, dtype=np.float32) < negative_threshold)
                 & (rms <= max(0.0004, baseline * 0.42)))
        runs: list[tuple[int, int]] = []
        start = None
        for i in range(lo, hi):
            if quiet[i]:
                if start is None:
                    start = i
            elif start is not None:
                if i - start >= pause_frames:
                    runs.append((start, i))
                start = None
        if start is not None and hi - start >= pause_frames:
            runs.append((start, hi))
        if runs:
            first, last = max(runs, key=lambda run: run[1])
            return offset + (first + last) // 2
    if not hard:
        return None

    # A full fallback is unavoidable if speech never pauses. Smooth across
    # multiple frames so a single misleading VAD dip cannot cut a word.
    smooth = np.convolve(rms, np.ones(7) / 7.0, mode="same")
    scale = max(1e-5, float(np.percentile(smooth[lo:hi], 75)))
    score = (np.asarray(probs[lo:hi], dtype=np.float32) * 0.5
             + np.minimum(smooth[lo:hi] / scale, 2) * 0.5)
    return offset + lo + int(np.argmin(score))


class Segmenter:
    """State machine over 32 ms frames: idle → speech → (pause | max length) → emit."""

    def __init__(self, cfg: VadConfig) -> None:
        self.cfg = cfg
        self.min_silence = _ms_to_frames(cfg.min_silence_ms)
        self.pad = _ms_to_frames(cfg.speech_pad_ms)
        self.min_speech = _ms_to_frames(cfg.min_speech_ms)
        self.short_reply_frames = _ms_to_frames(cfg.short_reply_ms)
        self.onset_frames = _ms_to_frames(cfg.onset_ms)
        self.prob_smooth_frames = max(1, cfg.prob_smooth_frames)
        # Dynamic threshold operates on existing CPU Silero probabilities:
        # no second VAD model, network call or GPU memory is required.
        self._dynamic_window_frames = _ms_to_frames(cfg.dynamic_window_s * 1000)
        self._dynamic_update_frames = _ms_to_frames(cfg.dynamic_update_s * 1000)
        self._dynamic_min_frames = min(
            self._dynamic_window_frames, _ms_to_frames(10_000),
        )
        self.max_frames = _ms_to_frames(cfg.max_utterance_s * 1000)
        self.hard_max_frames = self.max_frames + max(
            0, round(cfg.max_split_extension_s * SAMPLE_RATE / FRAME)
        )
        self.split_pause_frames = _ms_to_frames(cfg.split_pause_min_ms)
        self.adaptive_frames = (
            _ms_to_frames(cfg.adaptive_split_s * 1000)
            if cfg.adaptive_split and cfg.adaptive_split_s > 0 else 0
        )
        self.adaptive_pause_frames = max(
            self.split_pause_frames, _ms_to_frames(cfg.adaptive_pause_ms)
        )
        self.split_context_frames = _ms_to_frames(cfg.split_context_ms)
        self.split_search = _ms_to_frames(2000)  # look this far back for a quiet split point
        self.fallback_frames = _ms_to_frames(cfg.fallback_s * 1000) if cfg.fallback_s > 0 else 0
        self.fallback_min = _ms_to_frames(cfg.fallback_min_s * 1000)
        self.fallback_min_rms = 10 ** (cfg.fallback_min_db / 20)
        # Keep one Silero model instance for the lifetime of the segmenter.
        # A seek only resets recurrent state; it must not reload model weights.
        self.vad = StreamingSileroVAD(cfg.threshold, cfg.device, cfg.model_dir)
        self._reset(0, generation=0)
        if cfg.dynamic_threshold:
            log.info(
                "Dynamic Silero VAD enabled: baseline=%.2f/%.2f "
                "window=%.1fs update=%.1fs max_increase=%.2f",
                cfg.threshold, cfg.neg_threshold, cfg.dynamic_window_s,
                cfg.dynamic_update_s, cfg.dynamic_max_increase,
            )

    def _reset(self, next_sample: int, generation: int) -> None:
        """Fresh stream state, starting at sample `next_sample` (0 = take it from the first chunk)."""
        self._soft_limit = self.max_frames
        self._hard_limit = self.hard_max_frames
        if generation > 0:
            # Limit only the first utterance while rebuilding translation lead.
            self._soft_limit = min(self.max_frames, _ms_to_frames(5000))
            self._hard_limit = min(self.hard_max_frames, self._soft_limit + _ms_to_frames(1000))
        self._generation = generation
        self.vad.reset()
        self._leftover = np.zeros(0, dtype=np.float32)  # samples not yet a whole frame
        self._next_frame_sample = next_sample  # sample index of the next frame to process
        self._preroll: deque[np.ndarray] = deque(maxlen=self.pad + self.onset_frames)  # recent audio, including onset
        self._preroll_probs: deque[float] = deque(maxlen=self.pad + self.onset_frames)
        self._prob_history: deque[float] = deque(maxlen=self.prob_smooth_frames)
        # Reset learned background after each VOD seek / stream generation.
        self._speech_threshold = self.cfg.threshold
        self._exit_threshold = self.cfg.neg_threshold
        self._dynamic_idle_probs: deque[float] = deque(maxlen=self._dynamic_window_frames)
        self._dynamic_idle_since_update = 0
        self._onset_count = 0
        self._idle: list[np.ndarray] = []  # non-speech frames since the last utterance (fallback)
        self._idle_start = next_sample
        self._frames: list[np.ndarray] = []  # current utterance
        self._probs: list[float] = []
        self._start_sample = 0
        self._speech_start_sample = 0
        self._silence = 0  # consecutive silent frames at the end of the utterance
        self._in_speech = False
        self._continuing_speech = False
        self._read_at = 0.0
        self._read_end_sample = next_sample
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
        if self._next_frame_sample == 0 and not len(self._leftover):
            self._next_frame_sample = chunk.start_sample  # VOD may start mid-video
            self._idle_start = chunk.start_sample
        audio = np.concatenate([self._leftover, chunk.pcm])
        self._read_end_sample = self._next_frame_sample + len(audio)
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
        if len(self._leftover):
            frame = self._leftover
            self._leftover = np.zeros(0, dtype=np.float32)
            padded = np.pad(frame, (0, FRAME - len(frame)))
            self._step(frame, float(self.vad(padded)[0]))
            self._next_frame_sample += len(frame)
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
            safe_sample = self._next_frame_sample - sum(map(len, self._preroll))

        # Idle frames may still produce a fallback ASR probe.
        if not force and not self._in_speech and self.fallback_frames and self._idle:
            safe_sample = min(safe_sample, self._idle_start)

        target = min(self._pending_progress, safe_sample)

        if target <= self._last_progress_sample:
            return None

        self._last_progress_sample = target
        return AudioProgress(target, generation=self._generation)


    def _observe_idle_probability(self, prob: float) -> None:
        """Raise/lower a bounded threshold using sustained background evidence.

        Ignore active utterances, tentative speech onsets and speech-positive
        frames to avoid learning the speaker's voice as background noise.
        Only eligible idle audio counts toward the update interval. Keep
        the original onset/exit hysteresis distance when the gate adapts.
        """
        cfg = self.cfg
        if (not cfg.dynamic_threshold or self._in_speech or self._onset_count
                or not np.isfinite(prob) or prob >= self._speech_threshold):
            return
        self._dynamic_idle_probs.append(prob)
        self._dynamic_idle_since_update += 1
        if self._dynamic_idle_since_update < self._dynamic_update_frames:
            return
        self._dynamic_idle_since_update = 0
        if len(self._dynamic_idle_probs) < self._dynamic_min_frames:
            return

        background = float(np.quantile(self._dynamic_idle_probs, cfg.dynamic_quantile))
        target = min(
            min(1.0, cfg.threshold + cfg.dynamic_max_increase),
            max(cfg.threshold, background + cfg.dynamic_noise_margin),
        )
        # Smooth updates, with a maximum 0.02 probability change each time.
        change = max(-0.02, min(0.02, (target - self._speech_threshold) * 0.25))
        new_threshold = max(cfg.threshold, min(1.0, self._speech_threshold + change))
        if abs(new_threshold - self._speech_threshold) > 0.0001:
            self._speech_threshold = new_threshold
            self._exit_threshold = min(1.0, cfg.neg_threshold + new_threshold - cfg.threshold)
            log.info(
                "Dynamic VAD threshold gen=%d onset=%.3f exit=%.3f background=%.3f",
                self._generation, self._speech_threshold, self._exit_threshold, background,
            )

    def _step(self, frame: np.ndarray, prob: float) -> None:
        # The upstream postprocessor's smoothed flags are not used by this
        # project: smooth raw probabilities here before deciding on utterances.
        self._prob_history.append(prob)
        prob = sum(self._prob_history) / len(self._prob_history)
        if not self._in_speech:
            self._observe_idle_probability(prob)
            previous_preroll = len(self._preroll)
            self._preroll.append(frame)
            self._preroll_probs.append(prob)
            self._onset_count = self._onset_count + 1 if prob >= self._speech_threshold else 0
            if self._onset_count < self.onset_frames:
                self._add_idle(frame)
                return
            # Only sustained speech opens a segment. Keep the entire preroll
            # and all its confidence values instead of clipping its first word.
            self._emit_probe(len(self._idle) - previous_preroll)
            self._in_speech = True
            self._frames = list(self._preroll)
            self._probs = list(self._preroll_probs)
            self._start_sample = self._next_frame_sample - (len(self._frames) - 1) * FRAME
            self._speech_start_sample = max(
                self._start_sample,
                self._next_frame_sample - (self.onset_frames - 1) * FRAME,
            )
            self._silence = 0
            return

        self._frames.append(frame)
        self._probs.append(prob)
        threshold = self._exit_threshold if self._continuing_speech else self._speech_threshold
        if self._speech_start_sample is None and prob >= threshold:
            # A forced cut can leave only silence in the continuation. Its
            # next spoken onset belongs to the next positive frame, not the
            # earlier padded audio boundary retained for ASR context.
            self._speech_start_sample = self._next_frame_sample
        self._silence = self._silence + 1 if prob < self._exit_threshold else 0

        if self._silence >= self.min_silence:
            # Pause found: keep `pad` frames of trailing silence, the rest becomes preroll.
            retained_silence = min(self._silence, len(self._frames))
            n = min(len(self._frames), len(self._frames) - retained_silence + self.pad)
            self._emit(n, keep_rest=False)
        elif (self.adaptive_frames and len(self._frames) >= self.adaptive_frames
              and self._silence >= self.adaptive_pause_frames):
            # A real quiet gap is preferable to delaying live subtitles until
            # the maximum chunk length. Include its trailing silence once;
            # no PCM is repeated or discarded across consecutive cues.
            self._emit(len(self._frames), keep_rest=False)
        elif len(frame) == FRAME and len(self._frames) >= self._soft_limit:
            # Prefer a real sustained pause; extend at most 1s before
            # a smoothed-energy forced cut. Timestamped PCM remains contiguous.
            hard = len(self._frames) >= self._hard_limit
            if hard or len(self._frames) == self._soft_limit or len(self._frames) % 5 == 0:
                cut = _select_longform_cut(
                    self._frames, self._probs,
                    search_frames=self.split_search,
                    pause_frames=self.split_pause_frames,
                    context_frames=self.split_context_frames,
                    negative_threshold=self._exit_threshold,
                    hard=hard,
                )
                if cut is not None:
                    self._emit(cut, keep_rest=True)

    def _emit(self, n: int, keep_rest: bool) -> None:
        frames, probs = self._frames[:n], self._probs[:n]
        rest_frames, rest_probs = self._frames[n:], self._probs[n:]
        start = self._start_sample
        end = start + sum(map(len, frames))

        threshold = self._exit_threshold if self._continuing_speech else self._speech_threshold
        speech_probs = [p for p in probs if p >= threshold]
        # Keep brief, sustained high-probability replies without opening
        # the gate for isolated peaks or lowering the ordinary speech floor.
        confident_run = 0
        short_reply = False
        if not self._continuing_speech and 0 < len(speech_probs) < self.min_speech:
            for prob in probs:
                confident_run = confident_run + 1 if prob >= self.cfg.short_reply_probability else 0
                if confident_run >= self.short_reply_frames:
                    short_reply = True
                    break
        # A length cut does not end speech. An already accepted sequence can
        # continue below the onset threshold, including a short tail at EOF.
        # A continuation consisting entirely of silence still emits nothing.
        accepted = (len(speech_probs) >= self.min_speech or short_reply
                    or (self._continuing_speech and bool(speech_probs)))
        if accepted:
            utt = Utterance(
                start / SAMPLE_RATE,
                end / SAMPLE_RATE,
                np.concatenate(frames),
                generation=self._generation,
                speech_start_s=(self._speech_start_sample / SAMPLE_RATE
                                if self._speech_start_sample is not None else None),
                vad_confidence=sum(speech_probs) / len(speech_probs),
            )
            utt.timings.audio_read = self._read_at - (self._read_end_sample - end) / SAMPLE_RATE
            utt.timings.segmented = time.monotonic()
            self._out.append(utt)
            log.debug(
                "VAD utterance gen=%d %.2f-%.2fs speech_start=%.2f vad_conf=%.2f samples=%d",
                utt.generation, utt.start_s, utt.end_s,
                utt.speech_start_s or utt.start_s,
                utt.vad_confidence or 0.0,
                len(utt.audio),
            )
            self._soft_limit = self.max_frames
            self._hard_limit = self.hard_max_frames

        if keep_rest:
            self._continuing_speech |= accepted
            self._frames, self._probs = rest_frames, rest_probs
            self._start_sample = end
            threshold = self._exit_threshold if self._continuing_speech else self._speech_threshold
            first_positive = next(
                (i for i, p in enumerate(rest_probs) if p >= threshold), None,
            )
            self._speech_start_sample = (end + first_positive * FRAME
                                         if first_positive is not None else None)
        else:
            self._in_speech = False
            self._continuing_speech = False
            self._frames, self._probs = [], []
            self._preroll.clear()
            self._preroll.extend(rest_frames)  # deque(maxlen) keeps only the last `pad`
            self._preroll_probs.clear()
            self._preroll_probs.extend(rest_probs)
            self._onset_count = 0
            self._idle, self._idle_start = list(rest_frames), end

    def _add_idle(self, frame: np.ndarray) -> None:
        if not self.fallback_frames:
            return
        if not self._idle:
            self._idle_start = self._next_frame_sample
        self._idle.append(frame)
        # Bound the idle buffer: after emitting a probe, _idle is truncated.
        # But if _emit_probe skips (silence), frames still accumulate.
        # Cap at 2x fallback interval to prevent unbounded memory growth.
        max_idle = self.fallback_frames * 3
        if len(self._idle) >= max_idle:
            # Drop oldest frames and emit a probe from what remains
            excess = len(self._idle) - self.fallback_frames
            self._idle_start += sum(map(len, self._idle[:excess]))
            self._idle = self._idle[excess:]
        if len(self._idle) >= self.fallback_frames:
            # Cut at the quietest frame in the last ~2 s, so a word is less likely to be split.
            # The selected frame is included in the cut.
            lo = max(self.fallback_min - 1, len(self._idle) - self.split_search)
            rms = [float(np.sqrt(np.mean(f * f))) for f in self._idle[lo:]]
            self._emit_probe(lo + int(np.argmin(rms)) + 1)

    def _emit_probe(self, n: int) -> None:
        """Send idle frames to ASR only when fallback_s is explicitly enabled."""
        frames, self._idle = self._idle[:max(n, 0)], self._idle[max(n, 0):]
        start = self._idle_start
        self._idle_start += sum(map(len, frames))
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
        utt.timings.audio_read = self._read_at - (self._read_end_sample - end) / SAMPLE_RATE
        utt.timings.segmented = time.monotonic()
        self._out.append(utt)
        log.info(
            "VAD-negative probe gen=%d %.2f-%.2fs rms=%.4f fallback=%ss",
            utt.generation, utt.start_s, utt.end_s,
            float(np.sqrt(np.mean(audio * audio))),
            self.cfg.fallback_s,
        )


def run_segmenter(
    seg: Segmenter,
    in_q: "queue.Queue[AudioChunk | AudioProgress | SourceEOF | None]",
    out_q: "queue.Queue[Utterance | AudioProgress | SourceEOF | None]",
    get_generation: Callable[[], int] | None = None,
    stop: threading.Event | None = None,
    *,
    gate=None,
) -> None:
    """Thread body."""
    awaiting_first_utterance = False

    while True:
        item = in_q.get()

        if item is None or isinstance(item, SourceEOF):
            if isinstance(item, SourceEOF) and (
                    item.generation != seg._generation
                    or (get_generation is not None and item.generation != get_generation())):
                continue
            if stop is None or not stop.is_set():
                for utt in seg.flush():
                    out_q.put(utt)
                progress = seg.advance_progress(seg._next_frame_sample, force=True)
                if progress is not None:
                    out_q.put(progress)
            out_q.put(item)
            if item is None:
                break
            continue

        if stop is not None and stop.is_set():
            continue
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

        utterances = seg.feed(item)
        if ((stop is not None and stop.is_set())
                or (get_generation is not None and item.generation != get_generation())):
            continue
        for utt in utterances:
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
