---
sessionId: session-261119-234738-1m4p
---

# Technical Design

### Current Implementation

**Pipeline Architecture** (`pipeline.py`):
- Threaded pipeline: Ingest → VAD → ASR → Translation → Output
- Bounded queues: audio(64), utterance(16), text(16), subtitle(16)
- Generation-based seek invalidation
- `Stats` class tracks ASR/MT time and deadline misses
- Fragment merging for incomplete sentences
- **Queue operations are blocking** (`queue.Queue` with `maxsize`). Producers block on full queues — no "drop oldest" behavior exists.
- `AudioProgress` flows through VAD → ASR → translate → main loop → `output.advance()`. VAD's `advance_progress` emits progress based on what audio cannot belong to a future utterance (VAD state), not translation completion.

**Translation** (`translate/gemma4.py`):
- Gemma 4 E4B via llama.cpp CUDA server
- Reasoning: enabled, budget hardcoded as `max_tokens // 2` (256)
- Context: 6 lines, 600 chars max, with `useful_context` filtering
- Retry logic for empty outputs, missing names
- `_chat()` returns `(content, reasoning_content)` separately
- System prompt for professional subtitle translation

**ASR** (`asr/haas.py`):
- Haas-ja-v1.0 via speech.cpp HTTP server (Vulkan)
- Measures total HTTP request time, not server-side breakdown
- Context echo detection and recovery
- Generation-based cancellation

**VOD Sync** (`output/mpv.py`):
- `PAUSE_LEAD_SECONDS = 5.0` - pauses at start until translation reaches 5s lead
- **No re-pause** after initial unpause (line 6: "Once started, playback is uninterrupted")
- `_translation_ready()` checks `_translated_until >= _now + PAUSE_LEAD_SECONDS`
- `reset_buffer()` resets after seek
- `_translated_until` tracks audio position safely processed
- **Current watermark advances via `advance()` called from pipeline main loop on `AudioProgress` items** — these come from VAD's `advance_progress`, which advances based on VAD state (what audio cannot belong to a future utterance). During silence, VAD advances the watermark because idle frames are confirmed speech-free. During pending VAD decisions (buffered preroll), the watermark does NOT advance — it stops at `_start_sample` (the beginning of the current speech region).

**VAD Progress** (`vad.py`):
- `advance_progress()` / `release_progress()`: advances watermark only past audio that cannot belong to a future utterance
- During speech (`_in_speech`): `safe_sample = _start_sample` — watermark stops at speech onset
- During silence: `safe_sample = _next_frame_sample - preroll * FRAME` — watermark advances past confirmed silence
- Fallback probing: if enabled, watermark stops at `_idle_start` to allow ASR probe of idle audio
- **Key insight**: VAD progress already correctly distinguishes silence (advance) from pending speech (hold). The watermark issue is that it advances based on VAD decisions, not translation completion.

**SafePlayback** (`safe_playback.py`):
- `SafeInferenceGate`: bounded semaphore controlling concurrent inference operations
- `operation()` context manager: acquires slot, checks cancellation/generation/deadline
- **Does NOT block audio ingestion or VAD** — only gates ASR and translation model calls
- `max_inference_concurrency=1` by default (only one model request at a time)
- **Critical**: SafePlayback is an admission gate, not a flow controller. It limits concurrent model calls but does not prevent the translation worker from draining its queue.

**Configuration** (`config.py`):
- `TranslateConfig`: max_tokens=512, context_size=4096, context_lines=6
- `SafePlaybackConfig`: enabled=false, queue_size=4, audio_chunks=32
- No `reasoning_budget` field - derived in code
- No VOD sync parameters

**Diagnostics** (`runtime_diagnostics.py`):
- `RuntimeDiagnostics`: bounded JSONL log, metrics snapshot
- Tracks: gpu, stages, queues, counters, workers
- Counters: dropped, expired, pending_subtitles, latency_s
- No separate tracking of queue wait vs inference time
- No reasoning token count tracking

**Data Types** (`types.py`):
- `Timings`: audio_read, segmented, asr_start/end, mt_start/end
- `Subtitle`: start_s, end_s, ja, en, timings, generation
- No fields for queue wait times or reasoning tokens

### Key Decisions

1. **Reasoning Budget**: Make configurable via `config.toml` (`translate.reasoning_budget`). **Default: 256 (reasoning enabled)**. Benchmark 128, 192, and 256 with identical samples. **Do not automatically select the fastest configuration** — translation quality takes priority. Keep reasoning enabled at 256 unless benchmarking demonstrates that a lower budget preserves quality.

2. **VOD Sync (Hybrid)**: Combine MPV pause/resume with audio pacing:
   - Calculate `lead = translated_until - playback_position`
   - **Pause MPV when lead falls below 2s** (not `_now - _translated_until`)
   - **Resume MPV when lead reaches at least 8s** (hysteresis to prevent oscillation)
   - **User manual pause/play controls remain authoritative** — do not override user-initiated pauses
   - Slow audio ingestion when queue near capacity (mild backpressure)
   - **Keep ingestion and translation workers running** when MPV is paused for translation
   - **No pause for live streams**, only VOD

3. **Translation Watermark (Contiguous, Speech-Aware)**: The `_translated_until` watermark advances **only** across contiguous audio regions where:
   - Speech has been transcribed AND translated (subtitle published), OR
   - Lack of speech has been confirmed by VAD (silence gaps where VAD advanced progress)
   - **Confirmed silence advances the watermark** — VAD's `advance_progress` already correctly advances past idle frames that cannot belong to a future utterance. This is the correct behavior: silence IS a form of confirmed "no speech" and should count toward translation coverage.
   - **Pending VAD decisions do NOT advance** — during active speech regions, VAD stops the watermark at speech onset. This is already correct.
   - **Untranslated speech must NOT advance** — the watermark should advance to a subtitle's `end_s` only after that subtitle is published, not when VAD marks the region as processed. **This is the core fix**: separate VAD progress (audio processed by VAD) from translation progress (audio whose speech has been translated).
   - **EOF advances** — at end of stream, all remaining audio is flushed and the watermark advances to EOF.
   - **Music**: VAD treats music as non-speech (below threshold). If music is continuous, VAD advances the watermark through it as silence. This is correct: music contains no speech to translate.
   - **Long silence (60+ seconds)**: VAD advances the watermark continuously. No deadlock — no utterances are produced, so no translation is needed. The watermark simply tracks that this region has been confirmed speech-free.

4. **No Drop on Queue Full**: **Never drop valid VOD audio or utterances because a queue is full.** The existing bounded queues use blocking `put()` which applies backpressure — producers wait. This is correct behavior. **Remove any proposed "drop oldest if full" behavior.** Only discard:
   - Stale seek generations (generation mismatch)
   - Explicitly cancelled work (stop event, deadline expiry)
   - Expired live cues (playback position far past cue end)
   - Quality-guarded rejections (empty ASR, hallucinated outros)

5. **SafePlayback Deadlock Prevention**: SafePlayback must not create deadlocks with adaptive MPV pausing:
   - **SafePlayback is an admission gate** — it limits concurrent model calls via a semaphore. It does NOT block audio ingestion, VAD, or the translation worker's queue consumption.
   - **MPV auto-pause stops playback** — it does NOT stop any worker thread. Ingest, VAD, ASR, and translation continue running.
   - **No circular dependency**: MPV pause → workers continue → translation catches up → watermark advances → MPV resume.
   - **SafePlayback should NOT gate subtitle delivery** — the `show()` call on output is not wrapped in a SafePlayback operation. Subtitles flow freely from the pipeline to MPV.
   - **Audit**: The translation worker calls `gate.operation("mt", ...)` around the `translator.translate()` call. If the gate is full, the worker blocks on the semaphore. This is fine — the worker is waiting for a slot, not deadlocked. The ASR worker similarly blocks on `gate.operation("asr", ...)`. Neither blocks the other stage's queue consumption.

6. **Genuinely Contiguous Translation Watermark**: The watermark advances only across regions that are **contiguous from the start position**:
   - Region A (silence, confirmed by VAD): watermark advances through A
   - Region B (speech, translated): watermark advances to B's end_s after subtitle published
   - Region C (silence, confirmed by VAD): watermark advances through C (contiguous with B)
   - Region D (speech, NOT yet translated): watermark STOPS at D's onset. Even if VAD has processed D, the watermark does not advance past D until translation completes.
   - **Gap handling**: If there's an unresolved gap (e.g., VAD hasn't decided yet, or translation failed), the watermark stops before the gap. It resumes advancing once the gap is resolved.
   - **Implementation**: Track `_translated_until` separately from VAD progress. Advance `_translated_until` to `subtitle.end_s` when a subtitle is published. Advance through VAD-confirmed silence only if the silence is contiguous with the current watermark position (no untranslated speech between them).

7. **Haas Stall Investigation**: Add **client-side** timing instrumentation to distinguish:
   - Audio preparation/transfer (before HTTP request)
   - HTTP request duration (total round-trip time)
   - **Do not report server-side timing** (Vulkan inference, queue waiting) as known values — speech.cpp does not expose this via its API
   - Track GPU utilization during Haas requests to **infer** contention
   - Compare: Haas alone vs with Gemma vs during sustained playback

8. **Context Optimization**: Preserve existing 4096-token context. Inject only relevant glossary entries. **Do not implement custom prompt caching** unless profiling demonstrates a measurable bottleneck. Evaluate existing llama.cpp prefix reuse first.

9. **Parallel Inference**: Benchmark np=1 vs np=2 only after other fixes. **Preserve 4096 tokens per slot** during parallel benchmarking — do NOT reduce each slot to 2048. For two slots, allocate sufficient total context (e.g., `-c 8192` or use `-c 4096` per slot if the llama.cpp version supports it). Measure actual concurrent throughput. Enable only if sustained throughput improves without degradation.

10. **User Pause/Play Independence**: Test user pause/play and seeking independently of automatic translation pauses:
    - Track `_user_paused` flag separately from auto-pause state
    - Auto-pause should never override a user-initiated play
    - User-initiated pause should be respected — do not auto-resume
    - After seek, reset all pause state and require fresh translation lead

### Proposed Changes

#### Task 1: Reasoning Optimization
- Add `reasoning_budget` field to `TranslateConfig` (**default: 256**, reasoning enabled)
- Update `Gemma4Translator._args()` to use configurable budget
- When `reasoning_budget == 0`: set `--reasoning off` and omit `--reasoning-budget`
- Add reasoning token tracking in `_chat()` and `Timings`
- Implement benchmark script for configurations (128, 192, 256)
- **Do not automatically select the fastest configuration** — quality takes priority
- Log reasoning/final token split when available from API

#### Task 2: Context Preservation
- Inspect current context selection in `Gemma4Translator._context()` and `_window()`
- Ensure 3-5 relevant exchanges preserved
- Clear context after seek (already done via generation reset)
- Inject only relevant glossary entries (already done via `_terms_for()`)
- **Do not implement custom prompt caching** unless profiling demonstrates bottleneck
- Evaluate existing llama.cpp prefix reuse first

#### Task 3: VOD Synchronization (Hybrid)
- Add VOD sync config to `MpvConfig`: `vod_pause_threshold_s: float = 2.0`, `vod_resume_threshold_s: float = 8.0`
- **Fix pause condition**: calculate `lead = translated_until - playback_position`
- **Pause MPV when lead < 2s** (not when `_now - _translated_until > threshold`)
- **Resume MPV when lead >= 8s** (hysteresis to prevent oscillation)
- **User manual pause/play controls remain authoritative** — check `_user_paused` before auto-pausing
- **Keep ingestion and translation workers running** when MPV is paused for translation
- **Translation watermark**: advance `_translated_until` only across contiguous regions where speech has been translated OR silence confirmed by VAD
- **Confirmed silence advances the watermark** — this is correct and prevents deadlock during long silence
- **Untranslated speech blocks the watermark** — even if VAD processed the region, don't advance past it until translation completes
- **EOF advances the watermark** — at end of stream, flush all remaining
- Implement audio pacing in `ingest.py` when queue near capacity
- Reset state correctly after seeking
- Prevent stale subtitles from previous generations
- **No pause for live streams**, only VOD

#### Task 4: Queue Backpressure (No Drop)
- **Never drop valid VOD audio or utterances due to full queues**
- Existing bounded queues with blocking `put()` are correct — they apply backpressure
- **Remove any "drop oldest if full" behavior** from the plan
- Enable `safe_playback` with tuned parameters
- Add queue wait time tracking to `Timings`
- Instrument queue-wait times separately from model inference
- Ensure correct cancellation after seeking (generation-based)
- Prevent completed stale requests from consuming output capacity
- **Preserve chronological order** — bounded queues maintain FIFO order
- Only discard: stale generations, cancelled work, expired live cues, quality rejections

#### Task 5: Haas Stall Investigation
- Add **client-side** timing instrumentation to `SpeechCppServer.transcribe()`:
  - `audio_prep_s`: time to prepare audio payload (encoding, etc.)
  - `http_request_s`: total HTTP round-trip time (request sent to response received)
- **Do not claim to measure server-side Vulkan inference time** — speech.cpp does not expose this
- File description: "Client-side timing for Haas HTTP transcription"
- Track GPU utilization during Haas requests to **infer** contention
- Compare: Haas alone vs with Gemma vs during sustained playback
- Log timing breakdown per utterance

#### Task 6: Parallel Gemma Benchmarking
- Add `parallel_slots` to `TranslateConfig` (default: 1)
- Update `Gemma4Translator._args()` to use `-np` from config
- **Preserve 4096 tokens per slot** — do NOT reduce context per slot
- For np=2: increase total context allocation (e.g., `-c 8192`) where supported
- **Preserve identical per-request translation settings** across np=1 and np=2
- Benchmark np=1 vs np=2 with sustained dialogue
- Measure: throughput, latency, VRAM, quality, subtitle ordering

#### Task 7: Diagnostic Reporting
- Add queue wait fields to `Timings`
- Track reasoning token counts in diagnostics
- Fix deadline miss reporting (currently reports 0 when latency > 10s)
- Report: average, median, p95, max latency
- Separate: processing latency, queue delay, playback buffering
- Log reasoning token counts when API provides usage metadata

#### Task 8: Translation Quality
- Test with real HoloGTA dialogue samples
- Run identical inputs through reasoning configurations (128, 192, 256)
- Evaluate: semantic accuracy, naturalness, context, nicknames, profanity
- Preserve representative Japanese/English examples
- Compare outputs across configurations
- **Quality takes priority over speed** — do not select lower budget just for speed

#### Task 9: Regression Tests
- **60+ seconds of silence**: verify watermark advances, no deadlock, no missing subtitles
- **Continuous dialogue**: verify no backlog growth, stable latency, no expired subtitles
- **Full queues**: verify backpressure works, no dropped utterances, no deadlock
- **Failed translation**: verify retry logic, no stuck pipeline, graceful degradation
- **Rapid seeking**: verify generation reset, no stale subtitles, clean state
- **User pause/play**: verify independent of auto-pause, workers continue during pause
- **Clean shutdown**: verify all child processes terminate, no hanging threads
- Check for: deadlocks, missing subtitles, duplicated subtitles, expired subtitles

#### Task 10: Validation
- Run existing test suite after each change
- Test sustained playback (10+ minutes), seeking, and shutdown
- Validate all child processes terminate
- Measure peak VRAM with all components active
- Confirm reasoning never leaks to subtitles
- Rebuild EXE via PyInstaller
- Commit and push to `feature/haas-gemma4-reasoning`

### File Structure

**Modified files:**
- `src/auto_translate/config.py` - Add `reasoning_budget`, VOD sync params, parallel slots
- `src/auto_translate/types.py` - Add queue wait and reasoning token fields to `Timings`
- `src/auto_translate/translate/gemma4.py` - Use configurable reasoning budget, track tokens
- `src/auto_translate/output/mpv.py` - Adaptive VOD pause/resume, watermark logic, user pause tracking
- `src/auto_translate/ingest.py` - Audio pacing when queue near capacity
- `src/auto_translate/pipeline.py` - Queue wait tracking, improved Stats, watermark advancement on subtitle publish
- `src/auto_translate/speech_server.py` - **Client-side** timing for audio prep and HTTP round-trip
- `src/auto_translate/asr/haas.py` - Enhanced timing breakdown
- `src/auto_translate/runtime_diagnostics.py` - Enhanced metrics tracking
- `config.toml` - Add new configuration options

**New files:**
- `scripts/benchmark_reasoning.py` - Benchmark script for reasoning configurations
- `tests/test_vod_sync.py` - Tests for adaptive VOD synchronization
- `tests/test_reasoning_config.py` - Tests for reasoning budget configuration
- `tests/test_pipeline_regressions.py` - Regression tests for silence, continuous dialogue, full queues, seeking, shutdown

### Architecture Diagram

```mermaid
graph LR
    A[MPV Player] -->|Audio| B[Ingest + Pacing]
    B -->|AudioChunk| C[VAD]
    C -->|Utterance| D[Haas ASR Vulkan]
    D -->|Transcript| E[Gemma Translation CUDA]
    E -->|Subtitle| F[Pipeline Main]
    F -->|show()| G[MPV Output]
    G -->|Display| A
    G -->|lead = translated_until - pos| H{Sync Gate}
    H -->|lead < 2s: pause| A
    H -->|lead >= 8s: resume| A
    H -->|user pause: ignore| A
    B -->|queue full: block| B
    I[SafePlayback] -->|admission gate| D
    I -->|admission gate| E
    H -.->|orthogonal| I
    F -->|advance watermark| G
    C -->|AudioProgress| D
    D -->|AudioProgress| E
    E -->|AudioProgress| F
    F -->|advance()| G
```

### Risks

1. **VOD Pause Oscillation**: Hysteresis thresholds (2s/8s) may need tuning. Too aggressive pause causes choppy playback; too lenient causes missing subtitles.
2. **GPU Contention**: Haas Vulkan and Gemma CUDA may compete for GPU resources, causing intermittent stalls. Parallel inference may worsen this.
3. **Reasoning Quality**: Reducing reasoning budget may degrade translation quality for complex dialogue. Benchmarking required.
4. **Audio Pacing Artifacts**: Slowing audio ingestion may cause audio buffer underruns or artifacts if not implemented carefully.
5. **Backward Compatibility**: New config options should have sensible defaults to preserve existing behavior.
6. **Watermark Contiguity**: Tracking contiguous translation coverage requires careful handling of gaps between VAD progress and subtitle publication. The watermark must not skip over untranslated speech even if VAD has processed it.
7. **Blocking Queue Deadlock**: Bounded queues with blocking puts could deadlock if a downstream worker dies. The existing shutdown logic drains queues when a consumer dies, preventing permanent blockage.

# Delivery Steps

### ✓ Step 1: 1. Reasoning budget configuration and token tracking
**DONE**

### ✓ Step 2: 2. Adaptive VOD synchronization (hybrid pause + pacing) with contiguous watermark
**DONE**
Make reasoning budget configurable and add token tracking infrastructure.

**Config changes** (`config.py`):
- Add `reasoning_budget: int = 256` to `TranslateConfig` (0 = off, 128/192/256 = budget)
- **Default: 256 (reasoning enabled)**. Do not change default to 0 or any other value.
- Add `parallel_slots: int = 1` to `TranslateConfig`
- Add validation: `0 <= reasoning_budget <= max_tokens`

**Types changes** (`types.py`):
- Add to `Timings`: `asr_queue_wait: float = 0.0`, `mt_queue_wait: float = 0.0`, `reasoning_tokens: int = 0`, `final_tokens: int = 0`

**Translator changes** (`translate/gemma4.py`):
- Update `_args()`: use `self.cfg.reasoning_budget` instead of `max_tokens // 2`
- When `reasoning_budget == 0`: set `--reasoning off` and omit `--reasoning-budget`
- Update `_chat()`: extract and log `usage` metadata (reasoning_tokens, completion_tokens) if available
- Update `translate()`: record token counts in `Timings`
- Update server startup log to show reasoning budget

**Diagnostics** (`runtime_diagnostics.py`):
- Add counters: `mt_reasoning_tokens`, `mt_final_tokens`, `mt_reasoning_overflow`
- Record token counts in `translation_complete` events

**Tests**: Update `test_translation_optimizations.py` to verify reasoning config.

###   Step 2: 2. Adaptive VOD synchronization (hybrid pause + pacing) with contiguous watermark
Implement hybrid VOD sync: adaptive MPV pause/resume with hysteresis + audio pacing + genuinely contiguous watermark.

**Config changes** (`config.py`):
- Add to `MpvConfig`: `vod_pause_threshold_s: float = 2.0`, `vod_resume_threshold_s: float = 8.0`

**MPV changes** (`output/mpv.py`):
- **CRITICAL: Fix pause condition**. Calculate `lead = _translated_until - _now` (playback position).
  - **Pause MPV when `lead < 2.0`** (not when `_now - _translated_until > threshold` which was inverted)
  - **Resume MPV when `lead >= 8.0`** (hysteresis to prevent oscillation)
- **User manual pause/play controls remain authoritative**: track `_user_paused` separately
  - Auto-pause never overrides user-initiated play
  - User pause is respected — no auto-resume
  - After seek, reset all pause state
- **Keep ingestion and translation workers running** when MPV is paused for translation
- **Contiguous translation watermark**: advance `_translated_until` only across contiguous regions where:
  - Speech has been transcribed AND translated (subtitle published), OR
  - Lack of speech has been confirmed by VAD (silence gaps)
  - **Confirmed silence advances the watermark** — VAD-confirmed idle frames count as translation coverage
  - **Untranslated speech blocks the watermark** — even if VAD processed the region, don't advance past it
  - **EOF advances** — at end of stream, flush all remaining
- Add `_vod_should_pause()` method: check `lead` vs pause threshold
- Add `_vod_should_resume()` method: check `lead` vs resume threshold
- Modify scheduler loop (lines 1036-1100):
  - After initial unpause, continue checking translation lead each iteration
  - Log pause/resume transitions
- Add `_last_pause_reason` to distinguish: translation_lag, user_pause, seek
- Update `reset_buffer()`: clear pause state after seek
- **No pause for live streams**, only VOD

**Pipeline changes** (`pipeline.py`):
- **Advance watermark on subtitle publish**: when a subtitle is published, advance `_translated_until` to cover the region up to `subtitle.end_s` (including any contiguous VAD-confirmed silence between the current watermark and the subtitle)
- **Two-track watermark**: maintain `_vad_progress` (from VAD AudioProgress) and `_translated_until` (translation coverage). The watermark used for VOD sync is `_translated_until`.
- Record queue wait times in `Timings` before each operation
- **SafePlayback integration**: SafePlayback is an admission gate (semaphore) for model calls. It does NOT block audio ingestion, VAD, or subtitle delivery. No circular dependency with MPV pause.

**Ingest changes** (`ingest.py`):
- Audio pacing when downstream queue > 75% capacity: sleep briefly to slow ingestion
- Log pacing events

**Deadlock prevention**:
- **Silence/music**: VAD advances `_vad_progress`, watermark advances through confirmed silence. No utterances produced, no translation needed. No deadlock.
- **Seeking**: reset generation, cancel stale requests, clear both watermarks
- **Audio backpressure**: bounded queues with **blocking** puts (backpressure). **No drop on full**.
- **User pause**: keep all workers running, respect user control as authoritative
- **SafePlayback + MPV pause**: orthogonal controllers. SafePlayback gates model calls; MPV pause stops playback. No circular wait.

**Tests**: Add `tests/test_vod_sync.py` with mock MPV and translation coverage.

### ✓ Step 3: 3. Haas stall investigation and client-side timing instrumentation
Add **client-side** timing instrumentation to diagnose Haas Vulkan stalls.

**Speech server changes** (`speech_server.py`):
- Add timing to `transcribe()` method: track `audio_prep_s` and `http_request_s`
- **audio_prep_s**: time to encode/prepare audio payload before HTTP request
- **http_request_s**: total HTTP round-trip time (request sent to response received)
- **Do NOT report server-side timing** (Vulkan inference, queue waiting) as known values — speech.cpp does not expose this
- File/module description: "Client-side timing for Haas HTTP transcription"

**ASR changes** (`asr/haas.py`):
- Update `_decode()` to use timing-aware transcription
- Record timing breakdown in `Timings`: `asr_queue_wait`, `asr_http_s`
- **Do not add `asr_inference_s`** — we cannot measure server-side inference time
- Add `asr_rtf` (real-time factor) per utterance
- Log timing breakdown for utterances > 2s (potential stalls)

**GPU monitoring** (`gpu_budget.py`):
- Add sampling during ASR inference to detect GPU contention
- Track: GPU utilization, VRAM usage during Haas vs Gemma activity
- Log concurrent GPU activity (Haas Vulkan + Gemma CUDA)
- **Infer** contention from GPU metrics, do not report as measured server-side timing

**Diagnostics**:
- Add `asr_timing` events with breakdown
- Add counters: `asr_stall_count` (utterances > 2s), `asr_queue_wait_total`
- Record GPU metrics at time of ASR requests

**Benchmark script** (`scripts/benchmark_haas.py`):
- Run Haas alone: measure baseline inference time
- Run Haas + Gemma simultaneously: measure contention
- Run during sustained MPV playback: measure real-world behavior
- Compare timing breakdowns

### ✓ Step 4: 4. Queue backpressure (no drop) and diagnostic reporting fixes
Fix queue management and correct latency/diagnostic reporting.

**Queue backpressure**:
- **Never drop valid VOD audio or utterances due to full queues**
- Existing bounded queues with blocking `put()` are correct — they apply backpressure
- **No "drop oldest if full" behavior** — remove any such proposal
- Only discard: stale generations, cancelled work, expired live cues, quality rejections
- **Preserve chronological order** — bounded queues maintain FIFO order

**Pipeline changes** (`pipeline.py`):
- Record queue wait times: measure time between item production and consumption
- Add to `Stats`: track queue wait separately from inference
- Fix deadline miss reporting: use actual latency vs configurable deadline
- Add p95 latency calculation to `Stats.summary()`
- Add `expired_subtitles` counter (separate from `dropped`)
- Ensure generation-based cancellation propagates correctly
- Prevent completed stale requests from consuming output capacity

**Diagnostics changes** (`runtime_diagnostics.py`):
- Add to metrics: `queue_wait`, `processing_latency`, `playback_buffering`
- Add counters: `subtitle_expired`, `subtitle_late`, `queue_full_event`
- Record `subtitle_latency` events with full breakdown
- Add `latency_percentiles` to snapshot (p50, p95, p99)

**Config changes** (`config.py`):
- Enable `safe_playback` by default for VOD: `enabled = true`
- Tune: `queue_size = 8`, `audio_chunks = 48`, `max_audio_lead_s = 30.0`

**Tests**: Update `test_pipeline_stability.py` and `test_safe_playback_regressions.py`

### * Step 5: 5. Context optimization and parallel inference benchmark
Optimize translation context and benchmark parallel Gemma inference.

**Context optimization** (`translate/gemma4.py`):
- Inspect `_context()` and `_window()` for efficiency
- Ensure 3-5 relevant exchanges preserved (current: 6 lines, 600 chars)
- Verify context clearing after seek (already via generation reset)
- Verify only relevant glossary entries injected (already via `_terms_for()`)
- **Do NOT implement custom prompt caching** unless profiling demonstrates a measurable bottleneck
- Evaluate existing llama.cpp prefix reuse first

**Parallel inference** (`translate/gemma4.py`):
- Update `_args()`: use `self.cfg.parallel_slots` for `-np` flag
- **Preserve 4096 tokens per slot** — do NOT reduce context per slot
- For np=2: **increase total context allocation** (e.g., `-c 8192`) where supported by llama.cpp
- **Preserve identical per-request translation settings** across np=1 and np=2
- Add VRAM headroom check before enabling parallel slots

**Benchmark script** (`scripts/benchmark_parallel.py`):
- Run sustained dialogue with np=1: measure throughput, latency, VRAM
- Run same with np=2: measure same metrics
- Compare: translations/sec, per-request latency, peak VRAM, quality
- Test subtitle ordering with concurrent requests
- Measure impact on Haas Vulkan performance

**Config changes** (`config.py`):
- Add `translate.parallel_slots: int = 1` (default single slot)
- Add validation: `1 <= parallel_slots <= 2`

**Tests**: Add tests for parallel slot configuration

###   Step 6: 6. Translation quality benchmark, regression tests, and validation
Benchmark translation quality, add regression tests, and validate all changes.

**Benchmark script** (`scripts/benchmark_reasoning.py`):
- Load Japanese dialogue samples (HoloGTA-style: slang, nicknames, profanity, context)
- Run through each reasoning config: **128, 192, 256** (not 0 — reasoning is enabled by default)
- **Do not automatically select the fastest configuration** — quality takes priority
- Measure: reasoning tokens, final tokens, latency, empty responses
- Compare outputs: semantic accuracy, naturalness, nickname preservation
- Generate comparison report with Japanese/English pairs
- Evaluate: hallucinations, missing content, censorship

**Regression tests** (`tests/test_pipeline_regressions.py`):
- **60+ seconds of silence**: verify watermark advances through silence, no deadlock, no missing subtitles
- **Continuous dialogue**: verify no backlog growth, stable latency, no expired subtitles
- **Full queues**: verify backpressure (blocking put), no dropped utterances, no deadlock
- **Failed translation**: verify retry logic, no stuck pipeline, graceful degradation
- **Rapid seeking**: verify generation reset, no stale subtitles, clean watermark state
- **User pause/play**: verify independent of auto-pause, workers continue during user pause
- **Clean shutdown**: verify all child processes terminate, no hanging threads
- Check for: deadlocks, missing subtitles, duplicated subtitles, expired subtitles

**Quality tests**:
- Test with: normal conversation, fast exchanges, shouting, slang, nicknames, long context, short acks
- Verify no reasoning text in subtitles
- Verify profanity preservation
- Verify nickname preservation (Lap, Miko, Mikochi)
- Verify no gender assumptions
- Verify no invented dialogue
- **Quality takes priority over speed** — do not select lower budget just for speed

**Full validation**:
- Run complete existing test suite: `pytest tests/`
- Fix regressions rather than disabling tests
- Test sustained playback (10+ minutes of continuous dialogue)
- Test seeking repeatedly at different timestamps
- Confirm all child processes terminate on close
- Measure peak VRAM with Haas + Gemma + MPV + GUI active
- Confirm reasoning never leaks to subtitles
- Rebuild `VideoTranslate.exe` using PyInstaller config
- Commit and push to `feature/haas-gemma4-reasoning`

**Final report**:
- Root causes identified
- Files changed
- Reasoning-budget benchmark results
- Chosen reasoning configuration and justification
- Haas stall investigation results
- Single vs dual slot comparison
- Translation quality comparisons
- Average and p95 end-to-end latency
- Peak GPU memory usage
- Queue saturation measurements
- Expired/missing subtitle counts
- Test results
- EXE build verification
- Git commit hash and branch URL
- Unresolved issues
