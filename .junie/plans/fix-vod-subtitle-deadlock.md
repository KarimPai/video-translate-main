---
sessionId: session-261120-014511-q8y7
---

# Requirements

### Overview & Goals

Fix the VOD subtitle deadlock where translations work at the start of a VOD but progressively delay and then stop completely. The pipeline stalls because held fragments in `_join_fragments()` never release for VOD mode, while MPV auto-pauses and FFmpeg hits its 30-second read-ahead cap, creating a circular wait.

### Root Cause Analysis

**Primary cause: `_join_fragments()` has no release mechanism for VOD mode.**

In `pipeline.py` lines 378-386:
- **Live mode**: uses `_ticking()` with a `due()` callback that releases held fragments when playback approaches onset
- **VOD mode**: uses `items = iter(in_q.get, object())` — a blocking iterator with **no timeout and no `due` callback**

When a transcript ends without sentence-final punctuation, it's held indefinitely. No tick fires, no `due()` checks, so the transcript never reaches translation.

**Circular deadlock sequence:**
1. Fragment held in `_join_fragments()` waiting for continuation that never comes
2. AudioProgress markers parked behind held fragment
3. Watermark `_translated_until` doesn't advance past untranslated speech
4. MPV auto-pauses because `translation_lead < 0.5s` threshold
5. FFmpeg PCM reader hits 30s read-ahead cap (blocks because playback is paused)
6. No new audio → no new VAD utterances → no continuation fragment → held fragment stays forever
7. Pipeline stalls permanently

### Scope

**In Scope:**
- A. Fix `_join_fragments()` to release held VOD fragments via bounded waiting
- B. Fix playback/audio pacing deadlock between MPV auto-pause and FFmpeg read-ahead cap
- C. Add structured diagnostics for deadlock detection
- D. Add automatic recovery watchdog
- E. Regression tests for all deadlock scenarios

**Out of Scope:**
- Changing GPU layers, threads, model quantization, reasoning settings, or context size
- Modifying Haas ASR or Gemma translation logic
- Changing the English output guard behavior

### User Stories

1. As a viewer, I want subtitles to continue throughout VOD playback without permanent stalls
2. As a viewer, I want manual pause/play to remain authoritative — auto-pause should never override my intent
3. As a developer, I want structured diagnostics that distinguish deadlocks from legitimate silence

# Technical Design

### Current Implementation

**`_join_fragments()` (pipeline.py lines 128-167):**
- Holds transcripts without sentence-final punctuation
- For live mode: `_ticking()` yields `_TICK` every 0.25s; `due()` releases when playback nears onset
- For VOD mode: no ticking, no `due()` — held fragments wait forever

**`run_translate()` (pipeline.py lines 352-596):**
- Line 384: live mode uses `_ticking(in_q, MERGE_TICK_S)` with `due()` callback
- Line 386: VOD mode uses raw `items` — no ticking, no release

**MPV auto-pause (mpv.py lines 805-837):**
- Pauses when `translated_until - now < 0.5s`
- Resumes when `translated_until - now >= 3.0s`

**FFmpeg read-ahead (ingest.py lines 500-512):**
- Blocks when `sample / SAMPLE_RATE > playback_position + 30s`
- When MPV is paused, `playback_position` doesn't advance → FFmpeg blocks

### Key Decisions

1. **VOD fragment release uses ticking + progress-based due()**: Same pattern as live mode, but `due()` checks real-time wait timeout. A fragment is released when no continuation arrives within `vod_merge_timeout_s` (default 5s) of real-time.

2. **FFmpeg read-ahead cap relaxes during auto-pause**: When MPV is auto-paused (not manually paused), allow FFmpeg to continue reading up to 60s ahead to ensure audio is available for translation.

3. **Recovery watchdog**: Monitors forward progress. When work is pending but no subtitle published for 15s and player is auto-paused, force-release held fragments and resume playback.

### Proposed Changes

**A. Fix fragment merging (pipeline.py)**
- Change line 386 to use `_ticking()` with a VOD-aware `due()` callback
- `_vod_fragment_due()`: release when `time.monotonic() - held.timings.audio_read > vod_merge_timeout_s`
- Flush held fragment on SourceEOF immediately

**B. Fix playback/audio pacing (ingest.py)**
- In `_wait_for_vod_lead()`, double the lead cap to 60s during auto-pause
- Add `is_auto_paused` query to Ingest

**C. Diagnostics (pipeline.py, mpv.py)**
- Fragment age tracking with release reason logging
- Watermark positions every 10s: `_translated_until`, `_vad_progress`, pending speech
- Auto-pause/resume reasons with translation lead values

**D. Recovery watchdog (pipeline.py)**
- Stall detection: when `now - last_activity > 15s` with pending work and auto-pause, force-resume
- Idempotent and generation-safe

### File Structure

| File | Change |
|---|---|
| `src/auto_translate/pipeline.py` | VOD ticking + due(), EOF flush, recovery watchdog, diagnostics |
| `src/auto_translate/ingest.py` | Relax read-ahead cap during auto-pause |
| `src/auto_translate/output/mpv.py` | Enhanced auto-pause/resume logging |
| `tests/test_vod_deadlock.py` | New regression tests |
| `config.toml` / `config.example.toml` | Add `vod_merge_timeout_s` |
| `src/auto_translate/config.py` | Add `vod_merge_timeout_s` to TranslateConfig |

# Delivery Steps

### ✓ Step 1: Fix _join_fragments for VOD mode with ticking and due() release
Add bounded release mechanism for VOD fragment merging so held transcripts never wait indefinitely.
- Add `_ticking()` to VOD mode in `run_translate()` (line 386) so ticks fire every 0.25s
- Implement `_vod_fragment_due()` callback that releases held fragments when real-time wait exceeds `vod_merge_timeout_s` (default 5s)
- Flush held fragment on SourceEOF in `_join_fragments()` by checking for SourceEOF items
- Add `vod_merge_timeout_s` config option to TranslateConfig (default 5.0s)
- Update config.toml and config.example.toml with the new option
- Add fragment release reason logging (timeout, EOF, normal merge)

### ✓ Step 2: Fix FFmpeg read-ahead deadlock during MPV auto-pause
Prevent FFmpeg from blocking when MPV is auto-paused, breaking the circular deadlock.
- In `_wait_for_vod_lead()`, detect when MPV is auto-paused (not manually paused)
- Double the read-ahead cap from 30s to 60s during auto-pause to ensure audio continues flowing
- Add `is_auto_paused` property to Ingest that queries MPV pause state
- Preserve manual pause authority: never relax cap during manual pause

### ✓ Step 3: Add recovery watchdog and structured diagnostics
Detect and recover from deadlocks; add observability for debugging.
- Add stall detection in main loop: when no subtitle for 15s with pending work and auto-pause active, force-resume and flush
- Add fragment age logging: log held fragment timestamp range and age on each tick
- Add watermark diagnostics every 10s: _translated_until, _vad_progress, pending speech count, queue depths
- Log auto-pause/resume reasons with translation lead values
- Make recovery idempotent and generation-safe (no action across seeks)
- Rebuild executable and run test suite