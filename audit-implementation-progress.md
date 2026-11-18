# Audit Implementation Progress

## F01 — Qwen ASR prompt echo
- Status: Fixed (commit a967a29)
- Change: context_echo guard now checks full system prompt, not only cfg.context.
- Tests: tests/test_qwen3_asr.py 20/20 pass.

## F02 — Live metadata
- Status: Fixed (commit 9df82bf)
- Change: resolve_source now passes timeline to Source for live mode, preserving live_status, duration, upload_timestamp, video_id, etc. Downstream Holodex cast fetch and stream context now work for live sources.

## F03 — Short speech suppression
- Status: Investigated; no code change needed
- VAD short_reply logic, ASR quality guard (transcript_problem), and text_filter (rejection_reason) are all conservative and retain short reactions. Existing tests (test_anti_gibberish, test_drop_only_language_filter, test_single_asr_quality) pass.

## F04 — Retry/missing intervals
- Status: Fixed (commit 0da4bc8)
- Change: Bounded retries (max 2) for ASR transport failures (last_rejection == "error") in run_asr. Quality rejections not retried. Each attempt recorded as event with retries count.
- Test: test_vod_transport_failure_bounded_retries expects 3 transcribe calls.

## F05 — Glossary/name recognition
- Status: Fixed (commits bc6eaff, 06dea3b)
- Change: MAX_TERMS increased 12→16. _terms_for prioritizes short nickname exact matches under term cap. name_terms now includes nicknames (term in self.nicknames) so short names like ラプ→Lap pass the canonical_names filter. Glossary has context-aware disambiguation for ambiguous terms (ぼたん/みこ/まつり).
- Tests: short nicknames (ラプ→Lap, みこ→Miko, みこち→Mikochi, まつり→Matsuri, ぼたん→Botan, わため→Watame) are preserved, not expanded to literal words. Prompts include short nicknames.

## F06 — Fragment joining
- Status: Investigated; no code change needed
- _join_fragments has gap/span/generation/sentence-end boundary checks. Tests pass.

## F07 — Subtitle parser/truncation
- Status: Investigated; no code change needed
- split_cue wraps/splits rather than truncates. No silent truncation found.

## F08 — Live sync
- Status: Fixed (commit 85b7200)
- Change: Added subtitle_display_lag event recording drift between speech time and display position for late live subtitles. _render uses speech_start_s as show_from, not padded PCM start. Tests added for on-time/late subtitle timing and lag recording.

## F09 — Deduplication
- Status: Fixed (commit 73508b8)
- Change: Dedup now uses source-span identity (start_s, end_s, generation) instead of text-only matching with 10s time window. Different genuine intervals with same translated text are no longer suppressed. Only exact same-interval re-delivery is rate-limited.

## F10 — Benchmark
- Status: Fixed (commit cb46556 + benchmark latency metrics)
- Change: Benchmark harness exists at scripts/benchmark.py (CER, chrF, name recall, RTF metrics). Added --validate-only mode. Added subtitle_ready_s and speech_to_subtitle_s end-to-end latency metrics to rows and summary (subtitle_ready_p95_s, speech_to_subtitle_p95_s). Added regression tests for validate-only path.
- Blocker: Real-audio evaluation requires annotated WAV clips; none available in repo. Harness is ready.

## Additional fixes (beyond original F01-F10)
- **subtitle_clean.py**: has_generation_loop optimized from O(n^3) list comparison to O(n^2) tuple n-gram comparison with early exit for n<18.
- **config.toml**: safe_playback.enabled changed false→true (RTX 3070 8 GB VRAM bounding for concurrent Qwen3-ASR + Hy-MT2).
- **VideoTranslate.spec**: Added PySide6 plugins (platforms, styles, imageformats) to datas for reliable GUI launch on clean machines.

## Build
- EXE: C:\Games\video-translate-qwen-hymt\dist\VideoTranslate\VideoTranslate.exe (2,393,097 bytes, valid PE, launches successfully)
- PySide6 plugins included: qwindows.dll, qmodernwindowsstyle.dll, etc.

## Test Suite
- 419 tests pass, 1 skipped, 0 failures.
