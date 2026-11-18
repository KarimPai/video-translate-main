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
- Status: Partial (commit bc6eaff)
- Change: MAX_TERMS increased 12→16 in hymt.py to include more short nicknames (Lap, Mikochi, etc.) in translation hints.
- Glossary has context-aware disambiguation for ambiguous terms (ぼたん/みこ/まつり).

## F06 — Fragment joining
- Status: Investigated; no code change needed
- _join_fragments has gap/span/generation/sentence-end boundary checks. Tests pass.

## F07 — Subtitle parser/truncation
- Status: Investigated; no code change needed
- split_cue wraps/splits rather than truncates. No silent truncation found.

## F08 — Live sync
- Status: Pending; requires real-playback verification

## F09 — Deduplication
- Status: Fixed (commit 73508b8)
- Change: Dedup now uses source-span identity (start_s, end_s, generation) instead of text-only matching with 10s time window. Different genuine intervals with same translated text are no longer suppressed. Only exact same-interval re-delivery is rate-limited.

## F10 — Benchmark
- Status: Pending; requires real audio/annotated clips

## Test Suite
- 410 tests pass, 1 skipped, 0 failures (after all fixes)
