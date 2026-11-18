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
- Status: Fixed (commit 06dea3b)
- Change: MAX_TERMS increased 12→16 (commit bc6eaff). _terms_for prioritizes short nickname exact matches under term cap. Glossary has context-aware disambiguation for ambiguous terms (ぼたん/みこ/まつり).
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
- Status: Fixed (commit cb46556)
- Change: Benchmark harness exists at scripts/benchmark.py (CER, chrF, name recall, RTF metrics). Added --validate-only mode to validate manifest/audio without models. Added regression tests for validate-only path.
- Blocker: Real-audio evaluation requires annotated WAV clips; none available in repo. Harness is ready; run `python scripts/benchmark.py --validate-only cases.json` to validate, then `python scripts/benchmark.py cases.json --config config.toml` with annotated audio.

## Test Suite
- 419 tests pass, 1 skipped, 0 failures.
