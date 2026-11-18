# Audit Implementation Progress

## F01 — Qwen ASR prompt echo
- Status: Fixed (commit a967a29)
- Change: context_echo guard now checks full system prompt, not only cfg.context.
- Tests: tests/test_qwen3_asr.py 20/20 pass.

## F04 — Retry/missing intervals
- Status: Fixed (commit 0da4bc8). Bounded retries (max 2) for ASR transport failures (last_rejection == "error") in run_asr. Quality rejections not retried. Each attempt recorded as event with retries count. Test updated: test_vod_transport_failure_bounded_retries expects 3 transcribe calls.

## F05 — Glossary/name recognition
- Status: Partial; MAX_TERMS increased to 16 (commit bc6eaff). Glossary has context-aware disambiguation. Further work needed on nickname canonicalization.

## F06 — Fragment joining
- Status: Not yet modified; _join_fragments uses time gap/span and Japanese sentence end detection.

## F02 — Live metadata
- Status: Fixed (commit 9df82bf). resolve_source now passes timeline to Source for live mode.

## F01 — Qwen ASR prompt echo
- Status: Fixed (commit a967a29).

## F03 — Short speech suppression
- Status: Investigated; VAD short_reply logic and ASR quality guards are conservative. Tests pass.

## F07-F10
- Status: Pending.

