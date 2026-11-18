# Audit Implementation Progress

## F01 — Qwen ASR prompt echo
- Status: Fixed (commit a967a29)
- Change: context_echo guard now checks full system prompt, not only cfg.context.
- Tests: tests/test_qwen3_asr.py 20/20 pass.

## F02 — Live metadata
- Status: Investigated; Source dataclass has live, start_s, duration, timeline. Downstream trace required. Not yet modified.

## F03 — Short speech suppression
- Status: Not yet investigated.

## F04 — Retry/missing intervals
- Status: Not yet investigated.

## F05-F10
- Status: Pending.

