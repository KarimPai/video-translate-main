# Dual-ASR reliability review and verification

Branch `feature/haas-whisper-turbo-dual-asr-20261114` · Baseline `ecacb34` · 2026-11-15

The [original audit](https://github.com/KarimPai/video-translate-main/blob/ecacb34/review-report.md) records the initial review, architecture diagram and optional engineering proposals. All 15 numbered findings have been investigated. Four of them were also listed as quick wins.

## Sonnet fixes verified

| Baseline change | Verification and follow-up |
|---|---|
| MPV Job Object handle guard | The actual guard rejects missing, mock, non-convertible and small handles before creating a job. A regression exercises the mock path. Real handle assignment requires Windows. |
| MPV overlapped write cleanup | A simulated wait timeout followed by a cancellation error still raises the original write timeout. |
| MPV IPC property reader | A missing property name leaves the property cache intact. Added validation for non-object JSON and non-string names. A regression receives a valid property update and command response after malformed events. EOF still reaches the command waiter. |
| Whisper cancellation and stale replies | The existing spawned-process test checks repeated cancellation and worker reuse. The worker now tags per-request errors too. A regression delivers an obsolete error before the next valid transcript. |
| DualASR.close | Delegates to release. Exercised close during selector startup and confirmed that the newly created selector is disposed. Existing lifecycle tests cover owned workers and selectors. |

## Findings and dispositions

| ID | Change or verified behavior |
|---|---|
| I-1 · quick win | Allocate Whisper request IDs under the lifecycle lock. A separate single-flight guard prevents simultaneous reads or writes on one pipe while allowing close and the hard watchdog to interrupt a blocked decode. |
| I-2 | Arm and disarm the live PCM read deadline under the same condition lock used by its watchdog. The blocking read remains outside the lock. Pauses and queue backpressure between reads leave the watchdog disarmed. |
| I-3 | On Windows, use a zero-time wait on a process handle with SYNCHRONIZE access. An exited process is detectable while another handle remains open. An inaccessible or recycled live PID is treated conservatively. Added a Windows-only retained-handle test. |
| I-4 | The diagnostics writer drains earlier batches, consumes a distinct shutdown marker and exits. Explicit flushes remain reusable. Shutdown waits are bounded. Tested ordering, thread termination and queue accounting across a batch boundary. |
| I-5 | MpvOutput supports context-managed cleanup. The CLI registers each successfully constructed player immediately so an exception before the pipeline takes over closes it. Existing close is idempotent. |
| I-6 | PLaMo's model probe handles refused connections, including the Windows string form. Direct or wrapped timeouts retry within the existing startup budget. Unknown transport failures retain their cause. Wrong aliases and unexpected HTTP status codes still fail startup. |
| I-7 | The baseline already binds the viewer to 127.0.0.1. A config regression checks both the default and explicit LAN binding through output.web_host. LAN mode remains unauthenticated. |
| I-8 | Pass the watched FFmpeg process to the watchdog when starting its thread. The generation and shutdown checks still apply before a kill. |
| I-9 · quick win | Removed the duplicate MPV default-binding switch. Embedded and standalone players continue to load their respective key configurations. |
| I-10 | The scheduler calls recovery once on video failure and returns in both recovery outcomes. |
| I-11 | MPV reads its initialized config directly. Tests that construct partial players now supply the normal config where required. |
| I-12 · quick win | Removed the duplicate separation-worker closed-state initialization. |
| I-13 · quick win | DeepFilter close queues one stop marker across repeated or concurrent closes. Each caller waits for worker completion and can observe the processing error. |
| I-14 | Reconciler startup logs progress every ten seconds and observes a cancellation event during health polling. DualASR release signals it. Shutdown racing successful startup disposes the unpublished process and suppresses CPU retry. Initial startup remains synchronous with the existing 300-second cold-start allowance per placement. |
| I-15 | ASR selection and laughter filtering share a precomputed katakana translation table. The original character range, width normalization and punctuation handling are preserved. No end-to-end latency improvement was measured. |

## Test results and historical exit code

Linux, CPython 3.12.14. Each status below is the test process exit status.

| Run | Ran | Passed | Skipped | Exit |
|---|---:|---:|---:|---:|
| Clean archive of ecacb34, full unittest discovery | 230 | 228 | 2 | 0 |
| Updated branch, focused dual-ASR, failure paths, ingest pacing, live subtitles, RTX VSR and GUI controls | 82 | 80 | 2 | 0 |
| Updated branch, full unittest discovery | 241 | 238 | 3 | 0 |

Full-suite command from the repository root on Linux:

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
```

The historical Windows run was reported as exit 1. Its exact command and complete stdout/stderr were unavailable. The clean baseline rerun did not reproduce it. The original statement that all 230 passed cannot establish the historical process status. Windows CI now runs full discovery as a separate step so its output and exit status can be inspected independently of the expected nonzero batch-preflight checks.

The three skips in updated Linux discovery were native Windows Job Object cleanup, retained Windows process handles, and Qt shutdown (PySide6 unavailable). Compilation of src, tests and scripts and the Git whitespace check succeeded. Offline tests cover player control routing, seek generations, pause intent, live clock mapping, subtitle timing and VSR quality selection. Native Windows playback, visible fullscreen transitions, RTX enhancement, model quality on live Japanese audio and real GPU performance were not measured in this run.

## GitHub Actions verification

At implementation commit `416307d`, both Ubuntu and Windows passed the [offline quality workflow](https://github.com/KarimPai/video-translate-main/actions/runs/43707034532), including full unittest discovery on Windows and the native process-handle checks. The [Windows GUI workflow](https://github.com/KarimPai/video-translate-main/actions/runs/43707034308) also passed its offscreen startup, control-state, seekbar, fullscreen, windowless subprocess and launcher checks. These checks do not exercise video decoding or RTX VSR on physical hardware.

The quality and Windows GUI workflows at baseline `ecacb34` both recorded success as well. Neither workflow contains the missing output from the historical local test process.
