# Dual-ASR feature implementation and validation

Target branch: `feature/haas-whisper-turbo-dual-asr-20261114`.
Inspected starting commit: `bcf78dec13d862559f3cf1082e00d8b0c51258a2`.

The existing FireRedVAD → Haas + Whisper Large-v3 Turbo → LFM2.5-JP →
PLaMo 3 Translate 2B pipeline is preserved. No speaker identification,
voiceprints, enrollment, or new diarization models are added.

## Implemented controls

| Feature | Main options | Behavior |
| --- | --- | --- |
| Translation memory | `[translate] extended_memory = true`, `memory_lines = 16`, `context_lines = 3` | Bounded recent Japanese/English history, relevant older Japanese context, and conservative terminology reuse. History length supports 0–20; 10–20 is recommended. Seek generations and new sessions isolate memory. Spoken short nicknames and uncertain subjects stay conservative. |
| Smart ASR correction | `[dual_asr] smart_correction = true`, `correction_vod_retries = 1`, `correction_live_retries = 0`, separate correction budgets of 8.0/2.0 seconds | Distinguishes superficial differences, records available confidence signals and selection reasons, and retries localized weak audio only when evidence warrants it. Retry counts support 0–2. Reuses Whisper/LFM, checks cancellation/deadlines, and retains fallback/abstention. |
| Public YouTube chat context | `[chat] enabled = false`, `window_s = 30.0`, `max_messages = 100`, `poll_interval_s = 5.0`, `live_time_uncertainty_s = 5.0` | Independent asynchronous public-chat polling. Filters spam/instructions and supplies only bounded weak support for terminology already present in competing ASR candidates. Missing chat does not block translation. |
| Safe Playback | `[safe_playback] enabled = false`, `max_inference_concurrency = 1`, `queue_size = 4`, `audio_chunks = 32`, `max_audio_lead_s = 20.0` | Coordinates inference admission and bounds queued work/read-ahead, without changing MPV hardware decoding, requested resolution, or the separate RTX VSR setting. |
| Diagnostics | `[diagnostics] enabled = true`, `sample_interval_s = 2.0`, `max_bytes = 2000000`, `backups = 3` | Structured rotating logs, cached GPU metrics, stage timings, queue/latency/drop observations, playback facts, worker status, and an optional scoped diagnostics ZIP export. |

The existing GUI now includes these settings and a diagnostics tab. Unknown
hardware metrics and unverified VSR activity remain unknown. Install optional
NVML telemetry with `uv sync --locked --extra diagnostics`; Windows Setup uses
that extra. NVML is not another GPU inference model.

## Source changes

| Files | Purpose |
| --- | --- |
| `src/auto_translate/context_memory.py`, `src/auto_translate/translate/plamo.py` | Relevance selection, bounded translation context, conservative recurring terminology, and current-source translation policy. |
| `src/auto_translate/asr/dual.py`, `asr/selection.py`, `asr/whisper_turbo.py`, `asr/reconciler.py` | Disagreement classification, available confidence/spans, limited crop retries, cancellation/fallback, and structured selection evidence. |
| `src/auto_translate/live_chat.py` | Public live/replay chat retrieval, filtering, timing, generation isolation, and weak candidate-present lexical evidence. |
| `src/auto_translate/safe_playback.py`, `src/auto_translate/pipeline.py` | Cooperative inference gate, smaller optional queues, stale/expired response guards, stage/queue diagnostics, and shutdown reporting. Stage processing timings exclude gate-admission waiting. |
| `src/auto_translate/vad.py`, `src/auto_translate/separation.py`, `src/auto_translate/ingest.py` | Share inference admission with optional CUDA VAD and GPU separation, bound separation audio work, and propagate cancellation/generation changes through ingest. |
| `src/auto_translate/runtime_diagnostics.py`, `src/auto_translate/gpu_budget.py` | Optional persistent NVML measurements, graceful unavailable metrics, bounded diagnostics writer, rotation, and privacy-scoped export. |
| `src/auto_translate/__main__.py`, `src/auto_translate/config.py` | Typed configuration and CLI overrides, shared gate/session wiring, chat timing, and diagnostics lifecycle. |
| `src/auto_translate/gui.py`, `src/auto_translate/gui_runner.py` | Existing-window settings, diagnostics rendering, optional export, and backend command arguments. |
| `src/auto_translate/output/mpv.py`, `src/auto_translate/windows_subprocess.py` | Verifiable playback observations, queue/drop counters, recovery/error events, and bounded application-owned process observations. |
| `tests/test_context_memory_extended.py`, `tests/test_smart_asr_correction.py`, `tests/test_live_chat_context.py`, `tests/test_runtime_diagnostics.py`, `tests/test_safe_playback_regressions.py`, `tests/test_gui_diagnostics.py`, `tests/test_optional_gpu_admission.py`, `tests/test_playback_diagnostics.py`, `tests/test_plamo_safe_timeout.py` | Deterministic regressions for the new features and integration edge cases. Existing regressions are retained and updated where defaults intentionally changed. |
| `config.example.toml`, `config.dual-asr.example.toml`, `README.md`, `GUI_SETUP.md`, `FOUR_FEATURES.md` | Supported defaults, limitations, usage, and physical validation checklist. |
| `pyproject.toml`, `uv.lock`, `scripts/Setup-Windows.ps1`, `.github/workflows/windows-gui.yml` | Optional NVML installation, locked dependency resolution, frontend packaging, build triggers, and settings/dashboard smoke coverage. |

## Validation record

| Check | Result |
| --- | --- |
| Existing-suite baseline at the inspected commit | 265 tests run; 11 skipped; 0 failures/errors. Linux, deterministic fixtures/mocks. |
| Final complete suite | `PYTHONPATH=src:work/gui-test-deps QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -v`: 404 tests run, 394 passed, 10 skipped, 0 failures/errors (Linux, Python 3.12.14, real PySide6 6.12 offscreen). |
| Static checks and optional offscreen GUI checks | Passed: CI Ruff rules (`E9,F63,F7,F82`), source/tests/scripts compilation, `uv lock --check`, configuration loading, workflow YAML and embedded Python parsing, and `git diff --check`. Offscreen Qt settings/export/control tests are included in the final suite. |
| Commit SHA and push | The final delivery response records the verified Git commit and same-branch push result; see this report's containing commit in GitHub history. |
| Windows GUI workflow and artifact | The final delivery response records the observed remote workflow result and artifact link. A push triggers the full Windows offline suite with Qt, existing GUI smoke checks, then executable packaging. |
| Physical Windows 11 / RTX 3070 playback | Not performed in the Linux implementation environment. |

## Limits and hardware validation still required

Memory improves consistency conservatively; it cannot guarantee correct ASR,
recover inaudible dialogue, identify speakers, or establish omitted gender.
Crop retries require useful confidence/timestamp evidence and therefore do not
run for every disagreement. Serial inference may increase subtitle latency;
measure the 10–20 second live-delay target on the physical machine.

Public YouTube chat uses a best-effort public protocol that may change. Replay
offsets are used where supplied. Live PCM receipt yields a `receipt_estimate`
anchor rather than a verified broadcast clock. The configured
`live_time_uncertainty_s` (0–120 seconds) is a conservative allowance; it cannot
prove the actual encoder/network delay or establish exact alignment. Chat stays
weak evidence and never supplies unheard words or spoken subtitles by itself.

GPU memory observations describe whole-device dedicated VRAM rather than this
application alone. WDDM/driver support may leave metrics unavailable. Safe
Playback controls application inference requests, not MPV's internal driver work
or all Windows GPU activity, and does not impose an artificial VRAM ceiling.
Optional CUDA VAD and separation use the same admission gate; cancellation
cannot reliably interrupt a native GPU call already in progress. An externally
started PLaMo server remains outside application ownership: a canceled or timed
out HTTP request cannot forcibly stop its inference, and the application does
not terminate that external server. Account for that limitation when comparing
Safe Playback behavior with a pre-existing endpoint.

Diagnostics JSONL and GUI-owned logs rotate during operation. Inherited helper
logs rotate between process starts and can grow during a long session. The ZIP
export contains bounded allowlisted log tails and optional narrowly filtered
Windows event metadata; application logs may contain subtitle/video text and
should be reviewed before sharing. It does not gather unrelated files, cookies,
raw chat identities, dumps, or unfiltered event messages. Whole-OS freezes can
prevent any further application logging. A missing previous shutdown marker is
reported as possible interruption with unknown cause.

The NVIDIA WATCHDOG / `0x141` root cause remains unconfirmed. Unit tests, a
successful executable build, and offscreen Qt checks do not prove it is fixed.
On the physical Windows system:

1. Run long Japanese VOD/live sessions with memory on/off. Check exact spoken
   Lap/Laplus/honorifics, Miko/Mikochi, Watame, Matsuri/festival, and grass/laughter
   meanings; include conflicting recognizers, incomplete speech, and overlap.
2. Seek repeatedly during ASR/translation, switch videos, and stop/restart.
   Confirm stale subtitles, context, and model responses never reach new sessions.
3. Compare Safe Playback on/off at 360p/480p/720p/1080p with VSR independently
   on/off. Verify decoder observations, NVIDIA's VSR activity indicator, AV sync,
   subtitle latency, GPU temperature/power, and steady memory over time.
4. Test public live/replay chat and unavailable/disabled/members-only chat.
   Check timing estimates against the stream and confirm speech remains decisive.
5. Interrupt Whisper, LFM, PLaMo, MPV, and FFmpeg in controlled sessions. Check
   fallback/recovery logs, subtitle queues, and a clean subsequent start.
6. Close/kill the GUI during playback, seek, cancellation, and model cold start.
   Verify application-owned children exit and unrelated processes survive.
7. After any further system freeze, retain settings and last available logs;
   after reboot, optionally export the scoped Windows event metadata. Correlate
   real hardware evidence before attributing the fault to VSR, VRAM, or CPU.
