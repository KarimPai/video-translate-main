---
sessionId: session-261121-004641-1bmz
---

# Requirements

### Overview & Goals
Fix the remaining runtime bugs identified in the third fix pass: universal model switching, GPU memory management, Qwen3-ASR recovery, Whisper initialization, llama-server watcher lifecycle, live/VOD timeout correctness, and misleading regression tests.

### Scope

**In Scope:**
- P0: Universal model switching — each backend gets independent, valid model config; no cross-backend model contamination
- P1: GPU pressure management — retry relief when cooldown expires and pressure persists
- P1: llama-server restart watcher lifecycle — ensure single watcher per server, clean thread management
- P1: Qwen3-ASR request deadlines and recovery — single monotonic deadline per utterance, genuine context removal
- P1: Whisper model loading — CPU device_index fix, model directory validation
- P1: Live/VOD ASR timeouts — correct timeout derivation from backend profile
- P1: Regression test corrections — fix broken timeout simulation, add missing tests

**Out of Scope:**
- Rewriting unrelated modules
- Adding new dependencies
- Rebuilding the EXE
- Audio-language identification improvements beyond current text filter

### Key Requirements
1. Never send one backend's model path to another backend
2. Distinguish auto-populated defaults from user-entered overrides
3. Only forward explicit model overrides when user actually selected or edited them
4. Validate model compatibility before startup
5. Display missing-model errors clearly in the GUI
6. No automatic model downloads
7. Preserve Japanese dialogue accuracy
8. Keep language filter disabled by default

# Technical Design

### Current Implementation Issues

**P0 — Model Switching (gui.py lines 486-502, gui_runner.py lines 83-130):**
- `_on_asr_backend_changed()` and `_on_mt_backend_changed()` call `get_backend_config()` which merges common config with backend profile. But `start_translation()` passes `asr_model=self.asr_model.text().strip() or None` as an explicit override to `build_backend_command()`. When the backend profile is missing, the common `model_path` (e.g., Haas path) is shown in the GUI text field and sent as an explicit override — contaminating other backends.
- `build_backend_command()` forwards any non-None `asr_model` or `mt_model` as `--asr-model` / `--mt-model` CLI args, which `__main__.py` treats as explicit overrides with highest precedence.

**P1 — GPU Budget (gpu_budget.py lines 162-192):**
- The monitor loop only requests relief when `level != self._last_pressure` (line 180: `escalated = level != self._last_pressure`). During sustained low or critical pressure, `escalated` is False after the first transition, so no further relief requests occur even when cooldown expires.
- Line 191 updates `_last_pressure` unconditionally, meaning sustained pressure at the same level never triggers relief again.

**P1 — llama-server (llama_server.py lines 254-278):**
- `reset_for_restart()` sets `_shutdown` event (line 272), then immediately clears it (line 275). The cancellation watcher thread (`_watch_cancellation`, line 172-176) polls `_shutdown.wait(0.1)` — it may miss the brief signal window.
- `_cancel_thread` is set to None (line 273) without joining the old thread, so the old watcher may still be running when `start()` creates a new one (line 194-197 checks `if self._cancel_thread is None`).

**P1 — Qwen3-ASR (qwen3_asr.py lines 126-224, 294-395):**
- No single monotonic deadline per utterance: retries use individual timeouts but total time can exceed budget.
- `_transcribe_impl(no_context=True)` (line 298) doesn't meaningfully alter the request payload — the `no_context` parameter is accepted but the messages payload (lines 313-335) doesn't include any user-provided context, so `no_context=True` has no effect.
- The assistant prefix `language Japanese<asr_text>` is always sent regardless of `no_context`.

**P1 — Whisper (whisper_turbo.py lines 64-98):**
- Line 67: `device_index = None` for CPU — this may not be accepted by all faster-whisper/CTranslate2 versions.
- Line 157: `_looks_like_whisper_model()` accepts any directory with `whisper` in the name even without valid CTranslate2 files.
- Beam size is hardcoded to 5 (line 33), not configurable.

**P1 — Timeouts (__main__.py lines 278-290):**
- Line 290: `asr_profile.setdefault("response_timeout_s", decode_timeout)` is ineffective because `get_backend_config()` already returns a dict with `response_timeout_s` from common config.
- The `decode_timeout` calculation (line 278-279) uses `cfg.asr.response_timeout_s` before backend profile is resolved.

**P1 — Tests (test_regression.py lines 155-170):**
- Line 162: `if not calls:` is always False because `calls.append(timeout)` just executed, so the list is never empty. The timeout is never raised.

### Proposed Changes

#### P0: Model Switching Fix
1. **gui.py**: Add per-backend model path storage (`_asr_model_overrides: dict[str, str]` and `_mt_model_overrides: dict[str, str]`) to track user-entered paths separately for each backend.
2. **gui.py**: In `_on_asr_backend_changed()` and `_on_mt_backend_changed()`, populate the text field from the backend-specific profile only. Mark whether the value is a default (auto-populated) or user override.
3. **gui.py**: In `start_translation()`, pass `asr_model=None` when the text field contains the backend's default path (not a user override). Pass the actual path only when the user explicitly changed it.
4. **gui_runner.py**: Add a flag to distinguish default vs explicit model paths in `build_backend_command()`.
5. **config.example.toml**: Add backend-specific profiles for all six models with correct default paths.

#### P1: GPU Budget Fix
1. **gpu_budget.py**: Track last relief time separately for low and critical pressure levels.
2. **gpu_budget.py**: Request relief when cooldown expires AND pressure persists, regardless of whether the level changed.
3. **gpu_budget.py**: Add hysteresis — don't oscillate between low and critical by requiring a small gap before re-classification.
4. **gpu_budget.py**: Allow immediate escalation from low to critical.

#### P1: llama-server Fix
1. **llama_server.py**: In `reset_for_restart()`, join the old cancellation watcher thread before setting `_cancel_thread = None`.
2. **llama_server.py**: Use a generation counter instead of a brief event signal to ensure the watcher sees the restart signal.
3. **llama_server.py**: Ensure only one watcher exists — check and join before creating new ones in `start()`.

#### P1: Qwen3-ASR Fix
1. **qwen3_asr.py**: Add a monotonic deadline per utterance in `transcribe()`. Calculate remaining timeout for each retry.
2. **qwen3_asr.py**: Classify errors as transient (timeout, connection) vs permanent (HTTP error, Qwen3ASRError). No retry for permanent errors.
3. **qwen3_asr.py**: In `_transcribe_impl()`, verify that `no_context=True` actually removes context from the payload. Since Qwen3-ASR doesn't receive user context in the current implementation, document this and remove misleading behavior.
4. **qwen3_asr.py**: Add different live/VOD budget policies.

#### P1: Whisper Fix
1. **whisper_turbo.py**: Test `device_index=None` for CPU with the installed faster-whisper version. Use a supported configuration.
2. **whisper_turbo.py**: Strengthen `_looks_like_whisper_model()` to require CTranslate2 files, not just directory name.
3. **whisper_turbo.py**: Make beam size configurable via `AsrConfig`.

#### P1: Timeout Fix
1. **__main__.py**: Resolve backend profile first, then derive timeout from the effective profile.
2. **__main__.py**: Apply live request limit correctly.
3. **__main__.py**: Log the effective timeout actually passed to the backend.

#### P1: Test Fixes
1. **test_regression.py**: Fix the timeout simulation — use a call counter that raises on first call.
2. **test_regression.py**: Add test where every request fails.
3. **test_regression.py**: Add tests for GPU relief during sustained critical pressure.
4. **test_regression.py**: Add tests for llama-server thread lifecycle.
5. **test_regression.py**: Add tests for real config resolution and GUI command generation for all six models.
6. **test_regression.py**: Add Whisper model-directory compatibility tests.

### Architecture Diagram

```
graph LR
    A[GUI gui.py] -->|backend selection| B[get_backend_config]
    A -->|model path| C[build_backend_command]
    B -->|merged profile| D[__main__.py]
    C -->|CLI args| D
    D -->|asr profile| E[registry.create_asr]
    D -->|mt profile| F[registry.create_translate]
    E --> G[Haas / Qwen3-ASR / Whisper]
    F --> H[Gemma4 / PLaMo / Hy-MT2]
    G --> I[pipeline.run]
    H --> I
    I --> J[GPU Budget Monitor]
    J -->|relief request| H
```

### File Structure

**Modified files:**
- `src/auto_translate/gui.py` — model switching logic, per-backend path tracking
- `src/auto_translate/gui_runner.py` — build_backend_command model override handling
- `src/auto_translate/config.py` — no changes needed, profile system is correct
- `src/auto_translate/gpu_budget.py` — sustained pressure relief, hysteresis
- `src/auto_translate/llama_server.py` — watcher lifecycle fix
- `src/auto_translate/asr/qwen3_asr.py` — deadline, recovery, cancellation
- `src/auto_translate/asr/whisper_turbo.py` — CPU device_index, model validation, beam size
- `src/auto_translate/__main__.py` — timeout derivation fix
- `config.example.toml` — add backend profiles for all six models
- `tests/test_regression.py` — fix broken tests, add missing coverage

# Testing

### Validation Approach
Run the full test suite with `python -m pytest tests/ -q` and `ruff check .` after each stage.

### Key Scenarios

**P0 — Model Switching:**
1. Switch ASR backend from Haas to Qwen3-ASR — verify Qwen3-ASR profile path is used, not Haas path
2. Switch ASR backend from Qwen3-ASR to Whisper — verify Whisper profile path is used
3. Switch translation backend from Gemma4 to PLaMo — verify PLaMo profile path is used
4. User edits model path for Haas, switches to Qwen3-ASR — verify Haas override is preserved separately
5. Build backend command with default paths — verify no `--asr-model` or `--mt-model` CLI args when using defaults
6. Build backend command with user override — verify `--asr-model` CLI arg is present

**P1 — GPU Budget:**
1. Sustained critical pressure — verify relief is requested after cooldown expires
2. Sustained low pressure — verify relief is requested after longer cooldown
3. Escalation from low to critical — verify immediate relief request
4. Recovery from critical to normal — verify no relief request
5. Hysteresis — verify no oscillation around threshold boundaries

**P1 — llama-server:**
1. Reset for restart — verify old watcher thread is joined
2. Multiple restarts — verify only one watcher exists
3. Shutdown during restart — verify no revival
4. Startup failure — verify clean state

**P1 — Qwen3-ASR:**
1. Single timeout followed by success — verify retry works
2. All retries fail — verify server restart is attempted
3. Restart also fails — verify terminal error
4. Cancellation during transcription — verify no result returned
5. Context echo recovery with timeout — verify timeout is enforced
6. `no_context=True` — verify it has a meaningful effect or is documented as unsupported

**P1 — Whisper:**
1. CPU selection with `cuda_device=-1` — verify correct device and device_index
2. CUDA selection — verify device_index is forwarded
3. Non-Whisper model directory — verify rejection
4. Missing model — verify no download attempt
5. Missing faster-whisper dependency — verify clear error

**P1 — Timeouts:**
1. Live mode with Haas — verify live timeout is used
2. VOD mode with Qwen3-ASR — verify VOD timeout is used
3. Backend profile with custom timeout — verify profile timeout wins
4. CLI override — verify CLI timeout wins

### Edge Cases
- Backend profile missing — use compatible backend default, not another backend's model
- Model path points to wrong format — reject with clear error
- GPU monitoring unavailable — continue without relief
- llama-server process dies unexpectedly — detect and handle
- Qwen3-ASR server returns malformed response — classify as permanent error
- Whisper model directory is incomplete HF cache — reject gracefully
- Config file missing — use defaults
- faster-whisper not installed — skip Whisper tests with appropriate marker

### Test Changes
- Fix `test_qwen3_asr_timeout_followed_by_successful_retry` — the `if not calls` bug
- Add `test_qwen3_asr_all_failures_no_recovery`
- Add `test_gpu_sustained_critical_pressure_relief`
- Add `test_llama_server_watcher_lifecycle`
- Add `test_gui_model_switching_all_backends`
- Add `test_whisper_model_directory_validation`
- Add `test_timeout_resolution_live_vod`
- Add `test_build_backend_command_no_default_override`
- Ensure optional faster-whisper tests skip when dependency unavailable

# Delivery Steps

### ✓ Step 1: P0: Fix universal model switching in GUI and config
Each backend gets an independent, valid model configuration; no cross-backend model contamination.

- **gui.py**: Add `_asr_model_overrides` and `_mt_model_overrides` dicts to track per-backend user-entered paths. In `_on_asr_backend_changed()` and `_on_mt_backend_changed()`, store the current text field value before switching, then load the new backend's profile path. Track whether each path is a default (auto-populated from profile) or user override.
- **gui.py**: In `start_translation()`, pass `asr_model=None` when the text field shows the backend's default path. Pass the actual path only when the user explicitly changed it from the default.
- **gui_runner.py**: In `build_backend_command()`, only add `--asr-model` / `--mt-model` CLI args when the value is an explicit user override (not None).
- **config.example.toml**: Add backend-specific profiles for all six models with correct default paths (Qwen3-ASR with mmproj, Whisper with empty model_path, PLaMo, Hy-MT2).
- **tests/test_regression.py**: Add tests for GUI command builder with all six backends, testing both default and custom paths.

### ✓ Step 2: P1: Fix GPU pressure management
Retry relief when cooldown expires and pressure persists; use separate low/critical cooldowns with hysteresis.

- **gpu_budget.py**: Replace single `_last_relief_time` with `_last_low_relief_time` and `_last_critical_relief_time`.
- **gpu_budget.py**: Change relief condition from `escalated and not within_cooldown` to `not within_cooldown` — request relief when cooldown expires regardless of level change.
- **gpu_budget.py**: Add hysteresis to pressure classification: require free memory to rise above `low_free_gb + 0.05` before transitioning from low to normal, and above `emergency_free_gb + 0.05` before transitioning from critical to low.
- **gpu_budget.py**: Allow immediate escalation from low to critical (bypass critical cooldown when escalating).
- **tests/test_regression.py**: Add `test_gpu_sustained_critical_pressure_relief` with mocked GPU readings and controllable clock.

### ✓ Step 3: P1: Fix llama-server restart watcher lifecycle
Ensure only one cancellation watcher exists per server; stop and join obsolete watcher threads safely.

- **llama_server.py**: In `reset_for_restart()`, set `_shutdown` event and then join the old `_cancel_thread` with a timeout (e.g., 2 seconds) before setting it to None.
- **llama_server.py**: In `start()`, check if `_cancel_thread` exists and is alive before creating a new one — if so, join it first.
- **llama_server.py**: Add a `_watcher_generation` counter to ensure the watcher checks it hasn't been superseded.
- **tests/test_regression.py**: Add `test_llama_server_watcher_lifecycle` with mocked processes and thread synchronization.

### ✓ Step 4: P1: Fix Qwen3-ASR deadlines, recovery, and cancellation
Single monotonic deadline per utterance; genuine context removal in recovery; bounded retries.

- **qwen3_asr.py**: In `transcribe()`, add a monotonic deadline (`time.monotonic() + budget`). Calculate remaining timeout for each retry from the deadline. Different budgets for live (shorter) vs VOD (longer).
- **qwen3_asr.py**: Classify errors: transient (TimeoutException, ConnectError) allow retry; permanent (HTTPError with status != 5xx, Qwen3ASRError) do not retry.
- **qwen3_asr.py**: In `_transcribe_impl()`, the `no_context` parameter currently has no effect because no user context is sent in the payload. Document this explicitly: Qwen3-ASR recovery via context removal is unsupported in this implementation. Remove the misleading `no_context` parameter or add a clear comment.
- **qwen3_asr.py**: In `recover_context_echo()`, report that recovery is unsupported for Qwen3-ASR (since context isn't sent).
- **tests/test_regression.py**: Fix `test_qwen3_asr_timeout_followed_by_successful_retry` — change `if not calls:` to use a proper counter. Add `test_qwen3_asr_all_failures_no_recovery`. Add deadline enforcement tests.

### ✓ Step 5: P1: Fix Whisper model loading and configuration
Fix CPU device_index, strengthen model directory validation, make beam size configurable.

- **whisper_turbo.py**: For CPU inference (`cuda_device < 0`), test if `device_index=None` works with installed faster-whisper. If not, omit `device_index` entirely or use `{}` (empty dict) as CTranslate2 expects.
- **whisper_turbo.py**: Strengthen `_looks_like_whisper_model()` — require at least one file from `_CT2_MODEL_FILES` (model.bin, config.json, tokenizer.json, vocabulary.json). Remove the `or "whisper" in path.name.lower()` fallback that accepts incompatible directories.
- **whisper_turbo.py**: Add `beam_size` to `AsrConfig` (default 5, validated range 1-20). Use it in `_transcribe_impl()` instead of hardcoded `_DEFAULT_BEAM_SIZE`.
- **whisper_turbo.py**: Handle incomplete HF cache snapshots — check for required files before accepting a cached path.
- **tests/test_regression.py**: Add Whisper model-directory compatibility tests. Ensure optional faster-whisper tests skip when dependency unavailable.

### ✓ Step 6: P1: Fix live/VOD ASR timeouts and regression tests
Correct timeout derivation from backend profile; fix broken regression tests.

- **__main__.py**: Move timeout derivation after backend profile resolution. Compute `decode_timeout` from the merged `asr_profile` (which includes backend-specific values), not from `cfg.asr` (common config).
- **__main__.py**: Remove ineffective `asr_profile.setdefault("response_timeout_s", decode_timeout)` — instead derive the timeout from the already-merged profile.
- **__main__.py**: Log the effective timeout actually passed to the backend.
- **tests/test_regression.py**: Fix `test_qwen3_asr_timeout_followed_by_successful_retry` — the `if not calls:` is always False. Replace with a call counter that raises on first N calls.
- **tests/test_regression.py**: Add `test_qwen3_asr_all_failures_no_recovery` where every request fails.
- **tests/test_regression.py**: Add `test_timeout_resolution_live_vod` for defaults, backend profiles, and CLI overrides.
- **tests/test_regression.py**: Add `test_build_backend_command_no_default_override` to verify no model path is sent as CLI arg when using defaults.
- Run full verification: `python -m pytest tests/ -q` and `ruff check .`

### ✓ Step 7: P0: Fix GUI backend switching currentData() bug
The `currentData()` call in `_on_asr_backend_changed()` and `_on_mt_backend_changed()` returns the NEW selection after the signal fires, not the OLD backend. This means the previous backend's model path is never properly saved.

- **gui.py**: Track the previous backend explicitly using instance variables (`_last_asr_backend`, `_last_mt_backend`) instead of calling `currentData()` after the signal.
- **gui.py**: Save the old backend's path using the tracked previous backend, not `currentData()`.
- **tests/test_regression.py**: Add tests that simulate real Qt backend switching behavior.

### ✓ Step 8: P1: Fix Whisper recover_context_echo timeout enforcement
`recover_context_echo()` accepts a timeout argument but doesn't enforce it.

- **whisper_turbo.py**: Add timeout enforcement to `recover_context_echo()` — check remaining time before starting transcription.
- **tests/test_regression.py**: Add test for Whisper recovery timeout enforcement.

### ✓ Step 9: P1: Improve language filter for mixed Japanese/English text
The Unicode ratio filter incorrectly rejects valid Japanese dialogue containing Latin-script names.

- **text_filter.py**: Make the filter more conservative — keep text that contains ANY Japanese characters, not just a high ratio.
- **tests/test_regression.py**: Add tests for mixed Japanese/English text like "Jason said こんにちは".

### ✓ Step 10: P1: Apply effective backend config consistently in pipeline
Pipeline reads shared config directly instead of using backend-specific profiles.

- **pipeline.py**: Use backend-specific profiles for timeout and merge settings where applicable.
- **tests/test_regression.py**: Add tests for pipeline config resolution.

### ✓ Step 11: Final verification and commit
Run full test suite, ruff check, and commit verified changes.

---

## Fifth Fix Pass — Remaining Issues (Post-eba0ca5)

### ✓ Step 12: P0: Replace fake GUI test with real Qt integration test
The existing `test_gui_backend_switching_preserves_paths()` only validates `get_backend_config()` and `load_config()`. It never creates a Qt GUI instance or exercises dropdown signals.

- **tests/test_regression.py**: Replace with a real Qt offscreen test that creates `MainWindow`, triggers `currentIndexChanged` signals, and verifies model path fields.
- Test scenarios: Haas → Whisper → Haas, Haas → Qwen3-ASR → Whisper → Haas, Gemma → PLaMo → Hy-MT2 → Gemma.
- Verify custom paths are preserved on switch-back, default restoration clears stale overrides.
- Verify `build_backend_command()` receives correct args (no default path as CLI override).

### ✓ Step 13: P1: Fix Qwen3-ASR restart deadline enforcement
After `_restart()`, the remaining deadline is recalculated but `_warmup()` is not counted. If warmup is slow, the post-restart inference may have no budget left.

- **qwen3_asr.py**: In `transcribe()`, recalculate remaining deadline after `_restart()` and `_warmup()` before the post-restart inference attempt.
- Return early if deadline expired after warmup.
- **tests/test_regression.py**: Add test for slow restart consuming entire budget.

### ✓ Step 14: P1: Strengthen Whisper model resolution against Haas contamination
`_resolve_model_path()` checks `cfg.model_path` first. If the shared ASR config has `haas-ja-v1.0` as `model_path`, Whisper will try to load it.

- **whisper_turbo.py**: In `_resolve_model_path()`, if `cfg.model_path` is set but the directory does not contain Whisper CTranslate2 files, reject it and fall through to HF cache search.
- Never silently accept a non-Whisper model path.
- **tests/test_regression.py**: Add test with Haas path passed to Whisper resolver — verify rejection.

### ✓ Step 15: P1: Propagate effective backend config through pipeline
`pipeline.run()` passes `cfg.translate.context_lines`, `cfg.translate.merge_fragments`, etc. directly to `run_translate()`. These are shared config values, not backend-specific.

- **__main__.py**: Pass the merged backend profile values (not shared config) to `pipeline.run()`.
- **pipeline.py**: Accept backend-specific merge and context parameters as arguments.
- **tests/test_regression.py**: Add test verifying different backend profiles produce different runtime behavior.

### ✓ Step 16: P2: Fix language filter CJK overmatching
`_JAPANESE_CHARS` includes the entire CJK ideograph range (0x4E00-0x9FFF), which matches Chinese characters. Chinese text will be incorrectly kept as Japanese.

- **text_filter.py**: Add a heuristic: if text contains CJK characters but NO hiragana or katakana, treat it as uncertain (keep but log a warning) rather than confirmed Japanese.
- Pure hiragana/katakana presence is a strong Japanese signal. CJK-only text is ambiguous.
- **tests/test_regression.py**: Add test for Chinese text without kana — verify it's kept but not classified as confirmed Japanese.

### ✓ Step 17: P1: Fix Whisper hard timeout documentation
Whisper `recover_context_echo()` checks deadline only after inference returns. CTranslate2 inference blocks and cannot be interrupted from Python.

- **whisper_turbo.py**: Document that the timeout is a soft deadline — inference runs to completion, but the result is discarded if the deadline expired.
- Add a comment explaining that hard interruption of CTranslate2 is not supported without process-level termination.
- No code change needed — the current behavior (check after, discard if expired) is correct.

### ✓ Step 18: Final verification, ruff, and commit
- Run `python -m pytest tests/ -q` and `ruff check .`.
- Fix any lint or test failures.
- Commit and push to `video-translate-universal`.