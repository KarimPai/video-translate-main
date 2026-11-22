# AUDIT_CONTINUATION.md

Checkpoint for the next session. Branch `main`, nothing committed.

## Session 1 — Claude (code changes only)
Claude modified `llama_server.py`, `tests/test_plamo8b_integration.py`, and added `tests/conftest.py`, `tests/test_gpu_lifecycle.py`, and this file. No tests were executed.

## Session 2 — Junie (validation, 2026-11-23 01:44)

### Reviewed this session
`llama_server.py` (full file), `windows_subprocess.py`, `tests/conftest.py`, `tests/test_gpu_lifecycle.py`, `tests/test_plamo8b_integration.py` (full file), `src/auto_translate/registry.py` (factory functions).

### Confirmed defects (from Session 1)
1. **LlamaServer.start() could leave a loading child running** when the wait loop was interrupted by `CancelledError`, `KeyboardInterrupt` or any non-HTTP error. Cleanup depended on the 0.1 s watcher thread, which may never run if the backend is exiting. FIXED.
2. **No test-wide guard against real `llama-server` spawns** (no `conftest.py`). FIXED with an audit hook.
3. **PLaMo 8B tests read the user's real `config.toml`**, so they fail whenever local profiles are disabled or renamed. FIXED with an autouse fixture that redirects to a temporary config.

### Fixes made (Session 1)
- `src/auto_translate/llama_server.py`: the health-wait loop is wrapped; on any exception `_abort_startup()` stops only the child this call started. The timeout path still calls `close()` as before.
- `tests/conftest.py` (new): `sys.addaudithook` raises `RuntimeError` before any `subprocess.Popen` / `_winapi.CreateProcess` whose program name starts with `llama-server`, unless `VIDEO_TRANSLATE_RUN_REAL_MODELS=1`.
- `tests/test_gpu_lifecycle.py` (new, 7 tests, fully mocked): cancel during startup, health timeout, unexpected error, child exits early, restart never keeps two children, guard blocks spawn, guard honours opt-in.
- `tests/test_plamo8b_integration.py`: `_hermetic_repo_config` autouse fixture. Assertions unchanged. The real `config.toml` is not touched.

### Additional fix (Session 2)
- `tests/test_plamo8b_integration.py::test_plamo8b_factory_raises_without_model`: added `monkeypatch` on `resolve_local_plamo_model_path` to return `""`, preventing auto-discovery from finding a real model on disk and masking the intended failure. Without this, the factory silently succeeds when a PLaMo 8B GGUF is present.

### Test results
| Suite | Result |
|---|---|
| `test_gpu_lifecycle.py` (7 tests) | **7/7 PASSED** |
| `test_plamo8b_integration.py` (27 tests) | **27/27 PASSED** (after fix) |
| Full CPU-safe pytest (588 tests) | **588/588 PASSED** |
| Ruff | **Clean** (6 unused `noqa` directives auto-fixed) |
| `git diff --check` | **Clean** (no whitespace errors) |

### Process check
Two `llama-server` processes found (PIDs 1720, 12012) — both are user-owned GUI translation servers on ports 56420 and 56450, not test orphans. No test-owned child processes remain.

### Code review findings
- **llama_server.py `_abort_startup`**: Correctly guards with `self._lock`, checks `self.process is process` to avoid killing a replacement child, and swallows cleanup exceptions with `log.exception`. The `except BaseException` in `start()` catches `CancelledError`, `KeyboardInterrupt`, and any unexpected error. ✓
- **conftest.py audit hook**: Covers both `subprocess.Popen` and `_winapi.CreateProcess` events. The `_program()` helper correctly parses list/tuple argv, quoted strings, and plain strings. Case-insensitive matching via `.lower()`. Mocked `owned_popen` never reaches the audit event. ✓
- **windows_subprocess.py `owned_popen`**: Calls `subprocess.Popen` directly, triggering the audit hook. All launch paths are covered. ✓
- **test_gpu_lifecycle.py**: All 7 tests use `FakeProc` and monkeypatched `owned_popen`/`_free_port`. No real processes or GPU access. ✓
- **test_plamo8b_integration.py**: Hermetic config fixture prevents real config.toml dependency. GGUF validation is mocked. Model file existence test is properly `skipif`. ✓

## Not executed / unverified
- `detect_device()` runs `llama-server --list-devices` through `subprocess.run`. It is now blocked in tests unless mocked. Tests that call it unmocked will fail. That is intended.

## Observations, no fix made
- `LlamaServer.close()` is permanent. After a health timeout during `Gemma4Translator.reload()` the translator cannot restart (`start()` raises "server is closed"), and Gemma never calls `reset_for_restart()` (only Qwen3-ASR does). It surfaces as `TranslatorUnavailable`. Decide whether Gemma should reset on a timed-out reload.
- `GpuBudget.monitor()` has no outer try/except; an exception from `refresh()` or `update_metrics()` would silently end sampling.
- `_stop_process()` can raise if `kill()` then `wait(3)` times out, leaving the log handle open.
- `_owned` deque (`maxlen=64`) is only used for status reporting.
- `reload()` on repeated failure steps GPU layers down to 0 with no limit on restarts (Gemma). Qwen3-ASR is capped at 2.

## Not started
- Phase 1 remainder: registry/model-switching teardown order, `windows_job.py` nested-child behaviour, GUI close path, `speech_server.py`, Hy-MT/PLaMo `close()`.
- Phase 2 (MPV: `_seek_base()` / `_jump(-self._now)` replay-from-end, `_cached_seek_ready`, VOD late-subtitle drop at `mpv.py:693-718`).
- Phase 3 (VAD silence counter, `prob_smooth_frames`) and Phase 4 modules.

## Exact next step
Run the tests above. Then continue Phase 1 with `registry.py` plus the GUI/backend model switch (confirm the old translator/ASR `close()` completes before the replacement is constructed), then start Phase 2 at `mpv.py` `_seek_base` / `_jump`.
