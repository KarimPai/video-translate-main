---
sessionId: session-261121-004641-1bmz
---

# Requirements

### Overview & Goals
Targeted reliability pass fixing four priority issues: universal model resolution, Qwen3-ASR permanent error recovery, Whisper explicit path handling, and pipeline config propagation. Plus language filter simplification.

### Scope

**In Scope:**
- P0: Universal model resolution — backend-aware model_path resolution so backends never inherit another backend's model identity
- P1: Qwen3-ASR permanent error recovery — no restart for permanent HTTP, protocol, or config errors
- P1: Whisper explicit path handling — distinguish explicit user path from inherited Haas default
- P2: Pipeline config propagation tests — test actual production config path, not manual reproduction
- P2: Language filter simplification — remove misleading confidence-state comments

**Out of Scope:**
- Rewriting unrelated modules
- Adding new dependencies
- Rebuilding the EXE
- Audio-language identification improvements

### Key Requirements
1. Never inherit another backend's model_path, model format, or companion files
2. Distinguish explicit user paths from inherited defaults in Whisper
3. Restart Qwen3-ASR only for transient failures, never for permanent errors
4. Validate model compatibility before starting expensive processes
5. No automatic model downloads
6. Preserve Japanese dialogue accuracy
7. Keep language filter disabled by default

# Technical Design

### Current Implementation Issues

**P0 — Model Resolution (config.py `get_backend_config()`, registry.py factories):**
- `get_backend_config()` merges common config with backend profile. When a backend profile is absent (e.g., no `[asr.whisper]` or `[translate.plamo]`), the `model_path` from common config is inherited. This means PLaMo gets Gemma's GGUF path, Qwen3-ASR gets Haas's model path, and Whisper gets Haas's model identifier.
- The registry factories pass the merged config dict to backend constructors, which then try to load the wrong model.
- **Root cause**: `model_path` is a backend-identity field, not a shared runtime setting. It should never be inherited from common config when the backend profile is missing.

**P1 — Qwen3-ASR permanent error recovery (qwen3_asr.py lines 185-227):**
- The restart condition (`text is None and self._restarts < restart_limit`) fires for ANY failure, including permanent HTTP errors (400 Bad Request) and `Qwen3ASRError`. Permanent errors should never trigger a restart — the server cannot recover from a malformed request or invalid config.
- The current code breaks out of the retry loop for `httpx.HTTPError` and `Qwen3ASRError` (lines 185-196), but then the outer `if text is None` block still attempts restart because it only checks whether text is None, not the failure category.

**P1 — Whisper explicit path handling (whisper_turbo.py lines 106-151):**
- `_resolve_model_path()` falls through to HF cache search for ANY non-Whisper path, including an explicitly configured GGUF path. If the user sets `asr.model_path = "C:/models/my-whisper"` and that directory is incomplete, Whisper silently substitutes the default cached model instead of reporting the error.
- The resolver cannot distinguish between an explicit user path (Case B — reject with error) and an inherited Haas default (Case D — ignore and search cache).
- **Root cause**: `cfg.model_path` is always a string with no metadata about whether it was explicitly set or inherited.

**P2 — Pipeline config tests (test_regression.py):**
- `test_backend_profile_propagates_to_pipeline()` manually reproduces the config copying logic from `__main__.py` instead of testing the actual production path. If `__main__.py` changes, the test won't catch it.

**P2 — Language filter (text_filter.py):**
- Comments distinguish hiragana/katakana (confirmed Japanese) from CJK-only (uncertain), but both return `True` (keep). The distinction exists only in documentation, not behavior. This is misleading.

### Proposed Changes

#### P0: Backend-aware model resolution
1. **config.py**: In `get_backend_config()`, exclude `model_path`, `mmproj_path`, and `server_path` from common config inheritance when the backend profile is absent. These are backend-identity fields, not shared settings.
2. **config.py**: Add a `resolve_model_identity()` helper that returns backend-specific model paths only from the backend profile or explicit override — never from common config.
3. **config.example.toml**: Add documented backend profiles for all six models with correct paths.

#### P1: Qwen3-ASR permanent error classification
1. **qwen3_asr.py**: Track a `_terminal_failure` flag. Set it for permanent errors (HTTPError, Qwen3ASRError). Check it in the restart condition — never restart when `_terminal_failure` is set.
2. **qwen3_asr.py**: Distinguish transient server errors (5xx HTTP status) from permanent errors (4xx, protocol errors). Retry 5xx as transient.

#### P1: Whisper explicit vs inherited path
1. **whisper_turbo.py**: Add an `is_explicit` parameter to `_resolve_model_path()`. When `is_explicit=True` and the path is invalid, raise a clear error. When `is_explicit=False` (inherited), fall through to cache search.
2. **config.py**: Pass `is_explicit` based on whether the model_path came from an explicit override, backend profile, or common config.

#### P2: Pipeline config propagation test
1. **test_regression.py**: Replace manual config copying test with a test that invokes the actual `__main__.py` config resolution path.

#### P2: Language filter simplification
1. **text_filter.py**: Remove the misleading confidence-state comments. Both kana and CJK-only text are kept — document this simply as "keep any text containing Japanese characters".

### Architecture Diagram

```
graph LR
    A[GUI gui.py] -->|backend selection| B[get_backend_config]
    B -->|model_path excluded from common| C[resolve_model_identity]
    C -->|backend-specific only| D[__main__.py]
    D -->|asr profile| E[registry.create_asr]
    D -->|mt profile| F[registry.create_translate]
    E --> G[Haas / Qwen3-ASR / Whisper]
    F --> H[Gemma4 / PLaMo / Hy-MT2]
```

### File Structure

**Modified files:**
- `src/auto_translate/config.py` — exclude model identity fields from common inheritance, add resolve_model_identity()
- `src/auto_translate/asr/qwen3_asr.py` — terminal failure flag, no restart for permanent errors
- `src/auto_translate/asr/whisper_turbo.py` — explicit vs inherited path distinction
- `src/auto_translate/text_filter.py` — simplify comments
- `config.example.toml` — add backend profiles for all six models
- `tests/test_regression.py` — backend resolution tests, Qwen3-ASR permanent error tests, Whisper path tests

# Testing

### Validation Approach
Run the full test suite with `python -m pytest tests/ -q` and `ruff check .` after each stage.

### Key Scenarios

**P0 — Model Resolution:**
1. Haas + Gemma — verify correct models resolved, no cross-contamination
2. Haas + PLaMo — verify PLaMo does NOT get Gemma's model path
3. Haas + Hy-MT2 — verify Hy-MT2 does NOT get Gemma's model path
4. Qwen3-ASR + Gemma — verify Qwen3-ASR does NOT get Haas model path
5. Whisper + PLaMo — verify Whisper does NOT get Haas model path
6. Whisper + Hy-MT2 — verify no cross-contamination
7. Missing backend profile — verify model_path is empty, not inherited from common config
8. Invalid explicit path — verify clear error, no silent substitution
9. Missing Qwen3-ASR mmproj — verify clear error

**P1 — Qwen3-ASR permanent error recovery:**
1. HTTP 400 Bad Request — no restart, terminal error
2. Invalid response format — no restart, terminal error
3. Permanent Qwen3ASRError — no restart
4. Transient connection failure — bounded retry
5. Repeated timeout — bounded recovery with restart
6. Restart exceeds deadline — no post-restart inference
7. Cancellation during recovery — no stale result
8. Failed restart — proper cleanup and terminal error

**P1 — Whisper explicit path:**
1. Valid local Whisper model — load it
2. Incomplete local directory — clear error
3. Invalid explicit GGUF path — clear error, no fallback
4. Inherited Haas default — ignore and search cache
5. Valid cached Whisper model — use it
6. Incomplete cached snapshot — reject
7. Explicit path does not silently fall back
8. Missing model produces clear error

### Edge Cases
- Backend profile missing — model_path empty, not inherited
- Model path points to wrong format — reject with clear error
- Qwen3-ASR server returns malformed response — classify as permanent error
- Whisper model directory is incomplete HF cache — reject gracefully
- Config file missing — use defaults
- faster-whisper not installed — skip Whisper tests with appropriate marker

### Test Changes
- Add `test_model_resolution_all_six_backends` — test all 6 backends individually
- Add `test_model_resolution_combinations` — test 6 ASR+MT combinations
- Add `test_qwen3_asr_permanent_error_no_restart` — HTTP 400, invalid response, Qwen3ASRError
- Add `test_whisper_explicit_path_rejected` — explicit invalid path raises error
- Add `test_whisper_inherited_haas_path_ignored` — inherited Haas path falls through to cache
- Add `test_pipeline_config_from_production_path` — test actual __main__.py config resolution
- Simplify language filter comments in text_filter.py

# Delivery Steps

### ✓ Step 1: P0 — Fix backend-aware model resolution in config.py
`model_path`, `mmproj_path`, and `server_path` must never be inherited from common config when a backend profile is absent.

- **config.py**: In `get_backend_config()`, identify model-identity fields (`model_path`, `mmproj_path`, `server_path` for ASR; `model_path`, `server_path` for translate). When the backend profile is absent, do NOT include these fields from common config — set them to empty strings instead.
- **config.py**: Add `resolve_model_identity(cfg, backend_type, backend_name, explicit_overrides)` that returns model_path, mmproj_path, and server_path ONLY from explicit override or backend profile — never from common config.
- **config.example.toml**: Add documented `[asr.qwen3-asr]`, `[asr.whisper]`, `[translate.plamo]`, and `[translate.hymt]` sections with correct paths.
- **tests/test_regression.py**: Add `test_model_resolution_all_six_backends` — verify each of the 6 backends resolves correct model paths without cross-contamination.
- **tests/test_regression.py**: Add `test_model_resolution_combinations` — test Haas+Gemma, Haas+PLaMo, Haas+HyMT2, Qwen3+Gemma, Whisper+PLaMo, Whisper+HyMT2.

### ✓ Step 2: P1 — Fix Qwen3-ASR permanent error recovery
Permanent errors must never trigger a server restart.

- **qwen3_asr.py**: Add `_terminal_failure` flag. Set it for `httpx.HTTPError` (except 5xx) and `Qwen3ASRError`. Check it in the restart condition — skip restart when `_terminal_failure` is set.
- **qwen3_asr.py**: Distinguish 5xx HTTP errors (transient server error — retry) from 4xx (permanent client error — no retry).
- **tests/test_regression.py**: Add `test_qwen3_asr_http_400_no_restart` — HTTP 400 should not trigger restart.
- **tests/test_regression.py**: Add `test_qwen3_asr_invalid_response_no_restart` — malformed response should not trigger restart.
- **tests/test_regression.py**: Add `test_qwen3_asr_permanent_error_no_restart` — Qwen3ASRError should not trigger restart.

### ✓ Step 3: P1 — Fix Whisper explicit vs inherited path handling
Distinguish explicit user paths from inherited Haas defaults.

- **whisper_turbo.py**: Add `is_explicit` parameter to `_resolve_model_path()`. When `is_explicit=True` and path is invalid, raise `WhisperError` with clear message. When `is_explicit=False` (inherited), fall through to cache search.
- **config.py**: In `resolve_model_identity()`, track whether model_path came from explicit override, backend profile, or common config. Pass this as `is_explicit` to Whisper.
- **tests/test_regression.py**: Add `test_whisper_explicit_invalid_path_raises` — explicit GGUF path raises error, no fallback.
- **tests/test_regression.py**: Add `test_whisper_inherited_haas_path_ignored` — inherited Haas path falls through to cache.
- **tests/test_regression.py**: Add `test_whisper_valid_explicit_path_loaded` — valid explicit Whisper path is used.

### ✓ Step 4: P2 — Simplify language filter and improve pipeline config tests
Remove misleading comments; test actual production config path.

- **text_filter.py**: Simplify docstring and comments — both kana and CJK-only text are kept. Remove "confirmed Japanese" vs "uncertain" distinction since behavior is identical.
- **tests/test_regression.py**: Replace `test_backend_profile_propagates_to_pipeline()` with a test that imports and calls the actual `__main__.py` config resolution code path.
- Run `python -m pytest tests/ -q` and `ruff check .`.
- Commit and push to `video-translate-universal`.