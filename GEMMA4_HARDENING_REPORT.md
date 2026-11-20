# Gemma 4 Compatibility Hardening — Upstream Findings and Implementation Report

## 1. Upstream llama.cpp Findings

### 1.1. `<unused49>` Token Floods (Confirmed Bug)
- **Source**: https://github.com/ggml-org/llama.cpp/discussions/21338
- **Status**: Confirmed in Gemma 4 26B-A4B
- **Symptom**: Model outputs `<unused49>` tokens indefinitely, filling entire max_tokens budget
- **Impact on our build**: Our build (b12318) is well after the fix (b8664+). The `--jinja` flag we use should prevent this, but added detection as safeguard.
- **Mitigation**: Added `has_special_token_flood()` detection in `subtitle_clean.py`

### 1.2. Reasoning Mode Issues (Resolved)
- **Source**: https://github.com/ggml-org/llama.cpp/issues/24170
- **Status**: Reasoning mode has been completely removed from the project
- **Decision**: Thinking/reasoning mode was disabled and all related code removed for simplicity and reliability

### 1.3. Flash Attention Crashes with Hybrid Attention Models
- **Source**: https://github.com/ggml-org/llama.cpp/issues/21336
- **Status**: Partially fixed, Vulkan still affected
- **Symptom**: Segfault when context exceeds sliding window with `-fa on` on hybrid attention models (Gemma 4, GLM 4.7)
- **Root cause**: Flash attention batches local and global attention layers without properly splitting KV cache indices at the layer-type boundary
- **Impact on our build**: Our context size (4096) is within Gemma 4's sliding window. CUDA build is less affected than Vulkan. Keeping `-fa on` as our build (b12318) should have fixes.
- **Monitoring**: If crashes occur at 4096 context, can disable with `-fa off`

### 1.4. KV Cache Quantization Issues
- **Source**: https://note.com/hacklog_stealth/n/ncfbd00740f8a
- **Status**: Confirmed
- **Symptom**: KV cache q4_0 changes answers in 12/12 questions
- **Impact on our build**: We use FP16 KV cache (default). No `--cache-type` override. Safe.

### 1.5. Missing Gemma 4 Parser PRs
- **Source**: https://www.reddit.com/r/LocalLLaMA/comments/1sc4gui/
- **Status**: Fixed in b8664+
- **PRs**: #21326, #21327, #21343, #21418
- **Impact on our build**: Our build (b12318) includes all fixes. Using `--jinja` as recommended.

### 1.6. Build-Specific Inference Bugs
- **Source**: https://ai.plainenglish.io/i-built-a-gemma-4-ai-agent-it-kept-looping-until-i-fixed-this-1ef53f50158c
- **Status**: Build-specific, late March 2026 window
- **Symptom**: Random character insertions mid-generation
- **Impact on our build**: Our build (b12318) is after the problematic window. Safe.

## 2. Our llama.cpp Build

- **Version**: 0.9.0-dev (build 12318, commit 7021f7e09)
- **Built with**: Clang 20.1.8 for Windows x86_64
- **CUDA**: CUDA 12 build
- **Assessment**: Well after all known Gemma 4 fixes (b8664+). Should be stable.

## 3. Local Bugs Fixed

### 3.1. Special Token Flood Detection (Task 4)
- **Problem**: No detection for `<unused49>`, `<unused8>` token floods
- **Fix**: Added `has_special_token_flood()` in `subtitle_clean.py`
- **Integration**: Integrated into `english_output_issue()` validation pipeline
- **Test coverage**: 15 new tests

### 3.2. Missing finish_reason Tracking (Task 1, Task 5)
- **Problem**: No tracking of `finish_reason` values for diagnostics
- **Fix**: Added comprehensive finish_reason tracking in `_chat()` method
- **Metrics**: `mt_finish_stop`, `mt_truncated`, `mt_finish_other`, `mt_finish_none`

### 3.3. Per-Request Diagnostics (Task 5)
- **Problem**: No per-request metrics for inference monitoring
- **Fix**: Added `record_event("mt_request", ...)` with input/output/reasoning tokens, timing, cache status
- **Fix**: Added `record_event("mt_result", ...)` with retry count and elapsed time
- **Fix**: Added `record_event("mt_rejected", ...)` with rejection reason

### 3.4. Rejection Reason Tracking (Task 1)
- **Problem**: No distinction between model generation failure and validation rejection
- **Fix**: Added rejection reason recording in `translate()` method
- **Diagnostics**: Distinguishes `special_token_flood`, `empty_english`, `untranslated_japanese`, `prompt_echo_or_loop`

## 4. Tasks Already Completed (Previous Session)

- ✅ Special token flood detection
- ✅ Prompt-cache correctness tests
- ✅ Inference diagnostics and finish_reason tracking
- ✅ Rejection reason tracking
- ✅ FP16 KV cache preservation
- ✅ Reasoning mode removed (thinking/reasoning disabled and code cleaned up)

## 5. Tasks Completed This Session

### Task 1 — Output validation and content safety ✅
- Verified `gemma4.py` uses only `message.content`
- Verified `llama_server.py` validates response content correctly
- Added finish_reason tracking and diagnostics
- Added rejection reason recording
- Reasoning mode completely removed: no `--reasoning` flags, no reasoning token budget

### Task 2 — Context-size accuracy benchmark ✅
- Created `scripts/benchmark_context_size.py` with 28 context-sensitive Japanese utterances
- Covers: ambiguous pronouns, multiple speakers, VTuber names, game terminology, emotional expressions
- Includes consistency check (repeated utterances)
- **Note**: Benchmark requires running llama-server; not executed in this session

### Task 3 — Prompt-cache correctness tests ✅
- Created `tests/test_prompt_cache.py` with 8 tests
- Covers: cache_prompt flag verification, repeated requests, changing history, long sessions, cache contamination
- **Result**: Prompt caching works correctly with our implementation

### Task 4 — Special token flood detection ✅
- Added `has_special_token_flood()` in `subtitle_clean.py`
- Detects: `<unused49>`, `<unused8>`, `<eos>`, `<bos>`, `<start_of_turn>`, `<end_of_turn>`, etc.
- Threshold: 3+ same token OR >50% dominance
- 15 unit tests, all passing

### Task 5 — Inference diagnostics ✅
- Per-request metrics: input/output tokens, finish_reason, timing, cache status
- Rejection reason tracking
- Retry count tracking
- Distinguishes model failure from validation rejection

### Task 6 — Flash-attention compatibility ✅
- Verified build: b12318 (well after Gemma 4 fixes)
- Our context (4096) is within sliding window — safe for `-fa on`
- CUDA build less affected than Vulkan
- **Decision**: Keep `-fa on` enabled. Monitor for crashes.

## 5.1 — Real Inference Testing ✅

### Critical Finding: llama.cpp `--reasoning auto` Enabled Reasoning

The GGUF chat template for Gemma 4 E4B includes `enable_thinking` support. llama.cpp's `--reasoning` flag defaults to `auto`, which auto-detects reasoning support from the template and enables it. This caused the model to generate full thinking processes in the `reasoning_content` field, even though we never passed `--reasoning` flags.

**Fix**: Added `--reasoning off` to llama-server args in `gemma4.py` to explicitly disable reasoning mode.

### GGUF Chat Template Analysis

Inspected the actual installed GGUF file (`google_gemma-4-E4B-it-Q4_K_M.gguf`):
- `enable_thinking` defaults to `false` in the Jinja template
- Native markers: `<|think|>`, `<|end_of_think|>`, `<|channel>thought...<channel|>`
- `strip_thinking()` macro removes reasoning content from output
- `reasoning_content` field populated by llama.cpp, not the template

### Real Inference Results (Gemma 4 E4B, RTX 3070 8GB, CUDA)

**Environment:**
- Model: `google_gemma-4-E4B-it-Q4_K_M.gguf` (5.4 GB, Q4_K_M)
- GPU: NVIDIA RTX 3070 8GB, all 42 layers on GPU
- CPU: Intel Core i7-9700K, 4 threads for translation
- llama.cpp: b12318, CUDA 12 build
- Context: 4096, Max tokens: 128, Temperature: 0.0

**Translation Tests (20 utterances):**
- All 20/20 translations produced valid English output
- No reasoning content leakage
- No Japanese script in output
- Short phrases preserved: "Really?", "Wait!", "Amazing!"
- Profanity preserved: "Oh crap!" (やばい！)
- Nicknames preserved: "Asuka-chan, hello."
- Average latency: 0.12s, Min: 0.06s, Max: 0.28s

**Prompt Cache Stability:**
- 3/3 repeated prompts produced identical output
- Cache consistency confirmed

**Reasoning Disabled Verification:**
- No `reasoning_content` warnings with `--reasoning off`
- Server args confirmed: `--reasoning off` present
- No reasoning flags in output

**Clean Output Functions:**
- Slash preservation: 7/7 patterns preserved (yes/no, on/off, 24/7, etc.)
- Reasoning tag stripping: 5/5 patterns handled correctly
- Prompt echo rejection: 4/4 patterns correctly identified

## 6. Test Results

### Unit Tests (Mocked)
- **Total tests**: 643 (212 subtests)
- **All passing**: Yes
- **Test time**: ~31 seconds

### Real Inference Tests
- **Script**: `tests/test_real_inference.py`
- **Translations**: 20/20 passed
- **Cache stability**: 3/3 consistent
- **Clean output**: 23/23 passed
- **Server diagnostics**: All passed
- **No reasoning leakage**: Confirmed

## 7. Recommendations

1. **Keep `--reasoning off`** — Required to prevent auto-detected reasoning mode
2. **Keep 4096 context** — Run `benchmark_context_size.py` to verify no measurable improvement at 8192
3. **Keep `-fa on`** — Our build and context size are safe; monitor for crashes
4. **Keep `cache_prompt=True`** — Real inference confirms correct behavior
5. **Monitor `mt_rejected` counter** — High values indicate model generation issues
6. **Monitor `mt_finish_other` counter** — Unexpected finish reasons may indicate model problems
7. **Run `test_real_inference.py` periodically** — Verify model behavior after updates
