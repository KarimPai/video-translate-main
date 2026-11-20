---
sessionId: session-261121-030322-hv2p
---

# Requirements

### Overview & Goals

Fix four interconnected defects in the PLaMo translation pipeline that cause:
1. **Translation latency** — repeated failed generations and truncation retries waste time
2. **Missing English subtitles** — recognized Japanese speech disappears with no subtitle
3. **VOD playback stalls** — translation watermarks never advance past failed translations
4. **Suboptimal prompt construction** — PLaMo prompt format doesn't match the official chat template

Prioritize translation accuracy, completeness, and subtitle reliability over raw speed.

### Scope

**In Scope:**
- Fix PLaMo prompt construction to use official chat template format
- Fix truncation retry logic (identical retry with same token limit)
- Fix pending-speech lifecycle: clear pending speech on suppressed translations
- Fix VOD watermark advancement for failed translations
- Improve stop-token handling and end-of-generation detection
- Add comprehensive regression tests

**Out of Scope:**
- Changing default model selection
- Adding new ASR or translation backends
- Modifying Haas ASR, Qwen3-ASR, Whisper, Gemma4, or HY-MT2 behavior (except model-path isolation)
- Downloading new models
- Changing VAD segmentation parameters without benchmark evidence
- Removing existing working features

### User Stories

1. As a viewer, I want every recognized Japanese utterance to produce either an English subtitle or a clear suppression, so I don't see missing subtitles.
2. As a viewer, I want VOD playback to continue smoothly even when some translations fail, so playback doesn't get stuck.
3. As a user, I want PLaMo to produce complete translations without repeated truncation retries.
4. As a developer, I want regression tests that verify prompt format, retry behavior, and watermark lifecycle.

### Functional Requirements

1. **PLaMo Prompt**: Use the official chat template with `chat_template_kwargs` for source/target language. Keep Japanese speech separate from supporting instructions (glossary, context, style).
2. **Truncation Recovery**: When `finish_reason == "length"`, retry with a genuinely higher token limit (not the same 256). If still truncated, use a simplified prompt before giving up.
3. **Pending Speech Cleanup**: When a translation is suppressed (empty output, prompt echo, untranslated Japanese), clear the pending speech entry so the watermark can advance.
4. **Watermark Advancement**: Distinguish between successful translation coverage and processed-but-suppressed coverage. Never block playback permanently on a failed translation.
5. **Stop Tokens**: Handle PLaMo's `<|plamo:tag|>` and `<|plamo:bos|>` stop tokens correctly. Investigate and address the `special_eos_id` warnings.
6. **Retry Bounds**: Keep retry counts bounded (max 3). Record retry reasons and results. Avoid repeating identical deterministic requests.

### Non-Functional Requirements

- Accuracy over speed
- Preserve proper nouns, nicknames, honorifics
- No prompt leakage into subtitles
- No Japanese fallback in English field
- Tests must mock inference servers (no GPU usage)
- Preserve modular model-switching design

# Technical Design

### Current Implementation

**Key Files:**
- `src/auto_translate/translate/plamo.py` — PLaMo translator adapter (628 lines)
- `src/auto_translate/pipeline.py` — Pipeline: VAD → ASR → translate → output (1289 lines)
- `src/auto_translate/output/mpv.py` — MPV playback with watermark management (1893 lines)
- `src/auto_translate/config.py` — Configuration dataclasses (675 lines)
- `src/auto_translate/subtitle_clean.py` — Output validation (320 lines)

**Root Causes Identified:**

1. **Truncation Retry Bug** (`plamo.py:506`): `retry_tokens = min(self.cfg.max_tokens * 2, 256)` — when `max_tokens` is 256, the retry also uses 256. This is a no-op retry that produces the same truncated output deterministically.

2. **Missing Pending-Speech Cleanup** (`pipeline.py:644`): When `suppressed=True`, the `continue` statement skips the `out_q.put()` but never clears the pending speech that was registered at line 511. The watermark in `mpv.py` remains blocked forever on this speech region.

3. **Prompt Construction** (`plamo.py:208-235`): The current `build_plamo_prompt()` embeds glossary, context, style, and Japanese speech in a single user message. PLaMo 3 Translate's official chat template supports `chat_template_kwargs` with `source_lang` and `target_lang`, but the current implementation only sets `target_lang: "en"` and doesn't use `source_lang: "ja"`. The glossary/context format may not match what the model expects.

4. **Watermark Stalls** (`mpv.py:777-802`): `advance_to_subtitle()` is only called when a subtitle is published (line 1087 in pipeline.py). When a translation is suppressed, no call is made, so the pending speech entry remains and blocks the watermark.

### Key Decisions

1. **Fix retry token limit**: Change `retry_tokens = min(self.cfg.max_tokens * 2, 256)` to `retry_tokens = max(self.cfg.max_tokens * 2, 512)` so the retry actually increases the limit. Add a second retry layer with a simplified prompt (no glossary/context) before giving up.

2. **Clear pending speech on suppression**: In `pipeline.py`, when `suppressed=True`, call a new `complete_speech()` callback (or reuse `advance_to_subtitle` with a flag) to clear the pending speech entry and advance the watermark past the suppressed region.

3. **Add `source_lang` to chat_template_kwargs**: Update `build_plamo_prompt()` to include `"source_lang": "ja"` in the kwargs, matching PLaMo's official template.

4. **Simplified prompt as retry strategy**: Before increasing token limits, try a minimal prompt (Japanese text only, no glossary/context) which is more likely to complete within the token limit.

5. **Watermark advancement for suppressed translations**: Add a `complete_speech(start_s, end_s, generation)` method to the MPV output that clears pending speech and advances the watermark — called even when no subtitle is published.

### Proposed Changes

#### 1. Fix PLaMo Prompt Construction (`plamo.py`)

- Add `"source_lang": "ja"` to `chat_template_kwargs` in `build_plamo_prompt()`
- Restructure the user message to separate Japanese speech from supporting instructions
- Validate the prompt format against the installed chat template

#### 2. Fix Truncation Retry (`plamo.py`)

- Change `retry_tokens = min(self.cfg.max_tokens * 2, 256)` to `retry_tokens = max(self.cfg.max_tokens * 2, 512)`
- Add a simplified-prompt retry before increasing token limits
- Detect repeated deterministic output (same response as first attempt) and skip redundant retries
- Record retry reasons and outcomes in diagnostics

#### 3. Fix Pending-Speech Lifecycle (`pipeline.py` + `mpv.py`)

- Add `complete_speech(start_s, end_s, generation)` to `MpvOutput` — clears pending speech and advances watermark
- In `run_translate()`, call `complete_speech()` when `suppressed=True` (line 644)
- This ensures the watermark advances even when no subtitle is published

#### 4. Improve Stop-Token Handling (`plamo.py`)

- Investigate the `special_eos_id` warnings — they come from llama.cpp when the GGUF metadata defines special tokens not in the `special_eog_ids` array
- Ensure stop tokens are properly passed to llama-server via the `stop` field in the request body
- Add logging for `finish_reason` values to diagnose unexpected behavior

#### 5. VOD Recovery (`mpv.py`)

- Add a timeout for pending speech entries: if a speech region remains pending for more than 30 seconds without a subtitle, clear it and advance the watermark
- This prevents permanent stalls from edge cases where the cleanup callback is missed

### File Structure

**Modified Files:**
- `src/auto_translate/translate/plamo.py` — Prompt, retry, stop-token fixes
- `src/auto_translate/pipeline.py` — Pending speech cleanup on suppression
- `src/auto_translate/output/mpv.py` — `complete_speech()` method, pending speech timeout
- `src/auto_translate/subtitle_clean.py` — Minor: add detection for repeated deterministic output

**New Test File:**
- `tests/test_plamo_fixes.py` — Regression tests for all fixes

### Architecture Diagram

``` mermaid
graph LR
    A[VAD] --> B[ASR]
    B --> C[run_translate]
    C --> D[plamo.translate]
    D --> E{Valid Output?}
    E -->|Yes| F[Subtitle Published]
    E -->|No/Suppressed| G[complete_speech]
    F --> H[advance_to_subtitle]
    G --> H2[Watermark Advances]
    H --> I[MPV Scheduler]
    H2 --> I
    I --> J{Translation Lead?}
    J -->|Enough| K[Play]
    J -->|Low| L[Auto-Pause]
    L --> M{Resume Threshold?}
    M -->|Yes| K
    M -->|No| L
```

### Risks

1. **Prompt format changes may affect translation quality**: The restructured prompt may produce different translations. Mitigation: test with known Japanese samples and compare output.
2. **Higher token limits increase VRAM usage**: Doubling `max_tokens` from 256 to 512 increases context window usage. Mitigation: the simplified prompt retry reduces input tokens, offsetting the increase.
3. **Pending speech timeout may skip legitimate delays**: A 30-second timeout could clear speech that's still being translated. Mitigation: only apply when no inference queues remain and the generation hasn't changed.

# Testing

### Validation Approach

Add comprehensive regression tests in `tests/test_plamo_fixes.py` covering all fixes. Tests mock inference servers to avoid GPU usage.

### Key Scenarios

1. **PLaMo Prompt Format**: Verify `build_plamo_prompt()` produces correct `chat_template_kwargs` with both `source_lang` and `target_lang`.
2. **Glossary Handling**: Verify glossary terms are included in the prompt without leaking into output.
3. **Truncation Retry**: Verify retry uses a higher token limit than the first attempt.
4. **Simplified Prompt Retry**: Verify a minimal prompt is tried before increasing token limits.
5. **Repeated Output Detection**: Verify identical deterministic retries are detected and handled.
6. **Empty Translation**: Verify empty output triggers pending speech cleanup.
7. **Suppressed Translation**: Verify suppressed translations clear pending speech and advance watermark.
8. **Pending Speech Timeout**: Verify pending speech is cleared after timeout.
9. **VOD Auto-Pause Recovery**: Verify playback resumes after translation failure.
10. **Backend Isolation**: Verify model paths don't leak between backends.

### Edge Cases

1. Long Japanese utterances that exceed token limits
2. Rapid seeking during translation
3. Generation changes mid-translation
4. Server crash during retry
5. Multiple consecutive translation failures
6. Glossary terms that match the Japanese text exactly

### Test Changes

- **New**: `tests/test_plamo_fixes.py` — ~50 tests covering all fixes
- **Updated**: `tests/test_integration.py` — Update existing PLaMo tests for new prompt format
- **Run**: Full test suite with `pytest` and `ruff` checks

# Delivery Steps

### ✓ Step 1: Fix PLaMo prompt construction and truncation retry
Update `build_plamo_prompt()` to include `source_lang: "ja"` in chat_template_kwargs and restructure the user message. Fix the truncation retry bug (`retry_tokens = min(self.cfg.max_tokens * 2, 256)`) to use `max(self.cfg.max_tokens * 2, 512)`. Add simplified-prompt retry before increasing token limits. Detect and skip repeated deterministic output. Files: `src/auto_translate/translate/plamo.py`

### ✓ Step 2: Fix pending-speech lifecycle and watermark stalls
Add `complete_speech(start_s, end_s, generation)` method to `MpvOutput` in `mpv.py` to clear pending speech and advance the watermark. In `pipeline.py`, call `complete_speech()` when `suppressed=True` (line 644) so the watermark advances even when no subtitle is published. Add a 30-second timeout for pending speech entries to prevent permanent stalls. Files: `src/auto_translate/output/mpv.py`, `src/auto_translate/pipeline.py`

### ✓ Step 3: Improve stop-token handling and diagnostics
Investigate and address the `special_eos_id` warnings from llama.cpp. Ensure stop tokens are properly passed to llama-server. Add detailed logging for `finish_reason` values. Add retry reason and outcome recording. Files: `src/auto_translate/translate/plamo.py`

### ✓ Step 4: Add comprehensive regression tests
Create `tests/test_plamo_fixes.py` with ~50 tests covering: prompt format, glossary handling, truncation retry, simplified prompt retry, repeated output detection, empty translation, suppressed translation cleanup, pending speech timeout, VOD auto-pause recovery, backend isolation. Mock inference servers. Run full test suite and ruff checks. Files: `tests/test_plamo_fixes.py`, `tests/test_integration.py`