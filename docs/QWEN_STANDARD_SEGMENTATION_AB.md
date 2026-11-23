# Standard Qwen3-ASR 1.7B — controlled shorter-segment experiment

**Goal:** determine whether shorter audio requests recover more correct holoGTA
speech than the current longer Qwen3-ASR requests, **without** changing GUI
settings or production VAD prematurely.

This is *ASR-only*; it does **not** compare two actual Silero VAD settings and
cannot establish CER, missing-speech rate, or hallucination rate by itself.

## Setup on Windows

Use the existing 20-minute local file:
\`C:\Games\video-translate-main\hologta-stress-test.flac\`

Your old comparison script may be locally modified/untracked, so inspect
its status before overwriting it.

\`\`\`powershell
cd C:\Games\video-translate-main
git fetch origin main
git status --short -- scripts/compare_haas_qwen_windows.py tests/test_compare_haas_qwen_windows.py
\`\`\`

If **no personal changes** need preserving in those two files, sync only them
(not the whole branch):

\`\`\`powershell
git restore --source origin/main -- scripts/compare_haas_qwen_windows.py tests/test_compare_haas_qwen_windows.py
uv run pytest -q --basetemp ".\.pytest-qwen-short" tests/test_compare_haas_qwen_windows.py tests/test_audit_20261122.py
\`\`\`

If the tests pass, close VideoTranslate.exe and any **orphaned** project
llama-server workers, then run:

\`\`\`powershell
uv run python .\scripts\compare_haas_qwen_windows.py ".\hologta-stress-test.flac" --model qwen3-asr --window HG02 HG06 HG07 HG10 HG11 HG12 HG13 HG21 HG26 HG27 HG28 HG29 --qwen-short-target-s 10
\`\`\`

Output is printed to \`benchmarks\runs\asr-windows-...\`:

- \`haas_qwen_windows.jsonl\`: raw Japanese hypotheses with exact clip-local
  timestamps and the \`full\`, \`short\`, \`split\` variant.
- \`haas_qwen_windows.csv\`: same results, spreadsheet-friendly.
- \`report.md\`: timing and coverage **proxies**, not an accuracy ranking.

Upload the JSONL, CSV and report.

## What each arm does

- \`full\`: exact 3.6–16s reference audio window, standard Qwen3-ASR 1.7B Q8_0,
  and the **current** production conditional split-first recovery threshold (15s).
- \`short\`: **only when the full window exceeds 10s**, split at quieter 80ms
  energy frames around evenly spaced boundaries. Each chunk is at most 10s.
  A 0.24s overlap reduces hard word cutting. Parts are NOT merged or published.
- \`split\`: historical two-halves diagnostic with 0.4s overlap as an additional
  independent comparison. It is **not** the \`short\` production candidate.

Both settings use the same standard model and matching projector explicitly.
Qwen-JA files under \`models\qwen3-asr-ja\` are ignored. The model is loaded
once for the experiment; PLaMo/Gemma is not loaded. Both baseline and short
calls start without conversational history.

The pause detector is an energy heuristic, not speech verification. More
recognized characters may indicate real recovered speech **or** duplication,
gaming SFX and hallucinations. Count overlap words carefully.

## Evaluation rules

Listen to the original scenes and look especially for:

1. HG06/HG11 — whether short segmentation saves real dialogue without
   splitting particles, losing names or doubling repeated syllables.
2. HG07/HG13 — correct spoken nicknames: Rui, Lamy, Matsuri; do not expand
   short nicknames to stage names.
3. HG12 — the *meaning* of the 300,000-yen exchange, including negation.
4. HG26 — very short spoken replies.
5. HG27–HG29 — speech presence is not confirmed; if both models output
   nothing, do NOT declare verified silence.

If shortened decoding consistently improves the same verified excerpts, run a
**separate, matched full-pipeline Qwen+PLaMo 10s versus 14s VAD soft-cap A/B**
to check playback timing, cue boundaries and translation quality.

No \`config.toml\`, real Qwen model or production pipeline file was edited.
