# A/B: Current Qwen3-ASR 1.7B versus Neosophie Qwen3-ASR-JA

Compare the two ASR checkpoints on **exactly the same** fixed 16 kHz mono holoGTA
audio windows, on the same llama.cpp CUDA backend and Q8_0 quantization.

Do not replace the user's existing configuration; this experiment is ASR-only.
Output agreement is not a recognition accuracy score until the Japanese audio has
been independently verified. Speech presence in HG27–HG29 is unverified.

## Model source

Japanese fine-tune: https://huggingface.co/neosophie/Qwen3-ASR-1.7B-JA

**Use the two-file llama.cpp conversion**, NOT the incompatible one-file speech.cpp
GGUF:

https://huggingface.co/minimalcorp/Qwen3-ASR-1.7B-JA-GGUF

- Q8 model: Qwen3-ASR-1.7B-JA-Q8_0.gguf
- Q8 projector: mmproj-Qwen3-ASR-1.7B-JA-Q8_0.gguf
- Combined download: 2,190,132,448 bytes (about 2.19 GB).
- Conversion source revision: 987bda160f2dabfa6757550bcff7cdda2ba0648c.

Expected SHA-256, per conversion model card:

- Decoder: 77b760ea80ee34ac3c6746cf6028e3e0fd630b2fae26270bce1feb33b0a06ef5
- Projector: 81356edc3c64538f130d9156843beee7f4f65d7354820f0d5cabbf9db9798f13

## Important: explicit model selection

After adding the JA GGUF, there are two Qwen model families under models/.
The Qwen registry intentionally refuses to auto-select among them. The A/B
script now explicitly pins both model and projector file paths:

- Current: models/qwen3-asr/Qwen3-ASR-1.7B-Q8_0.gguf plus matching mmproj.
- Fine-tune: models/qwen3-asr-ja/Qwen3-ASR-1.7B-JA-Q8_0.gguf plus matching mmproj.

No edits to config.toml or the registry are needed for this A/B. If a custom
model directory is used, pass --standard-model-dir and/or --ja-model-dir.
To choose Qwen for regular playback afterward, configure its model_path and
mmproj_path explicitly in [asr.qwen3-asr]. Leaving them blank causes the
registry to report that multiple versions were found.

## Windows PowerShell

From C:\Games\video-translate-main:

1. Fetch new files without overwriting other local edits:

       git fetch origin main
       git status --short -- scripts/compare_qwen_ja.py tests/test_compare_qwen_ja.py
       git restore --source origin/main -- scripts/compare_qwen_ja.py tests/test_compare_qwen_ja.py

2. Run the CPU-only tests:

       uv run pytest -q --basetemp ".\.pytest-qwen-ja" tests/test_compare_qwen_ja.py tests/test_compare_haas_qwen_windows.py

3. Download just the two Q8 files using the Hugging Face CLI through uvx:

       uvx --from huggingface_hub hf download minimalcorp/Qwen3-ASR-1.7B-JA-GGUF Qwen3-ASR-1.7B-JA-Q8_0.gguf mmproj-Qwen3-ASR-1.7B-JA-Q8_0.gguf --local-dir ".\models\qwen3-asr-ja"

   Existing files under models/qwen3-asr are unchanged.

4. Run a no-GPU dry run:

       uv run python .\scripts\compare_qwen_ja.py ".\hologta-stress-test.flac" --dry-run

5. Run both models sequentially (12 difficult windows):

       uv run python .\scripts\compare_qwen_ja.py ".\hologta-stress-test.flac"

   Optional: --all-windows for all 30 fixed reference windows; or
   --window HG02 HG07 HG11 HG12 HG13 for a shorter targeted test.

Output is printed, normally:

    benchmarks\runs\qwen-ja-YYYYMMDD-HHMMSS\hypotheses.jsonl
    benchmarks\runs\qwen-ja-YYYYMMDD-HHMMSS\review.csv
    benchmarks\runs\qwen-ja-YYYYMMDD-HHMMSS\report.md
    benchmarks\runs\qwen-ja-YYYYMMDD-HHMMSS\manifest.json

Upload all four files for comparison. The model is not used in GUI playback
until you explicitly choose it and integrate that configuration.

## Interpretation

- Qwen JA may have better proper noun recognition, but it could also worsen
  short Japanese reactions, mixed-language speech, repetition, or silence.
- Numeric differences in characters or percentage agreement are not CER.
- Compare the exact audio with both hypotheses; evaluate named people,
  300,000-yen exchanges, crosstalk, and real/phantom speech.
- The selected windows are fixed independently of VAD; the experiment measures
  recognition behavior, **not** optimal VAD thresholds or live segmentation.
- Keep auto language detection for both initial arms, avoiding a confounding
  forced-Japanese difference. Forced-language prompts require a separate A/B.
- The runtime keeps normal Qwen ASR decoder and >=15s recovery settings; both
  candidate models inherit the same behavior.
