# Parakeet JA secondary ASR (Windows RTX 3070)

This experimental branch replaces Nemotron 3.5 with NVIDIA's Japanese-only
`nvidia/parakeet-tdt_ctc-0.6b-ja`. Haas (Vulkan), LFM2.5-JP selector,
PLaMo translation, FireRedVAD and MPV remain unchanged.

## Install

1. Download the **Windows CUDA** release of CrispASR from
   https://github.com/CrispStrobe/CrispASR/releases
   and extract `crispasr.exe` and all bundled runtime DLLs together to
   `tools\crispasr\` (or set `CRISPASR_EXE` to its full executable path).
2. Download `parakeet-tdt-0.6b-ja-q8_0.gguf` from
   https://huggingface.co/cstr/parakeet-tdt-0.6b-ja-GGUF
   to `models\parakeet-tdt-0.6b-ja-q8_0.gguf`.
   Alternatively set `PARAKEET_JA_MODEL` to its full file path.
   Q8_0 is intentional: the publisher reports TDT repetition problems at Q4_K.
3. Keep `dual_asr.language_routing = false`. Parakeet JA cannot classify
   English or Indonesian. Japanese forced decoding can produce nonsense on
   other languages; do not interpret that as valid translation.
4. Run `uv sync`, then `Start translate.bat`.
5. Confirm `logs\parakeet.log` and `logs\reconciler.log` show healthy
   startup. If CrispASR or the model is missing, secondary ASR is disabled
   and the existing Haas path continues.

The secondary uses CrispASR's persistent HTTP server on a loopback ephemeral
port. It sends each 16 kHz mono utterance as WAV and reads a JSON transcript.
The server stays resident until shutdown.

## Limitations and validation

- This branch has not yet been tested on the user's Windows RTX 3070.
- English passthrough from Nemotron is removed, not replaced.
- A 2-second live secondary timeout may be too aggressive on first GPU use;
  tune `dual_asr.live_timeout_s` only after measuring.
- Existing comparison JSONL fields retain some historical `nemotron` names
  in downstream tools; review analytics before comparing old and new runs.
- Run `uv run --no-sync python -m unittest discover -s tests -v` before
  using for long streams. Check actual GPU memory, first inference latency,
  transcript quality and process cleanup.
