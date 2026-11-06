# PLaMo 3 Translate 2B test backend

This is an experimental local backend on the `test` branch. It does not use Ollama.
It connects to a local `llama.cpp` server using the OpenAI-compatible chat endpoint.

## 1. Install llama.cpp

On Windows, install the current llama.cpp package:

```cmd
winget install llama.cpp
```

Use a Windows build that includes GPU support (CUDA or Vulkan). The RTX 3070 should use GPU offload; CPU-only inference is not the intended test configuration.

## 2. Start the model server

The translator starts this server automatically when needed and stops only the process it launched when the translator exits. You no longer need a separate CMD window. If a server is already responding at the configured URL, it is reused and left running.\n\nThe server writes its output to `logs\\plamo-server.log`.\n\nThe equivalent manual command is:

```cmd
llama-server -hf pfnet/plamo-3-translate-2b-GGUF:Q4_K_M -c 8192 --port 8000 --alias plamo-3-translate --jinja --no-prefill-assistant -ngl 99
```

The first launch downloads the approximately 1.81 GB Q4_K_M model. Later launches use the local Hugging Face cache and can run offline. The automatic launcher manages the server lifetime.

## 3. Select PLaMo in config.toml

Add these entries under the existing `[translate]` table:

```toml
backend = "plamo"
plamo_base_url = "http://127.0.0.1:8000/v1"
plamo_model = "plamo-3-translate"
```

Leave the existing HY-MT2 `model` and `base_url` values unchanged. They remain available when switching `backend = "hymt2"`.

## 4. Run the translator

In another CMD window, run the normal VOD command:

```cmd
uv run auto-translate "YOUTUBE_VOD_URL" --mode vod --mpv --no-web
```

Startup should report `PLaMo 3 Translate translator ready`. Translation requests use Japanese-to-English mode, the configured glossary, and recent JA/EN history. The backend does not apply HY-MT2's minimum request interval.

## 5. Compare

Use the same VOD and timestamps for both backends. Record:
- `mt` time per line
- latency and text queue growth in the 30-second stats
- actual GPU VRAM use while Haas and VAD are also running
- translation accuracy for names, slang, jokes, and conversational fragments

To return to HY-MT2, change only `backend = "hymt2"`. Stop the llama.cpp server when it is not needed.

The app does not launch or manage `llama-server` yet; it must be running locally before selecting PLaMo.
