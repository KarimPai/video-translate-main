# Windows GUI setup and executable

The GUI starts the Voxtral speech backend and PLaMo translator from this checkout. Use Windows 11 x64 with a current NVIDIA driver; the intended GPU is an RTX 3070 8 GiB. The implementation and automated tests do not substitute for a real audio/playback validation on that machine.

## Install and launch

1. Clone `experiment/voxtral-mini-4b-single-asr` into its own folder, or extract the complete `VideoTranslate-Voxtral-Windows-GUI` workflow artifact.
2. Run `Setup.bat`. It installs dependencies and verifies the pinned audio.cpp CUDA runtime, Voxtral Q4_K checkpoint, and llama.cpp runtime. Successful model downloads are reused.
3. Double-click `Start GUI.vbs`, or `VideoTranslate.exe` when using a packaged artifact. Keep the complete `_internal` folder beside the executable.
4. Enter a YouTube URL and select the video resolution. Start translation. The **Console** tab retains translations and logs when the URL controls hide.

`Start GUI.vbs` uses `pythonw.exe`; application-owned native helpers suppress external console windows. Setup itself has a console so installation errors remain visible. The launcher logs startup failures to `logs/gui-startup.log` and displays a Windows error dialog when necessary.

The full artifact includes `src/`, `scripts/`, `web/`, the glossary, example configuration, launchers, and dependency lockfile. Setup creates `.venv/` and downloads runtimes/models after extraction. Neither a model checkpoint nor a preexisting environment is embedded in `VideoTranslate.exe`. An EXE-only artifact requires the matching complete distribution.

## Controls

The player retains pause/resume, timeline seeking for VOD, fullscreen, and 360p/480p/720p/1080p selection. MPV retains standard hardware decoding and scaling. Qt owns the seekbar while MPV's competing OSC is disabled. Fullscreen retains playback controls, hides them after inactivity, and reveals them on mouse movement.

Translation memory is enabled by default with 16 recent pairs and at most 20. Seeking and switching videos reset translation context. Optional YouTube chat supplies limited candidate-present terminology evidence. Safe Playback bounds inference concurrency and queues. Diagnostics reports available GPU observations, Voxtral/PLaMo timings, playback facts, and owned workers; unsupported metrics display as unavailable.

Voxtral processes each completed CPU FireRedVAD utterance offline using a resident local audio.cpp server. VAD timestamps determine subtitle timing. A seek invalidates pending results without repeatedly reloading weights; stop/timeout closes the owned server. There is no streaming delay control in the utterance-based backend.

## Configuration and local paths

Copy `config.example.toml` to `config.toml` for local overrides. Use the new schema provided on this branch.

| Setting | Default |
| --- | --- |
| `asr.voxtral_server_path` | `.runtime/audio-v0.9.5-cuda12/audiocpp_server.exe` |
| `asr.voxtral_model_path` | `models/voxtral/voxtral-mini-4b-realtime-2602-q4_k.gguf` |
| `asr.cuda_device` | `0` |
| `asr.threads` | `8` |
| `asr.startup_timeout_s` | `180` |
| `asr.response_timeout_s` | `45` |
| `asr.live_response_timeout_s` | `20` |
| `vad.device` | `cpu` |

Setup records discovered MPV and runtime paths in `.cache/`. All required project paths are relative to the independent clone; no paths to another checkout are required. Checkpoint integrity uses the exact pinned Hugging Face revision and SHA256, and only the specified Q4_K Voxtral checkpoint is downloaded by default.

## Build and troubleshooting

The `windows-gui.yml` GitHub Actions workflow runs Windows/Qt regression checks, builds the frontend with Python 3.12 and PyInstaller, launches the packaged EXE in an offscreen Qt smoke test, includes the source backend/setup, and uploads the complete distribution. Pushes on this branch or `workflow_dispatch` can trigger it. Download artifacts from a successful run; a checked-in workflow is not evidence of a successful EXE build.

For a local frontend build after installing PySide6 and PyInstaller:

```cmd
python -m PyInstaller --noconfirm --clean --onedir --windowed --name VideoTranslate --paths src --hidden-import auto_translate.runtime_diagnostics --hidden-import auto_translate.gui src/auto_translate/gui_bootstrap.py
```

Place the complete frontend distribution inside the installed checkout, or include the same source/setup files as the workflow. Launch from the independent project directory.

If audio.cpp cannot start, verify the complete `.runtime/audio-v0.9.5-cuda12/` directory, Microsoft Visual C++ x64 Redistributable, and NVIDIA driver. Setup checks the bundled CUDA runtime DLLs and native executable startup. If Voxtral cannot load or runs out of CUDA memory, inspect `logs/voxtral-server.log` and the GPU dashboard. Keep unrelated GPU workloads modest, preserve the selected checkpoint, and allow PLaMo CPU fallback. Do not infer runtime memory from the model's download size.

Stop and close the GUI during startup and transcription as part of hardware validation. The application should terminate its own session, MPV, audio.cpp, and any owned translation server while preserving unrelated external processes. No hardware accuracy or latency result is claimed by a GUI launch smoke test.
