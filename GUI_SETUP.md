# Video Translate Windows GUI (first iteration)

A PySide6 frontend for the existing Haas + Whisper Large-v3 Turbo + LFM +
PLaMo backend. No model, playback timing, HLS or translation logic is replaced.

## Run it

1. Keep the project in its existing folder, e.g. `C:\Games\video-translate-dual-asr`.
2. Run `Setup.bat` once if the backend is not installed.
3. Double-click **Start GUI.bat**. It installs **only** PySide6 if missing; it does not resync/reinstall the existing CUDA backend.
4. Paste a direct YouTube watch/live link, choose 360/480/720/1080p, then Start.
5. The Console tab captures real-time JP and EN subtitle lines and backend logs.
6. When **MPV successfully loads the video**, the URL/quality controls disappear.
   Stop or playback failure restores them, with the previous URL retained.

The Player tab hosts the same MPV application via `--wid=<Qt HWND>` on Windows.
MPV's native OSC, translation timing, VOD seeking, stream buffering and playback
hardware decode remain owned by the existing backend. Window resizing uses Qt.

## Build the GUI executable

Double-click **Build GUI EXE.bat** (Windows only).

The resulting launcher is `dist\VideoTranslate\VideoTranslate.exe`.
It is **not** a fully portable/self-contained translation engine: run it from
the original repository folder under `dist`. The existing
`.venv\Scripts\python.exe`, setup-installed mpv/speech.cpp/llama-server,
CUDA libraries, model cache and `config.toml` are still required. This
deliberately prevents duplicating GBs of models or interfering with CUDA
multiprocessing and model-update workflows.

The classic **Start translate.bat** continues to work unchanged.

## Diagnostics and limitations

- Session output is visible in Console and saved in `logs\run.log`. The player
  still writes to `logs\mpv.log` and the ASR comparison log.
- During model initialization, the URL box remains visible (disabled), and the\n  embedded MPV video host remains the default visible tab.
  It hides only after the MPV `file-loaded` signal reaches the GUI.
- If embedded MPV doesn't display video, inspect `logs\mpv.log`. Native
  `--wid` rendering is Windows-specific and needs local testing on the GPU.
- Stop first requests backend termination, then force-kills it after 4 seconds
  if it is stuck; the existing Windows Job Object should terminate descendants.
- This first iteration uses MPV's own on-screen controller instead of
  replicating timeline/seek logic inside Qt.
