# Video Translate Windows GUI

The PySide6 frontend controls the existing Haas + Whisper Large-v3 Turbo +
LFM + PLaMo backend. Model inference, MPV subtitles and translation timing
are unchanged.

## What to double-click

- **Start GUI.vbs** in the project root is the primary silent launcher.
- **VideoTranslate.exe** in `dist\VideoTranslate\` is an optional packaged
  launcher; only present after downloading the GitHub Actions build artifact.
- **Setup.bat** is for initial runtime installation or repairing dependencies.
- **Start translate.bat** remains an optional command-line fallback for
  troubleshooting and comparisons.

`Start GUI.vbs` automatically installs PySide6 using `uv` on first launch if
needed. It preserves existing Python dependencies and does not redownload ASR
or translation models. Installation may require internet access.

## GUI operation

1. Open `C:\Games\video-translate-dual-asr\Start GUI.vbs`.
2. Paste a YouTube live or VOD URL, select 360p, 480p, 720p (default), or 1080p.
3. Choose **Start Translation**.
4. Once MPV confirms the video is loaded, the URL and quality controls hide.
5. Use the **Player** tab for embedded MPV and the **Console** tab for Japanese
   transcripts, English translations, and backend logs.
6. Choose **Stop Translation** to restore the setup controls; the URL is retained.

The GUI launches `pythonw.exe` and hides consoles from helper processes in GUI
sessions only. Logs are still available under `logs\`, including `run.log`,
`mpv.log`, and `asr-comparison.jsonl`.

## Obtain the compiled EXE

The repository's **Windows GUI frontend build** GitHub Actions workflow builds
and uploads the `VideoTranslate-Windows-GUI` artifact.

1. Open the latest successful Windows GUI frontend build run in GitHub Actions.
2. Download the `VideoTranslate-Windows-GUI` artifact.
3. Extract the artifact into the project as
   `C:\Games\video-translate-dual-asr\dist\VideoTranslate\`.
4. Open `VideoTranslate.exe`.

The EXE is **not a standalone model bundle**. It uses the original project's
`.venv`, installed MPV/speech.cpp/llama.cpp, shared model caches, and
`config.toml`. Do not move only the EXE outside the project.

## Troubleshooting

- If the launcher reports a missing Python environment, run `Setup.bat`.
- If double-clicking `Start GUI.vbs` opens no window, inspect
  `logs\\gui-startup.log`. The launcher now shows errors from
  `auto_translate.gui_bootstrap` in a Windows dialog.
- To diagnose without the silent launcher, open CMD in the project folder and run:
  `set PYTHONPATH=%CD%\\src && .venv\\Scripts\\python.exe -m auto_translate.gui_bootstrap`.
  Any startup traceback also goes to `logs\\gui-startup.log`.
- If Qt installation fails, ensure `uv` is installed and internet is available,
  then open `Start GUI.vbs` again.
- If the player remains blank, inspect `logs\mpv.log` and `logs\run.log`.
- Embedded MPV and RTX 3070 GPU translation still require testing on the
  actual Windows machine. CI verifies GUI startup and EXE packaging, not
  local video rendering.
