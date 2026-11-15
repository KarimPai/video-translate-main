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

## RTX Video Super Resolution (VSR)

The **RTX Video Super Resolution** checkbox is available at all four qualities.
It defaults to off and keeps its setting when you change resolution or start another
video in the same window. Enable RTX VSR under **System → Video** in the NVIDIA
app and check its activity indicator during playback.

MPV requests the NVIDIA D3D11 video processor using the actual decoded height.
The filter targets 1440p with a maximum 3× scale. A 360p source scales to 1080p.
Filter acceptance and playback fallback reasons appear in `logs/run.log`.
Only the driver can confirm enhancement.

## Playback controls

The **Player** tab has a complete **VOD timeline** showing the current playback
position and video duration. Click anywhere on the timeline for an instant
jump, drag the slider handle and release to seek, or use **-10s / +10s** to
jump. Keyboard arrow, Page Up/Down, Home and End actions on the focused
slider also seek. The Console tab logs `GUI: seek requested` when a command
is sent, `GUI player command queued` when received, and `GUI seek command`
when MPV starts processing it; the Player status reports the acknowledgement.
Seeking goes to the backend's MPV scheduler, which synchronizes the audio
reader and English subtitles (and reuses RAM-backed translations when possible).

Select **Fullscreen** to expand the embedded MPV player across the screen.
**The playback bar and full-duration VOD seekbar auto-hide after 3 seconds**
without mouse movement, expanding the video. **Move the mouse anywhere over
the embedded video to reveal them again**. The bar stays visible while dragging
the seek slider. The button becomes **Exit Fullscreen**, so you can click it
after moving the mouse; **F11**, **F**, double-clicking the video, or **Esc**
also switch/exit fullscreen. Outside fullscreen, the controls remain visible.
The fullscreen switch belongs to the Qt host, since MPV is embedded through
a native HWND.

For **livestreams**, the seek slider and seek buttons are disabled to avoid
breaking the delayed live subtitle alignment. Pause remains available.

The GUI launches `pythonw.exe` and suppresses external console windows,
including the embedded `mpv.exe`. In GUI mode, MPV uses `--no-terminal`
and the Windows `CREATE_NO_WINDOW` flag. It still renders inside the Qt
player and accepts IPC seeking/subtitle commands. The classic CLI is
unchanged; logs remain in `logs/run.log`, `logs/mpv.log`, and other files.

**No EXE download is needed for this particular console fix.** It changes
the Python MPV backend. Update the existing Git branch, then reopen the
already-installed `dist/VideoTranslate/VideoTranslate.exe`.

## Updating without downloading the entire GUI again

For an existing EXE installation, the **Windows GUI frontend build** GitHub
Actions workflow now uploads two choices:

- **`VideoTranslate-EXE-Only`**: small incremental download containing only
  `VideoTranslate.exe`. Use for ordinary GUI code changes; keep the existing
  `_internal` folder untouched.
- **`VideoTranslate-Windows-GUI`**: complete package containing EXE and Qt
  runtime. Use for first installation, missing/corrupt `_internal`, or when
  the workflow announces an incompatible runtime/dependency update.

Update the existing Git branch first so the Python backend source is current.
Close VideoTranslate, download and extract the EXE-only artifact into
`C:\Games\video-translate-dual-asr\dist\VideoTranslate\`, and overwrite only
`VideoTranslate.exe`. Start the EXE normally. **Do not delete `_internal`.**

The GUI build pins the Python, PySide6 and PyInstaller versions to match the
existing November 14 full-runtime release. EXE-only patches require an
already-working compatible full install, not just the EXE file. If it fails,
download the full artifact instead.

The EXE is **not a standalone model bundle**. It uses the original project's
`.venv`, installed MPV/speech.cpp/llama.cpp, shared model caches, and
`config.toml`.

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
