# -*- mode: python ; coding: utf-8 -*-
# GUI-only PyInstaller build for video-translate-universal.
#
# The GUI uses QProcess to start the real translation backend with the
# checkout's .venv/Scripts/pythonw.exe. Never freeze the ASR, VAD, or
# translation engines into the GUI: those models and libraries already
# belong to the separate backend environment.
#
# Default to fast incremental rebuilds without UPX. For a release-size build:
#   $env:VIDEO_TRANSLATE_UPX='1'; python -m PyInstaller --noconfirm VideoTranslate.spec
# Avoid --clean for ordinary incremental rebuilds; use it after major
# dependency or spec changes if stale analysis is suspected.

import os

USE_UPX = os.environ.get('VIDEO_TRANSLATE_UPX') == '1'

a = Analysis(
    ['src/auto_translate/gui_bootstrap.py'],
    pathex=['src'],
    binaries=[],
    datas=[],
    hiddenimports=[
        # GUI and lightweight registry metadata only.
        'auto_translate.gui',
        'auto_translate.gui_runner',
        'auto_translate.registry',
        'auto_translate.runtime_diagnostics',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Backends execute in the separate .venv Python process. These are not
        # required for populating the GUI's registry/model dropdowns.
        'auto_translate.asr',
        'auto_translate.translate',
        'auto_translate.vad',
        'auto_translate.ingest',
        'auto_translate.output',
        'auto_translate.pipeline',
        'auto_translate.llama_server',

        # Avoid pulling heavy inference frameworks into the PySide6-only GUI.
        # These remain installed in the project's .venv for the backend.
        'torch', 'torchaudio', 'torchvision',
        'ctranslate2', 'faster_whisper', 'av',
        'onnxruntime', 'silero_vad_lite',
        'transformers', 'tokenizers', 'safetensors',
        'numpy', 'scipy', 'soundfile', 'huggingface_hub',

        # GUI uses QtCore, QtGui and QtWidgets; no embedded web/QML player.
        'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets',
        'PySide6.QtWebEngineQuick', 'PySide6.QtQml',
        'PySide6.QtQuick', 'PySide6.QtQuickWidgets',
        'PySide6.Qt3DCore', 'PySide6.Qt3DRender',
        'PySide6.Qt3DInput', 'PySide6.Qt3DExtras',
        'PySide6.QtPdf', 'PySide6.QtPdfWidgets',

        # Build tools and unused transitive dependencies.
        'setuptools', 'pip', 'wheel', 'cython',
        'altgraph', 'pefile', 'pywin32-ctypes',
        'mpmath', 'sympy', 'networkx',
        'joblib', 'threadpoolctl', 'lazy_loader',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VideoTranslate',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=USE_UPX,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=USE_UPX,
    upx_exclude=[],
    name='VideoTranslate',
)
