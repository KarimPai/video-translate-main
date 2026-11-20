# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for video-translate-universal

a = Analysis(
    ['src/auto_translate/gui_bootstrap.py'],
    pathex=['src'],
    binaries=[],
    datas=[],
    hiddenimports=[
        'auto_translate.runtime_diagnostics',
        'auto_translate.registry',
        'auto_translate.gui',
        'auto_translate.asr.qwen3_asr',
        'auto_translate.asr.whisper_turbo',
        'auto_translate.translate.plamo',
        'auto_translate.translate.hymt',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    upx=True,
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
    upx=True,
    upx_exclude=[],
    name='VideoTranslate',
)
