"""Worker entrypoint. Do not import FireRedVAD, ONNXRuntime-GPU or parent CUDA DLLs.

Run: python -u -m auto_translate.asr.reazon_worker '<asr-config-json>'
All protocol replies must use stdout; diagnostics use stderr.
"""
from __future__ import annotations

import base64
import ctypes
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

_HANDLES = []


def prepare_sherpa_windows_dlls() -> str:
    """Load the sherpa-onnx-core ORT DLL before importing sherpa's native .pyd."""
    if sys.platform != "win32":
        return "non-Windows runtime"
    spec = importlib.util.find_spec("sherpa_onnx")
    if not spec or not spec.submodule_search_locations:
        raise RuntimeError(
            "sherpa-onnx missing. Run 'uv sync' in the project folder."
        )
    pkg_dir = Path(next(iter(spec.submodule_search_locations)))
    lib_dir = pkg_dir / "lib"
    ort_dll = lib_dir / "onnxruntime.dll"
    if not ort_dll.exists():
        raise RuntimeError(
            f"Matching sherpa-onnx-core onnxruntime.dll missing at {ort_dll}. "
            "Run 'uv sync' or 'uv pip install --reinstall sherpa-onnx-core'."
        )
    # Keep the AddDllDirectory cookie and preload handle alive for the
    # whole subprocess lifetime. Never modify global System32 DLLs.
    _HANDLES.append(os.add_dll_directory(str(lib_dir)))
    os.environ["PATH"] = str(lib_dir) + os.pathsep + os.environ.get("PATH", "")
    try:
        _HANDLES.append(ctypes.WinDLL(str(ort_dll)))
    except OSError as exc:
        raise RuntimeError(
            f"Cannot preload compatible ONNX Runtime DLL at {ort_dll}: {exc}"
        ) from exc
    print(f"Reazon worker preloaded ONNX Runtime from {ort_dll}",
          file=sys.stderr, flush=True)
    return str(ort_dll)


def send(message: dict) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    from .reazon import ReazonK2ASR
    from ..config import AsrConfig
    import numpy as np

    cfg = AsrConfig(**json.loads(sys.argv[1]))
    prepare_sherpa_windows_dlls()
    asr = ReazonK2ASR(cfg)
    send({"type":"ready","detail":"ReazonSpeech k2 v2 / sherpa-onnx CPU"})

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            if req.get("type") == "shutdown":
                break
            if req.get("type") != "transcribe":
                raise ValueError(f"Invalid ASR worker request: {req.get('type')}")
            pcm = base64.b64decode(req["pcm"], validate=True)
            if len(pcm) % 2:
                raise ValueError("Invalid 16-bit PCM byte count")
            audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            text = asr.transcribe(audio)
            send({"type":"result","id":req["id"],"text":text})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            send({"type":"error","message":repr(exc)})
            break
    asr.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        send({"type":"error","message":str(exc)})
        sys.exit(1)
