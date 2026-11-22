"""Shared pytest safety net.

Real ``llama-server`` processes load multi-GB GGUF models into an 8 GB GPU.
Earlier agent sessions started them from routine tests and left orphans behind.
This hook makes that impossible by default: any attempt to spawn a process whose
program name starts with ``llama-server`` raises ``RuntimeError`` *before* the
child is created, unless the run explicitly opts in with
``VIDEO_TRANSLATE_RUN_REAL_MODELS=1`` (the same switch the real-model suites use).

Mocked tests are unaffected: patched ``subprocess.run``/``owned_popen`` never
reach the interpreter's spawn audit events.
"""
from __future__ import annotations

import os
import sys

_OPT_IN = "VIDEO_TRANSLATE_RUN_REAL_MODELS"
_BLOCKED_PREFIX = "llama-server"


def real_models_allowed() -> bool:
    return os.environ.get(_OPT_IN) == "1"


def _program(command) -> str:
    """Program basename of a Popen argv list/str or a CreateProcess command line."""
    if command is None:
        return ""
    if isinstance(command, (list, tuple)):
        first = str(command[0]) if command else ""
    else:
        text = str(command).strip()
        if text.startswith('"'):
            first = text[1:].split('"', 1)[0]
        else:
            first = text.split(" ", 1)[0]
    return os.path.basename(first.replace("\\", "/")).lower()


def _block_real_llama_server(event: str, args: tuple) -> None:
    if event not in ("subprocess.Popen", "_winapi.CreateProcess") or real_models_allowed():
        return
    # Popen: (executable, args, cwd, env). CreateProcess: (application, command_line, ...).
    for candidate in args[:2]:
        if _program(candidate).startswith(_BLOCKED_PREFIX):
            raise RuntimeError(
                "Blocked: tests must not start a real llama-server (GPU/VRAM safety). "
                f"Mock owned_popen/subprocess, or set {_OPT_IN}=1 for an approved "
                "integration run."
            )


sys.addaudithook(_block_real_llama_server)
