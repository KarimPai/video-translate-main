"""Build identity for the source GUI and self-contained packaged frontend."""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


def build_identity(root: Path, *, frozen=None, bundle_dir=None) -> dict:
    frozen = getattr(sys, "frozen", False) if frozen is None else frozen
    if frozen:
        directory = Path(bundle_dir or getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        manifest = directory / "gui-build.json"
        if not manifest.is_file():
            return {"commit": "unversioned", "gui_sha256": None, "protocol": 1, "packaged": True}
        data = json.loads(manifest.read_text(encoding="utf-8-sig"))
        if (not isinstance(data, dict) or data.get("schema_version") != 1 or data.get("protocol") != 1
                or not re.fullmatch(r"[0-9a-f]{40}", str(data.get("commit", "")))
                or not re.fullmatch(r"[0-9a-f]{64}", str(data.get("gui_sha256", "")))):
            raise ValueError("Invalid GUI build manifest; rebuild using scripts/Update-GUI.ps1")
        return {**data, "packaged": True}
    root = Path(root)
    commit = "source"
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=2,
            stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).strip()
    except (OSError, subprocess.SubprocessError):
        pass
    gui = root / "src" / "auto_translate" / "gui.py"
    digest = hashlib.sha256(gui.read_bytes()).hexdigest() if gui.is_file() else None
    return {"commit": commit, "gui_sha256": digest, "protocol": 1, "packaged": False}


def verify_build(identity: dict, commit: str, gui_sha256: str) -> None:
    if identity.get("commit") != commit or identity.get("gui_sha256") != gui_sha256.lower():
        raise ValueError("Staged EXE build identity does not match the requested source; installation stopped")
