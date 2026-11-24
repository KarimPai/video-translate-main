"""Offline discovery of relocated model files beneath the project models/ folder.

The registry advertises backend implementations, not installed models. This
module resolves only files whose on-disk format matches the selected backend.
No downloads, model switching, or changes to saved configuration occur.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)
_MIN_CT2_BYTES = 1024 * 1024


def models_root() -> Path:
    # src/auto_translate/model_discovery.py -> repository/models
    return Path(__file__).resolve().parents[2] / "models"


def _relative_model_path(value: str) -> Path | None:
    """Recover the suffix after 'models' from current or old Windows paths."""
    parts = [part for part in re.split(r"[\\/]+", value) if part and part != "."]
    positions = [i for i, part in enumerate(parts) if part.casefold() == "models"]
    if positions and positions[-1] < len(parts) - 1:
        return Path(*parts[positions[-1] + 1:])
    if len(parts) == 1 and ":" not in parts[0]:
        return Path(parts[0])
    return None


def _inside(candidate: Path, root: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _gguf_ok(path: Path) -> bool:
    if not path.is_file() or path.suffix.casefold() != ".gguf":
        return False
    try:
        with path.open("rb") as source:
            return source.read(4) == b"GGUF"
    except OSError:
        return False


def _whisper_ok(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    try:
        names = {child.name.casefold() for child in directory.iterdir() if child.is_file()}
        return (
            {"model.bin", "config.json"}.issubset(names)
            and bool({"tokenizer.json", "vocabulary.json"} & names)
            and (directory / "model.bin").stat().st_size >= _MIN_CT2_BYTES
        )
    except OSError:
        return False


def inventory(root: Path | None = None) -> tuple[list[Path], list[Path]]:
    """Inventory only structurally valid GGUF files and CT2 Whisper folders."""
    directory = root if root is not None else models_root()
    if not directory.is_dir():
        return [], []
    ggufs = sorted(
        (path for path in directory.rglob("*") if _gguf_ok(path)),
        key=lambda path: str(path).casefold(),
    )
    whisper = sorted(
        {path.parent for path in directory.rglob("model.bin") if _whisper_ok(path.parent)},
        key=lambda path: str(path).casefold(),
    )
    return ggufs, whisper


def log_inventory(root: Path | None = None) -> None:
    directory = root if root is not None else models_root()
    ggufs, whisper = inventory(directory)
    log.info("Local model inventory: root=%s GGUF=%d Whisper CT2=%d",
             directory, len(ggufs), len(whisper))
    for path in ggufs:
        log.info("Local GGUF model: %s", path)
    for path in whisper:
        log.info("Local Whisper CT2 model: %s", path)


def _resolve_gguf(configured: str, root: Path) -> str:
    if not configured:
        return configured
    explicit = Path(configured).expanduser()
    if explicit.is_file() and (explicit.is_absolute() or _inside(explicit, root)):
        return configured
    if not configured.casefold().endswith(".gguf"):
        return configured
    suffix = _relative_model_path(configured)
    if suffix is not None:
        candidate = root / suffix
        if _inside(candidate, root) and _gguf_ok(candidate):
            return str(candidate.resolve())
    ggufs, _ = inventory(root)
    leaf = re.split(r"[\\/]+", configured)[-1].casefold()
    exact = [path for path in ggufs if path.name.casefold() == leaf]
    if len(exact) == 1:
        return str(exact[0].resolve())
    if len(exact) > 1:
        log.error("Ambiguous local GGUF '%s'; set an exact path: %s",
                  leaf, ", ".join(str(path) for path in exact))
    else:
        log.warning("Configured GGUF not found under %s: %s", root, configured)
    return configured  # The selected backend retains its normal failure behavior.


def _relocate_qwen_projector(configured: str, model_path: str, root: Path) -> str:
    """Relocate a missing projector only when its configured suffix finds a sibling."""
    if not configured or not model_path or Path(configured).expanduser().is_file():
        return configured
    suffix = _relative_model_path(configured)
    if suffix is not None:
        candidate = root / suffix
        model = Path(model_path).expanduser()
        if (_inside(candidate, root) and _gguf_ok(candidate)
                and candidate.resolve().parent == model.resolve().parent):
            return str(candidate.resolve())
    return configured


def _resolve_whisper(configured: str, root: Path) -> str:
    if configured:
        original = Path(configured).expanduser()
        if original.is_dir() and (original.is_absolute() or _inside(original, root)):
            return configured  # Preserve strict validation of an explicit folder.
    suffix = _relative_model_path(configured) if configured else None
    if suffix is not None:
        candidate = root / suffix
        if _inside(candidate, root) and _whisper_ok(candidate):
            return str(candidate.resolve())
    _, folders = inventory(root)
    if not folders:
        return configured
    if len(folders) == 1:
        # Do not silently use regular Whisper when Turbo was requested.
        requested = configured.casefold()
        candidate_name = str(folders[0].relative_to(root)).casefold()
        if "turbo" in requested and "turbo" not in candidate_name:
            log.warning("Whisper Turbo requested, but only non-Turbo local model found: %s", folders[0])
            return configured
        return str(folders[0].resolve())
    # Never silently choose Whisper variants when more than one is installed.
    leaf = re.split(r"[\\/]+", configured.rstrip("\\/"))[-1].casefold() if configured else ""
    exact = [folder for folder in folders if folder.name.casefold() == leaf]
    if len(exact) == 1:
        return str(exact[0].resolve())
    if "turbo" in leaf:
        turbo = [folder for folder in folders if "turbo" in str(folder.relative_to(root)).casefold()]
        if len(turbo) == 1:
            return str(turbo[0].resolve())
    log.error("Multiple local Whisper models found. Configure an explicit directory: %s",
              ", ".join(str(folder) for folder in folders))
    return configured


def resolve_selected_model_paths(
    backend_type: str, backend_name: str, profile: dict,
    root: Path | None = None,
) -> dict:
    """Recover moved model paths for the selected backend, never cross-load.

    Explicit paths still win when present. Missing configured paths can be
    relocated by suffix or exact filename, except Qwen projectors, which
    can only be relocated alongside their checkpoint. Whisper can discover
    its own single valid local CT2 model without an explicit path.
    """
    root = root if root is not None else models_root()
    resolved = dict(profile)
    if not root.is_dir():
        log.warning("Local models directory not found: %s", root)
        return resolved
    keys: tuple[str, ...] = ()
    if backend_type == "asr" and backend_name == "whisper":
        old = str(resolved.get("model_path") or "")
        new = _resolve_whisper(old, root)
        if new != old:
            resolved["model_path"] = new
            log.info("Resolved Whisper local model: %s", new)
        return resolved
    if backend_type == "asr" and backend_name == "qwen3-asr":
        keys = ("model_path", "mmproj_path")
    elif backend_type == "translate" and backend_name in ("gemma4", "plamo", "plamo8b", "hymt"):
        keys = ("model_path",)
    for key in keys:
        old = str(resolved.get(key) or "")
        if key == "mmproj_path":
            new = _relocate_qwen_projector(old, str(resolved.get("model_path") or ""), root)
        else:
            new = _resolve_gguf(old, root)
        if new != old:
            resolved[key] = new
            log.info("Resolved %s %s %s: %s", backend_type, backend_name, key, new)
    return resolved
