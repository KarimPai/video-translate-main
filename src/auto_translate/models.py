import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download


@dataclass(frozen=True)
class ModelFile:
    repo_id: str
    revision: str
    filename: str
    directory: str
    size: int
    sha256: str


MODELS = (
    ModelFile(
        repo_id="ggml-org/Qwen3-ASR-1.7B-GGUF",
        revision="36a678687ba7d07a74ca70ccb0e36902e005fb80",
        filename="Qwen3-ASR-1.7B-Q8_0.gguf",
        directory="models/qwen3-asr",
        size=2165034944,
        sha256="58e22d0532d4eacaf034cfac17a6fed159f37c41390c710186783be439d1fc57",
    ),
    ModelFile(
        repo_id="ggml-org/Qwen3-ASR-1.7B-GGUF",
        revision="36a678687ba7d07a74ca70ccb0e36902e005fb80",
        filename="mmproj-Qwen3-ASR-1.7B-Q8_0.gguf",
        directory="models/qwen3-asr",
        size=355709344,
        sha256="46c1d533af3f354ceb37ce855dbceff7da7fa7cf1e6a523df3b13440bd164c0d",
    ),
    ModelFile(
        repo_id="tencent/Hy-MT2-7B-GGUF",
        revision="ab8472660ac61fac25f1af43fac2599d52a8a775",
        filename="Hy-MT2-7B-Q4_K_M.gguf",
        directory="models/hy-mt",
        size=4624648896,
        sha256="9f96256500f3fc1ab4d64336b58f52a949a95ad7516b0c229476eef782f9f77b",
    ),
)


def verify_file(path: Path, model: ModelFile) -> bool:
    if not path.is_file() or path.stat().st_size != model.size:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == model.sha256


def find_reusable_model(model: ModelFile, project_dir: Path) -> Path | None:
    """Find verified model weights without copying or modifying source files.

    Search only known video-translate model directories and HF snapshots;
    avoid recursive scans through installed Python environments.
    """
    candidates: list[Path] = [project_dir / model.directory / model.filename]
    configured_hub = os.environ.get("HF_HUB_CACHE")
    configured_home = os.environ.get("HF_HOME")
    hubs = [Path(configured_hub)] if configured_hub else []
    if configured_home:
        hubs.append(Path(configured_home) / "hub")
    hubs.append(Path.home() / ".cache" / "huggingface" / "hub")
    hf_repo = "models--" + model.repo_id.replace("/", "--")
    for hub in hubs:
        candidates.extend((hub / hf_repo / "snapshots").glob("*/" + model.filename))
    games = Path("C:/Games") if os.name == "nt" else None
    if games and games.is_dir():
        for project in games.glob("video-translate*"):
            for subdir in ("models/hy-mt", "models/qwen3-asr", "models/hymt2", "models"):
                candidates.append(project / subdir / model.filename)
    seen = set()
    for candidate in candidates:
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        if verify_file(candidate, model):
            return candidate.resolve()
    return None


def resolve_cached_gguf(configured_path: str, model: ModelFile) -> str:
    path = Path(configured_path).expanduser()
    if path.is_file():
        return str(path)
    if path.name != model.filename:
        return configured_path  # Never substitute a model of another quantization.
    found = find_reusable_model(model, Path.cwd())
    if found:
        print(f"Reusing existing {model.filename} at {found}", flush=True)
        return str(found)
    return configured_path


def ensure_model(model: ModelFile, project_dir: Path) -> Path:
    destination = project_dir / model.directory / model.filename
    print(f"Checking {destination}...", flush=True)
    if verify_file(destination, model):
        print(f"Size and SHA256 verified; reusing {model.filename}", flush=True)
        return destination

    reusable = find_reusable_model(model, project_dir)
    if reusable and reusable != destination.resolve():
        print(f"Reusing verified {model.filename} without copying: {reusable}", flush=True)
        return reusable

    print(f"Downloading {model.filename} at revision {model.revision}...", flush=True)
    source = Path(hf_hub_download(
        repo_id=model.repo_id,
        revision=model.revision,
        filename=model.filename,
    ))
    if not verify_file(source, model):
        raise ValueError(f"Cached model failed size/SHA256 verification: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Stage beside the destination so interrupted copies cannot replace it.
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".model-copy-") as folder:
        staged = Path(folder) / model.filename
        shutil.copyfile(source, staged)
        staged.replace(destination)
    print(f"Size and SHA256 verified; installed {destination}", flush=True)
    return destination


def main() -> None:
    from .config import load_config

    cfg = load_config(Path("config.toml"))
    selected = (MODELS[-1],) if cfg.asr.backend == "dual" else MODELS
    for model in selected:
        configured = (cfg.translate.model_path if model == MODELS[-1] else
                      cfg.asr.mmproj_path if "mmproj" in model.filename else cfg.asr.model_path)
        path = Path(configured).expanduser()
        if path.is_file():
            if verify_file(path, model):
                print(f"Reusing configured model: {path.resolve()}", flush=True)
                continue
            raise ValueError(f"Configured model failed SHA256 verification: {path}")
        ensure_model(model, Path.cwd())


if __name__ == "__main__":
    main()
