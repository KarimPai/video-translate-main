import hashlib
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download, try_to_load_from_cache


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
        repo_id="pfnet/plamo-3-translate-2b-GGUF",
        revision="4f672106b847200bfdc0f6af6c321cd94eba1aed",
        filename="plamo-3-translate-2b-Q4_K_M.gguf",
        directory="models/plamo",
        size=1807203040,
        sha256="1ffd9073afcc70e35ccc942428aa2b73d7aa2b9ea779bf504b775369e0276c13",
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


def cached_model(model: ModelFile) -> Path | None:
    hit = try_to_load_from_cache(model.repo_id, model.filename, revision=model.revision)
    if isinstance(hit, str):
        path = Path(hit)
        if path.is_file() and path.stat().st_size == model.size:
            return path
    return None


def model_at_default_path(path: str) -> ModelFile | None:
    candidate = Path(path).resolve()
    return next((model for model in MODELS
                 if candidate == (Path(model.directory) / model.filename).resolve()), None)


def ensure_model(model: ModelFile, project_dir: Path, configured: str = "") -> Path:
    default = project_dir / model.directory / model.filename
    destination = Path(configured).expanduser().resolve() if configured else default
    print(f"Checking {destination}...", flush=True)
    if verify_file(destination, model):
        return destination
    if destination != default.resolve():
        raise ValueError(f"Configured model failed size/SHA256 verification: {destination}")
    source = cached_model(model)
    if source is None:
        print(f"Downloading {model.filename} at revision {model.revision}...", flush=True)
        source = Path(hf_hub_download(repo_id=model.repo_id, revision=model.revision, filename=model.filename))
    if not verify_file(source, model):
        raise ValueError(f"Cached model failed size/SHA256 verification: {source}")
    return source


def main() -> None:
    from .config import load_config

    cfg = load_config(Path("config.toml"))
    paths = (cfg.asr.model_path, cfg.asr.mmproj_path, cfg.translate.model_path)
    for model, configured in zip(MODELS, paths, strict=True):
        print(f"Verified {model.filename}: {ensure_model(model, Path.cwd(), configured)}", flush=True)


if __name__ == "__main__":
    main()
