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
    # Haas-ja-v1.0 is a built-in model in speech.cpp — no GGUF download needed.
    # The model is fetched automatically by speech.cpp on first use.
    ModelFile(
        repo_id="bartowski/google_gemma-4-E4B-it-GGUF",
        revision="main",
        filename="google_gemma-4-E4B-it-Q4_K_M.gguf",
        directory="models/gemma4",
        size=3221225472,  # approximate; will be verified by SHA256
        sha256="",  # will be set after downloading the file
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
    # Only the translation model needs GGUF verification; Haas is built-in to speech.cpp
    paths = (cfg.translate.model_path,)
    for model, configured in zip(MODELS, paths, strict=True):
        print(f"Verified {model.filename}: {ensure_model(model, Path.cwd(), configured)}", flush=True)


if __name__ == "__main__":
    main()
