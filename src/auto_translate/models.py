import hashlib
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


def ensure_model(model: ModelFile, project_dir: Path) -> Path:
    destination = project_dir / model.directory / model.filename
    print(f"Checking {destination}...", flush=True)
    if verify_file(destination, model):
        print(f"Size and SHA256 verified; reusing {model.filename}", flush=True)
        return destination

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
    for model in MODELS:
        ensure_model(model, Path.cwd())


if __name__ == "__main__":
    main()
