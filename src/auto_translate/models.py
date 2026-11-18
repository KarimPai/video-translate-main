import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


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
    from .config import load_config

    cfg = load_config(Path("config.toml"))
    asr = Path(cfg.asr.model_path)
    required = ("added_tokens.json", "config.json", "generation_config.json", "merges.txt",
                "model.safetensors", "normalizer.json", "preprocessor_config.json",
                "special_tokens_map.json", "tokenizer_config.json", "vocab.json")
    missing = [name for name in required if not (asr / name).is_file()]
    if missing:
        snapshot_download(repo_id="BuzzASR/japanese",
                          revision="a64b1cf652ba0eb73a9671b0462d142b2c9b17b3",
                          local_dir=asr, allow_patterns=missing)
    gguf = Path(cfg.translate.model_path)
    if not verify_file(gguf, MODELS[0]):
        raise ValueError(f"Configure translate.model_path with the verified HY-MT2 GGUF: {gguf}")


if __name__ == "__main__":
    main()
