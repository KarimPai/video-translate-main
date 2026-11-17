"""Fetch and verify the pinned Voxtral Q4_K checkpoint, outside Git."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

REPO = "mistral-experimental/AudioCPP-Voxtral-Mini-4B-Realtime-2602-GGUF"
REVISION = "4c3212f91349c32211172ce332f266a6e9501018"
FILENAME = "voxtral-mini-4b-realtime-2602-q4_k.gguf"
SIZE = 3097662432
SHA256 = "8cafef18ea3e4cad81da8ffc4e72b69d2eab2c159c2e68428e2e088accbfc7f8"


def verify(path: Path) -> bool:
    """Check size and the authoritative Hugging Face LFS SHA256."""
    if not path.is_file() or path.stat().st_size != SIZE:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest() == SHA256


def download(destination: str | Path = "models/voxtral") -> Path:
    from huggingface_hub import hf_hub_download

    folder = Path(destination).resolve()
    path = folder / FILENAME
    if verify(path):
        return path
    folder.mkdir(parents=True, exist_ok=True)
    # HF resumes interrupted downloads and reuses local metadata. An existing
    # corrupt final file must be redownloaded, even if its metadata looks valid.
    resolved = Path(hf_hub_download(
        REPO, FILENAME, revision=REVISION, local_dir=folder,
        force_download=path.exists(),
    ))
    if not verify(resolved):
        raise RuntimeError(f"Voxtral checkpoint integrity check failed: {resolved}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("models/voxtral"))
    args = parser.parse_args()
    print(download(args.destination))


if __name__ == "__main__":
    main()
