"""Resolve the setup-prefetched default GGUF from Hugging Face's shared cache."""
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError

REPO = "pfnet/plamo-3-translate-2b-GGUF"
REVISION = "4f672106b847200bfdc0f6af6c321cd94eba1aed"
FILENAME = "plamo-3-translate-2b-Q4_K_M.gguf"


def download(*, offline=False):
    return hf_hub_download(REPO, FILENAME, revision=REVISION, local_files_only=offline)


def model_arguments(identifier):
    if identifier == REPO + ":Q4_K_M":
        try:
            return ["-m", download(offline=True)]
        except LocalEntryNotFoundError:
            pass  # Existing installations can still use llama.cpp's own cache.
    return ["-hf", identifier]


if __name__ == "__main__":
    print(download())
