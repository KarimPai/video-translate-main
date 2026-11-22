"""Regression tests for offline model discovery and moved project folders."""
from pathlib import Path

from auto_translate.model_discovery import inventory, resolve_selected_model_paths


def _gguf(path: Path, *, valid: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((b"GGUF" if valid else b"FAIL") + b"\0" * 32)
    return path


def _whisper(path: Path, *, valid: bool = True) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.bin").write_bytes(b"0" * (1024 * 1024 + 1))
    (path / "config.json").write_text("{}", encoding="utf-8")
    if valid:
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    return path


def test_discover_whisper_with_missing_profile_path(tmp_path):
    found = _whisper(tmp_path / "speech" / "faster-whisper-large-v3-turbo")
    result = resolve_selected_model_paths(
        "asr", "whisper", {"model_path": ""}, root=tmp_path
    )
    assert result["model_path"] == str(found.resolve())


def test_discover_whisper_with_outdated_folder_name(tmp_path):
    found = _whisper(tmp_path / "whisper" / "h2oai-large-v3-turbo")
    result = resolve_selected_model_paths(
        "asr", "whisper", {"model_path": "models/whisper-large-v3-turbo"}, root=tmp_path
    )
    assert result["model_path"] == str(found.resolve())


def test_incomplete_whisper_is_not_selected(tmp_path):
    _whisper(tmp_path / "incomplete-whisper", valid=False)
    profile = {"model_path": "models/whisper-large-v3-turbo"}
    assert resolve_selected_model_paths("asr", "whisper", profile, root=tmp_path) == profile


def test_multiple_whisper_variants_are_not_guessed(tmp_path):
    _whisper(tmp_path / "whisper-large-v3")
    _whisper(tmp_path / "whisper-large-v2")
    profile = {"model_path": ""}
    assert resolve_selected_model_paths("asr", "whisper", profile, root=tmp_path) == profile


def test_requested_turbo_is_selected_only_when_unique(tmp_path):
    found = _whisper(tmp_path / "models" / "faster-whisper-large-v3-turbo")
    _whisper(tmp_path / "models" / "faster-whisper-large-v3")
    profile = {"model_path": "models/whisper-large-v3-turbo"}
    result = resolve_selected_model_paths("asr", "whisper", profile, root=tmp_path)
    assert result["model_path"] == str(found.resolve())


def test_relocate_qwen_main_and_projector_without_cross_loading(tmp_path):
    main = _gguf(tmp_path / "qwen3-asr" / "Qwen3-ASR-1.7B-Q8_0.gguf")
    proj = _gguf(tmp_path / "qwen3-asr" / "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf")
    profile = {
        "model_path": r"C:\Games\video-translate-universal\models\qwen3-asr\Qwen3-ASR-1.7B-Q8_0.gguf",
        "mmproj_path": r"C:\Games\video-translate-universal\models\qwen3-asr\mmproj-Qwen3-ASR-1.7B-Q8_0.gguf",
    }
    result = resolve_selected_model_paths("asr", "qwen3-asr", profile, root=tmp_path)
    assert result == {"model_path": str(main.resolve()), "mmproj_path": str(proj.resolve())}


def test_relocate_translator_gguf_from_old_project_directory(tmp_path):
    model = _gguf(tmp_path / "plamo3" / "plamo-3-translate-8b-Q3_K_M.gguf")
    result = resolve_selected_model_paths(
        "translate", "plamo8b",
        {"model_path": "C:/Games/video-translate-universal/models/plamo3/plamo-3-translate-8b-Q3_K_M.gguf"},
        root=tmp_path,
    )
    assert result["model_path"] == str(model.resolve())


def test_gguf_requires_valid_header(tmp_path):
    _gguf(tmp_path / "plamo3" / "broken.gguf", valid=False)
    profile = {"model_path": "models/plamo3/broken.gguf"}
    assert resolve_selected_model_paths("translate", "plamo", profile, root=tmp_path) == profile


def test_ambiguous_gguf_basename_is_not_guessed(tmp_path):
    _gguf(tmp_path / "first" / "model.gguf")
    _gguf(tmp_path / "second" / "model.gguf")
    profile = {"model_path": "C:/other/models/third/model.gguf"}
    assert resolve_selected_model_paths("translate", "gemma4", profile, root=tmp_path) == profile


def test_inventory_reports_only_valid_local_models(tmp_path):
    valid = _gguf(tmp_path / "gemma" / "good.gguf")
    _gguf(tmp_path / "gemma" / "bad.gguf", valid=False)
    whisper = _whisper(tmp_path / "whisper")
    _whisper(tmp_path / "incomplete", valid=False)
    ggufs, folders = inventory(tmp_path)
    assert ggufs == [valid]
    assert folders == [whisper]


def test_haas_backend_never_inherits_or_discovers_gguf(tmp_path):
    _gguf(tmp_path / "haas" / "something.gguf")
    profile = {"model_path": "haas-ja-v1.0"}
    assert resolve_selected_model_paths("asr", "haas", profile, root=tmp_path) == profile
