"""Exercise normal Qwen startup routing without opening a model server."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from auto_translate import registry
from auto_translate.config import Config, resolve_model_identity
from auto_translate.llama_server import LlamaServerError
from auto_translate.model_discovery import resolve_selected_model_paths


@pytest.fixture
def installed_pairs(tmp_path, monkeypatch):
    # Point registry discovery at this fixture instead of the real models.
    monkeypatch.setattr(registry, "__file__", str(tmp_path / "src/auto_translate/registry.py"))
    monkeypatch.chdir(tmp_path)
    pairs = {}
    for label, directory, stem in (
        ("standard", "qwen3-asr", "Qwen3-ASR-1.7B-Q8_0"),
        ("ja", "qwen3-asr-ja", "Qwen3-ASR-1.7B-JA-Q8_0"),
    ):
        parent = tmp_path / "models" / directory
        parent.mkdir(parents=True)
        model, projector = parent / f"{stem}.gguf", parent / f"mmproj-{stem}.gguf"
        model.write_bytes(b"GGUF")
        projector.write_bytes(b"GGUF")
        pairs[label] = model, projector
    return pairs


def test_default_selects_standard_pair_with_both_installed(installed_pairs, monkeypatch):
    constructor = Mock()
    monkeypatch.setattr("auto_translate.asr.qwen3_asr.Qwen3ASR", constructor)

    registry.get_default_registry().create_asr("qwen3-asr", {})

    cfg = constructor.call_args.args[0]
    assert (cfg.model_path, cfg.mmproj_path) == tuple(map(str, installed_pairs["standard"]))


def test_missing_standard_model_does_not_select_ja(installed_pairs):
    installed_pairs["standard"][0].unlink()
    with pytest.raises(ValueError, match="default model .* was not found"):
        registry.get_default_registry().create_asr("qwen3-asr", {})


def test_missing_standard_projector_does_not_select_ja_projector(installed_pairs):
    installed_pairs["standard"][1].unlink()
    with pytest.raises(ValueError, match="no audio projector"):
        registry.get_default_registry().create_asr("qwen3-asr", {})


@pytest.mark.parametrize("label", ["standard", "ja"])
def test_same_name_projector_elsewhere_requires_explicit_path(installed_pairs, monkeypatch, label):
    model, sibling = installed_pairs[label]
    unrelated = installed_pairs["ja" if label == "standard" else "standard"][0].parent / sibling.name
    sibling.replace(unrelated)
    constructor = Mock()
    monkeypatch.setattr("auto_translate.asr.qwen3_asr.Qwen3ASR", constructor)
    profile = {} if label == "standard" else {"model_path": str(model)}

    with pytest.raises(ValueError, match=r"no matching sibling.*--asr-mmproj"):
        registry.get_default_registry().create_asr("qwen3-asr", profile)
    constructor.assert_not_called()

    registry.get_default_registry().create_asr("qwen3-asr", {
        **profile, "mmproj_path": str(unrelated),
    })
    assert constructor.call_args.args[0].mmproj_path == str(unrelated)


@pytest.mark.parametrize("stale_folder", ["qwen3-asr", "qwen3-asr-ja"])
def test_stale_projector_path_is_not_relocated_to_an_unrelated_directory(
    tmp_path, installed_pairs, stale_folder,
):
    model, sibling = installed_pairs["standard"]
    unrelated = installed_pairs["ja"][0].parent / sibling.name
    sibling.replace(unrelated)
    stale = tmp_path / "old-project" / "models" / stale_folder / sibling.name
    profile = {"model_path": str(model), "mmproj_path": str(stale)}

    resolved = resolve_selected_model_paths("asr", "qwen3-asr", profile, root=model.parent.parent)

    assert resolved == profile
    with pytest.raises(LlamaServerError, match="Qwen3-ASR mmproj not found"):
        registry.get_default_registry().create_asr("qwen3-asr", resolved)


@pytest.mark.parametrize("missing_index", [0, 1], ids=["model", "projector"])
def test_explicit_missing_file_errors_before_startup(installed_pairs, missing_index):
    model, projector = installed_pairs["standard"]
    installed_pairs["standard"][missing_index].unlink()
    label = "model" if missing_index == 0 else "mmproj"
    with pytest.raises(LlamaServerError, match=f"Qwen3-ASR {label} not found"):
        registry.get_default_registry().create_asr("qwen3-asr", {
            "model_path": str(model), "mmproj_path": str(projector),
        })


@pytest.mark.parametrize(("model_label", "projector_label"), [("standard", "ja"), ("ja", "standard")])
def test_same_size_other_checkpoint_projector_is_rejected(
    installed_pairs, monkeypatch, model_label, projector_label,
):
    constructor = Mock()
    monkeypatch.setattr("auto_translate.asr.qwen3_asr.Qwen3ASR", constructor)
    with pytest.raises(ValueError, match="model and mmproj do not match"):
        registry.get_default_registry().create_asr("qwen3-asr", {
            "model_path": str(installed_pairs[model_label][0]),
            "mmproj_path": str(installed_pairs[projector_label][1]),
        })
    constructor.assert_not_called()


def test_gui_identity_drops_projector_only_when_checkpoint_is_overridden(installed_pairs):
    model, projector = map(str, installed_pairs["standard"])
    ja_model, ja_projector = map(str, installed_pairs["ja"])
    cfg = Config(asr_profiles={"qwen3-asr": {"model_path": model, "mmproj_path": projector}})

    def identity(**overrides):
        return resolve_model_identity(cfg, "asr", "qwen3-asr", overrides)

    assert identity(model_path=model)["mmproj_path"] == projector
    assert identity(model_path=ja_model) == {"model_path": ja_model, "mmproj_path": ""}
    assert identity(model_path=ja_model, mmproj_path=ja_projector)["mmproj_path"] == ja_projector


@pytest.mark.parametrize("entry", ["cli", "gui"])
@pytest.mark.parametrize("override", [False, True], ids=["default", "ja-override"])
def test_qwen_pair_through_cli_and_gui_launch(
    tmp_path, monkeypatch, installed_pairs, entry, override,
):
    from auto_translate import __main__, gui_runner
    from auto_translate.registry import ModelRegistry

    config = tmp_path / "config.toml"
    ja_model, ja_projector = map(str, installed_pairs["ja"])
    if override:
        # For a model-only GUI override the standard profile's projector must
        # not be carried over. CLI may supply both files explicitly.
        std_model, std_projector = map(str, installed_pairs["standard"])
        config.write_text(
            f"[asr.qwen3-asr]\nmodel_path = {json.dumps(std_model)}\n"
            f"mmproj_path = {json.dumps(std_projector)}\n",
            encoding="utf-8",
        )
    else:
        config.write_text("", encoding="utf-8")
    url = "https://www.youtube.com/watch?v=abcdefghijk"
    if entry == "gui":
        if sys.platform != "win32":
            pytest.skip("GUI launch supplies a Windows MPV window handle")
        python = tmp_path / "python.exe"
        python.with_name("pythonw.exe").touch()
        monkeypatch.setattr(gui_runner, "backend_python", lambda *args, **kwargs: python)
        _, command = gui_runner.build_backend_command(
            tmp_path, url, 480, 42, asr_backend="qwen3-asr",
            asr_model=ja_model if override else None, diagnostics=False,
        )
        args = command[3:] + ["--gui-control-port", "12345", "--gui-control-token", "test"]
    else:
        args = [url, "--asr-backend", "qwen3-asr"]
        if override:
            args += ["--asr-model", ja_model, "--asr-mmproj", ja_projector]

    class ReachedQwen(Exception):
        pass

    constructor = Mock(side_effect=ReachedQwen)
    monkeypatch.setattr("auto_translate.asr.qwen3_asr.Qwen3ASR", constructor)
    monkeypatch.setattr("auto_translate.windows_job.bind_process_tree", Mock())
    monkeypatch.setattr("auto_translate.windows_subprocess.guard_gui_spawns", Mock())
    monkeypatch.setattr("auto_translate.gui_control.BackendControl", Mock())
    monkeypatch.setattr("auto_translate.ingest.resolve_source", Mock(
        return_value=SimpleNamespace(live=False, timeline={}),
    ))
    monkeypatch.setattr("auto_translate.gpu_budget.GpuBudget", Mock())
    monkeypatch.setattr("auto_translate.runtime_diagnostics.configure_diagnostics", Mock())
    monkeypatch.setattr("auto_translate.model_discovery.log_inventory", Mock())
    monkeypatch.setattr(ModelRegistry, "create_vad", Mock())
    monkeypatch.setattr(sys, "argv", ["video-translate", *args, "--config", str(config)])
    monkeypatch.delenv("VIDEO_TRANSLATE_CLI_LOG", raising=False)

    with pytest.raises(ReachedQwen):
        __main__.main()

    selected = constructor.call_args.args[0]
    expected = installed_pairs["ja" if override else "standard"]
    assert (selected.model_path, selected.mmproj_path) == tuple(map(str, expected))
