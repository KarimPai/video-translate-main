"""Model registry — discover, register, and instantiate backends by name.

The registry holds factory functions for ASR, translation, and VAD backends.
Each backend declares its capabilities (runtime, device, language) so the
pipeline can validate compatibility before loading.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackendCapabilities:
    """What a backend can do."""

    # Inference runtime: "speech_cpp", "llama_cpp", "faster_whisper", "native", etc.
    runtime: str
    # Device: "cpu", "cuda", "vulkan", or "auto"
    device: str = "auto"
    # Source language for ASR; target language for translation
    source_lang: str = "ja"
    target_lang: str = "en"
    # Estimated VRAM usage in GB (0 for CPU-only)
    vram_gb: float = 0.0
    # Description shown in GUI
    description: str = ""


@dataclass
class BackendEntry:
    """Registered backend with factory and capabilities."""

    name: str
    backend_type: str  # "asr", "translate", "vad"
    factory: Callable[..., Any]  # (config_dict, **kwargs) -> backend instance
    capabilities: BackendCapabilities
    config_schema: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """Central registry for all available backends.

    Usage:
        registry = ModelRegistry()
        registry.register_asr("haas", haas_factory, capabilities)
        backend = registry.create_asr("haas", config_dict)
    """

    def __init__(self) -> None:
        self._backends: dict[str, dict[str, BackendEntry]] = {
            "asr": {},
            "translate": {},
            "vad": {},
        }

    def register_asr(
        self,
        name: str,
        factory: Callable[..., Any],
        capabilities: BackendCapabilities,
        config_schema: dict[str, Any] | None = None,
    ) -> None:
        """Register an ASR backend."""
        self._backends["asr"][name] = BackendEntry(
            name=name,
            backend_type="asr",
            factory=factory,
            capabilities=capabilities,
            config_schema=config_schema or {},
        )
        logger.info(
            "Registered ASR backend '%s' (runtime=%s, device=%s)",
            name, capabilities.runtime, capabilities.device,
        )

    def register_translate(
        self,
        name: str,
        factory: Callable[..., Any],
        capabilities: BackendCapabilities,
        config_schema: dict[str, Any] | None = None,
    ) -> None:
        """Register a translation backend."""
        self._backends["translate"][name] = BackendEntry(
            name=name,
            backend_type="translate",
            factory=factory,
            capabilities=capabilities,
            config_schema=config_schema or {},
        )
        logger.info(
            "Registered translation backend '%s' (runtime=%s, device=%s)",
            name, capabilities.runtime, capabilities.device,
        )

    def register_vad(
        self,
        name: str,
        factory: Callable[..., Any],
        capabilities: BackendCapabilities,
        config_schema: dict[str, Any] | None = None,
    ) -> None:
        """Register a VAD backend."""
        self._backends["vad"][name] = BackendEntry(
            name=name,
            backend_type="vad",
            factory=factory,
            capabilities=capabilities,
            config_schema=config_schema or {},
        )
        logger.info(
            "Registered VAD backend '%s' (runtime=%s, device=%s)",
            name, capabilities.runtime, capabilities.device,
        )

    def list_backends(self, backend_type: str | None = None) -> dict[str, list[str]]:
        """List registered backend names by type."""
        if backend_type:
            return {backend_type: list(self._backends[backend_type].keys())}
        return {k: list(v.keys()) for k, v in self._backends.items()}

    def get_entry(self, backend_type: str, name: str) -> BackendEntry | None:
        """Get a registered backend entry."""
        return self._backends.get(backend_type, {}).get(name)

    def create_asr(self, name: str, config: dict[str, Any], **kwargs: Any) -> Any:
        """Create an ASR backend by name."""
        entry = self.get_entry("asr", name)
        if entry is None:
            available = list(self._backends["asr"].keys())
            raise ValueError(
                f"Unknown ASR backend '{name}'. Available: {available}"
            )
        logger.info("Creating ASR backend '%s'", name)
        return entry.factory(config, **kwargs)

    def create_translate(self, name: str, config: dict[str, Any], **kwargs: Any) -> Any:
        """Create a translation backend by name."""
        entry = self.get_entry("translate", name)
        if entry is None:
            available = list(self._backends["translate"].keys())
            raise ValueError(
                f"Unknown translation backend '{name}'. Available: {available}"
            )
        logger.info("Creating translation backend '%s'", name)
        return entry.factory(config, **kwargs)

    def create_vad(self, name: str, config: dict[str, Any], **kwargs: Any) -> Any:
        """Create a VAD backend by name."""
        entry = self.get_entry("vad", name)
        if entry is None:
            available = list(self._backends["vad"].keys())
            raise ValueError(
                f"Unknown VAD backend '{name}'. Available: {available}"
            )
        logger.info("Creating VAD backend '%s'", name)
        return entry.factory(config, **kwargs)

    def create_backend(self, backend_type: str, name: str, config: Any, **kwargs: Any) -> Any:
        """Create a backend by type and name. Dispatches to the specific create method."""
        if backend_type == "asr":
            return self.create_asr(name, config, **kwargs)
        elif backend_type == "translate":
            return self.create_translate(name, config, **kwargs)
        elif backend_type == "vad":
            return self.create_vad(name, config, **kwargs)
        else:
            raise ValueError(f"Unknown backend type '{backend_type}'. Available: asr, translate, vad")

    def get_capabilities(self, backend_type: str, name: str) -> BackendCapabilities | None:
        """Get capabilities for a registered backend."""
        entry = self.get_entry(backend_type, name)
        if entry is None:
            return None
        return entry.capabilities


# Global default registry instance
_default_registry: ModelRegistry | None = None


def get_default_registry() -> ModelRegistry:
    """Get the global default registry, creating it if needed."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ModelRegistry()
        _register_builtin_backends(_default_registry)
    return _default_registry


def _register_builtin_backends(registry: ModelRegistry) -> None:
    """Register built-in backends that are always available."""
    # Silero VAD is always available
    registry.register_vad(
        "silero",
        factory=_silero_vad_factory,
        capabilities=BackendCapabilities(
            runtime="silero",
            device="cpu",
            description="Silero streaming VAD (ONNX, CPU)",
        ),
    )

    # Haas ASR
    registry.register_asr(
        "haas",
        factory=_haas_asr_factory,
        capabilities=BackendCapabilities(
            runtime="speech_cpp",
            device="vulkan",
            source_lang="ja",
            vram_gb=0.6,
            description="Haas-ja-v1.0 via speech.cpp (Vulkan GPU)",
        ),
    )

    # Qwen3-ASR
    registry.register_asr(
        "qwen3-asr",
        factory=_qwen3_asr_factory,
        capabilities=BackendCapabilities(
            runtime="llama_cpp",
            device="cuda",
            source_lang="ja",
            vram_gb=2.0,
            description="Qwen3-ASR 1.7B via llama.cpp (CUDA GPU, multimodal)",
        ),
    )

    # Whisper Large-v3 Turbo
    registry.register_asr(
        "whisper",
        factory=_whisper_asr_factory,
        capabilities=BackendCapabilities(
            runtime="faster_whisper",
            device="cuda",
            source_lang="ja",
            vram_gb=1.7,
            description="Whisper Large-v3 Turbo via faster-whisper (CPU/CUDA)",
        ),
    )

    # Gemma 4 E4B translation
    registry.register_translate(
        "gemma4",
        factory=_gemma4_translate_factory,
        capabilities=BackendCapabilities(
            runtime="llama_cpp",
            device="cuda",
            source_lang="ja",
            target_lang="en",
            vram_gb=5.0,
            description="Gemma 4 E4B via llama.cpp (CUDA GPU)",
        ),
    )

    # PLaMo 3 Translate
    registry.register_translate(
        "plamo",
        factory=_plamo_translate_factory,
        capabilities=BackendCapabilities(
            runtime="llama_cpp",
            device="cuda",
            source_lang="ja",
            target_lang="en",
            vram_gb=1.8,
            description="PLaMo 3 Translate 2B via llama.cpp (CUDA GPU)",
        ),
    )

    # PLaMo 3 Translate 8B
    # VRAM breakdown for Q3_K_M 8B with context_size=8192, parallel_slots=2:
    #   - Model weights: ~4.0 GB (Q3_K_M quantization)
    #   - KV cache: ~1.0 GB (8192 context × 2 slots × ~64B per token)
    #   - Compute buffers: ~0.5 GB (FP16 intermediate activations)
    #   - Total: ~5.5 GB
    registry.register_translate(
        "plamo8b",
        factory=_plamo8b_translate_factory,
        capabilities=BackendCapabilities(
            runtime="llama_cpp",
            device="cuda",
            source_lang="ja",
            target_lang="en",
            vram_gb=5.5,
            description="PLaMo 3 Translate 8B via llama.cpp (CUDA GPU)",
        ),
    )

    # Hy-MT2 7B translation
    registry.register_translate(
        "hymt",
        factory=_hymt_translate_factory,
        capabilities=BackendCapabilities(
            runtime="llama_cpp",
            device="cuda",
            source_lang="ja",
            target_lang="en",
            vram_gb=4.5,
            description="Hy-MT2 7B via llama.cpp (CUDA GPU)",
        ),
    )


def _silero_vad_factory(config: Any, **kwargs: Any) -> Any:
    """Factory for Silero VAD Segmenter."""
    from .config import VadConfig
    from .vad import Segmenter
    if isinstance(config, dict):
        config = VadConfig(**config)
    return Segmenter(config)


def _haas_asr_factory(config: Any, **kwargs: Any) -> Any:
    """Factory for Haas-ja-v1.0 ASR."""
    from .asr.haas import HaasASR
    from .config import AsrConfig
    if isinstance(config, dict):
        config = dict(config)
        # Haas is the default ASR backend — if model_path is missing, use the
        # compatible default "haas-ja-v1.0". But if it was explicitly cleared
        # (empty string), keep it empty so the backend constructor raises a
        # clear missing-model error.
        if "model_path" not in config:
            config["model_path"] = "haas-ja-v1.0"
        config = AsrConfig(**config)
    context = kwargs.pop("context", "")
    # Support both positional and keyword args for context
    if isinstance(context, str):
        return HaasASR(config, context=context)
    # If context was passed as positional (old signature), use kwargs
    kwargs.pop("glossary", None)
    return HaasASR(config, context=context)


def _qwen3_asr_factory(config: Any, **kwargs: Any) -> Any:
    """Factory for Qwen3-ASR.

    Qwen3-ASR must never inherit Haas's model path from dataclass defaults.
    Raises clear errors if model_path or mmproj_path are missing.
    """
    from .asr.qwen3_asr import Qwen3ASR
    from .config import AsrConfig
    if isinstance(config, dict):
        config = dict(config)
        # Qwen3-ASR is NOT the default ASR — do NOT inherit Haas model path.
        # If model_path is missing from config, set it empty so the backend
        # constructor raises a clear missing-model error.
        if "model_path" not in config:
            config["model_path"] = ""
        if "mmproj_path" not in config:
            config["mmproj_path"] = ""
        config = AsrConfig(**config)
    # Validate Qwen3-ASR required fields before attempting to load
    if not config.model_path or not config.model_path.strip():
        raise ValueError(
            "Qwen3-ASR backend selected but no Qwen3-ASR model is configured. "
            "Set asr.model_path to a Qwen3-ASR GGUF model file, "
            "or configure [asr.qwen3-asr] model_path in config.toml."
        )
    if not config.mmproj_path or not config.mmproj_path.strip():
        raise ValueError(
            "Qwen3-ASR backend selected but no audio projector (mmproj) is configured. "
            "Set asr.mmproj_path to a Qwen3-ASR mmproj GGUF file, "
            "or configure [asr.qwen3-asr] mmproj_path in config.toml."
        )
    context = kwargs.pop("context", "")
    return Qwen3ASR(config, context=context)


def _whisper_asr_factory(config: Any, **kwargs: Any) -> Any:
    """Factory for Whisper Large-v3 Turbo.

    Passes is_explicit flag so Whisper knows whether the model path was
    explicitly configured (GUI/CLI override or backend profile) vs inherited
    from common config (e.g., Haas default).

    Whisper must never inherit Haas's model path from dataclass defaults.
    """
    from .asr.whisper_turbo import WhisperASR
    from .config import AsrConfig
    if isinstance(config, dict):
        config = dict(config)
        # Determine if model_path was explicitly set for this backend
        is_explicit = kwargs.pop("_whisper_is_explicit", False)
        # Whisper is NOT the default ASR — do NOT inherit Haas model path.
        # If model_path is present in config (even empty), use it as-is.
        # If absent, set empty so Whisper searches its own HF cache.
        if "model_path" not in config:
            config["model_path"] = ""
            is_explicit = False
        config = AsrConfig(**config)
    else:
        is_explicit = False
    context = kwargs.pop("context", "")
    return WhisperASR(config, context=context, is_explicit=is_explicit)


def _gemma4_translate_factory(config: Any, **kwargs: Any) -> Any:
    """Factory for Gemma 4 E4B translation."""
    from .config import TranslateConfig
    from .glossary import Glossary
    from .stream_context import StreamContext
    from .translate.gemma4 import Gemma4Translator
    if isinstance(config, dict):
        config = dict(config)
        # Gemma4 is the default translation backend — if model_path is missing,
        # use the compatible default. But if explicitly cleared, keep it empty.
        if "model_path" not in config:
            config["model_path"] = "models/gemma4/google_gemma-4-E4B-it-Q4_K_M.gguf"
        config = TranslateConfig(**config)
    glossary = kwargs.get("glossary", Glossary({}))
    stream = kwargs.get("stream", StreamContext())
    gpu_layers = kwargs.get("gpu_layers", None)
    return Gemma4Translator(config, glossary, stream=stream, gpu_layers=gpu_layers)


def _plamo_translate_factory(config: Any, **kwargs: Any) -> Any:
    """Factory for PLaMo 3 Translate translation.

    PLaMo must never inherit Gemma4's model path from dataclass defaults.
    Raises a clear error if no PLaMo model is configured.
    """
    from .config import TranslateConfig
    from .glossary import Glossary
    from .stream_context import StreamContext
    from .translate.plamo import PlamoTranslator
    if isinstance(config, dict):
        config = dict(config)
        # PLaMo is NOT the default translation backend — do NOT inherit Gemma4 path.
        if "model_path" not in config:
            config["model_path"] = ""
        config = TranslateConfig(**config)
    # Validate PLaMo model is configured before attempting to load it
    if not config.model_path or not config.model_path.strip():
        raise ValueError(
            "PLaMo translation backend selected but no PLaMo model is configured. "
            "Set translate.model_path to a PLaMo 3 Translate GGUF model file, "
            "or configure [translate.plamo] model_path in config.toml."
        )
    glossary = kwargs.get("glossary", Glossary({}))
    stream = kwargs.get("stream", StreamContext())
    gpu_layers = kwargs.get("gpu_layers", None)
    return PlamoTranslator(config, glossary, stream=stream, gpu_layers=gpu_layers)


def _hymt_translate_factory(config: Any, **kwargs: Any) -> Any:
    """Factory for Hy-MT2 7B translation.

    Hy-MT2 must never inherit Gemma4's model path from dataclass defaults.
    Raises a clear error if no Hy-MT2 model is configured.
    """
    from .config import TranslateConfig
    from .glossary import Glossary
    from .stream_context import StreamContext
    from .translate.hymt import HymtTranslator
    if isinstance(config, dict):
        config = dict(config)
        # Hy-MT2 is NOT the default translation backend — do NOT inherit Gemma4 path.
        if "model_path" not in config:
            config["model_path"] = ""
        config = TranslateConfig(**config)
    # Validate Hy-MT2 model is configured before attempting to load it
    if not config.model_path or not config.model_path.strip():
        raise ValueError(
            "Hy-MT2 translation backend selected but no Hy-MT2 model is configured. "
            "Set translate.model_path to a Hy-MT2 7B GGUF model file, "
            "or configure [translate.hymt] model_path in config.toml."
        )
    glossary = kwargs.get("glossary", Glossary({}))
    stream = kwargs.get("stream", StreamContext())
    gpu_layers = kwargs.get("gpu_layers", None)
    return HymtTranslator(config, glossary, stream=stream, gpu_layers=gpu_layers)


def _plamo8b_translate_factory(config: Any, **kwargs: Any) -> Any:
    """Factory for PLaMo 3 Translate 8B.

    Reuses the same PlamoTranslator class as PLaMo 2B but with a different
    model path. PLaMo 8B must never inherit Gemma4's or PLaMo 2B's model path.
    """
    from .config import TranslateConfig
    from .glossary import Glossary
    from .stream_context import StreamContext
    from .translate.plamo import PlamoTranslator
    if isinstance(config, dict):
        config = dict(config)
        # PLaMo 8B is NOT the default translation backend — do NOT inherit Gemma4 path.
        if "model_path" not in config:
            config["model_path"] = ""
        config = TranslateConfig(**config)
    # Validate PLaMo 8B model is configured before attempting to load it
    if not config.model_path or not config.model_path.strip():
        raise ValueError(
            "PLaMo 8B translation backend selected but no PLaMo 8B model is configured. "
            "Set translate.model_path to a PLaMo 3 Translate 8B GGUF model file, "
            "or configure [translate.plamo8b] model_path in config.toml."
        )
    glossary = kwargs.get("glossary", Glossary({}))
    stream = kwargs.get("stream", StreamContext())
    gpu_layers = kwargs.get("gpu_layers", None)
    return PlamoTranslator(config, glossary, stream=stream, gpu_layers=gpu_layers)


__all__ = [
    "BackendCapabilities",
    "BackendEntry",
    "ModelRegistry",
    "_plamo8b_translate_factory",
    "get_default_registry",
]
