"""Model registry — discover, register, and instantiate backends by name.

The registry holds factory functions for ASR, translation, and VAD backends.
Each backend declares its capabilities (runtime, device, language) so the
pipeline can validate compatibility before loading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

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
    # FireRed VAD is always available
    registry.register_vad(
        "firered",
        factory=_firered_vad_factory,
        capabilities=BackendCapabilities(
            runtime="firered",
            device="cpu",
            description="FireRed streaming VAD (CPU)",
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


def _firered_vad_factory(config: Any, **kwargs: Any) -> Any:
    """Factory for FireRed VAD Segmenter."""
    from .config import VadConfig
    from .vad import Segmenter
    if isinstance(config, dict):
        config = VadConfig(**config)
    return Segmenter(config)


def _haas_asr_factory(config: Any, **kwargs: Any) -> Any:
    """Factory for Haas-ja-v1.0 ASR."""
    from .config import AsrConfig
    from .asr.haas import HaasASR
    if isinstance(config, dict):
        config = AsrConfig(**config)
    context = kwargs.pop("context", "")
    # Support both positional and keyword args for context
    if isinstance(context, str):
        return HaasASR(config, context=context)
    # If context was passed as positional (old signature), use kwargs
    glossary = kwargs.pop("glossary", None)
    return HaasASR(config, context=context)


def _qwen3_asr_factory(config: Any, **kwargs: Any) -> Any:
    """Factory for Qwen3-ASR."""
    from .config import AsrConfig
    from .asr.qwen3_asr import Qwen3ASR
    if isinstance(config, dict):
        config = AsrConfig(**config)
    context = kwargs.pop("context", "")
    return Qwen3ASR(config, context=context)


def _whisper_asr_factory(config: Any, **kwargs: Any) -> Any:
    """Factory for Whisper Large-v3 Turbo."""
    from .config import AsrConfig
    from .asr.whisper_turbo import WhisperASR
    if isinstance(config, dict):
        config = AsrConfig(**config)
    context = kwargs.pop("context", "")
    return WhisperASR(config, context=context)


def _gemma4_translate_factory(config: Any, **kwargs: Any) -> Any:
    """Factory for Gemma 4 E4B translation."""
    from .config import TranslateConfig
    from .glossary import Glossary
    from .stream_context import StreamContext
    from .translate.gemma4 import Gemma4Translator
    if isinstance(config, dict):
        config = TranslateConfig(**config)
    glossary = kwargs.get("glossary", Glossary({}))
    stream = kwargs.get("stream", StreamContext())
    gpu_layers = kwargs.get("gpu_layers", None)
    return Gemma4Translator(config, glossary, stream=stream, gpu_layers=gpu_layers)


def _plamo_translate_factory(config: Any, **kwargs: Any) -> Any:
    """Factory for PLaMo 3 Translate translation."""
    from .config import TranslateConfig
    from .glossary import Glossary
    from .stream_context import StreamContext
    from .translate.plamo import PlamoTranslator
    if isinstance(config, dict):
        config = TranslateConfig(**config)
    glossary = kwargs.get("glossary", Glossary({}))
    stream = kwargs.get("stream", StreamContext())
    gpu_layers = kwargs.get("gpu_layers", None)
    return PlamoTranslator(config, glossary, stream=stream, gpu_layers=gpu_layers)


def _hymt_translate_factory(config: Any, **kwargs: Any) -> Any:
    """Factory for Hy-MT2 7B translation."""
    from .config import TranslateConfig
    from .glossary import Glossary
    from .stream_context import StreamContext
    from .translate.hymt import HymtTranslator
    if isinstance(config, dict):
        config = TranslateConfig(**config)
    glossary = kwargs.get("glossary", Glossary({}))
    stream = kwargs.get("stream", StreamContext())
    gpu_layers = kwargs.get("gpu_layers", None)
    return HymtTranslator(config, glossary, stream=stream, gpu_layers=gpu_layers)


__all__ = [
    "BackendCapabilities",
    "BackendEntry",
    "ModelRegistry",
    "get_default_registry",
]
