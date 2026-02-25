from __future__ import annotations

from typing import Any, Optional

from .base import (
    AdapterResolution,
    ArchitectureAdapter,
    BackendConfig,
    BackendHookSpec,
    BackendRunInputs,
    ModelBackend,
    ProjectionSpec,
    is_hooked_transformer_model,
)
from .hf_backend import HFLLMBackend
from .operators import ProjectionAttentionOperator
from .registry import (
    ResolutionError,
    get_registered_architecture_adapters,
    inspect_model_architecture,
    register_architecture_adapter,
    reset_architecture_adapter_registry,
)
from .tlens_backend import TLensBackend


def resolve_backend(model: Any, backend: Optional[ModelBackend]) -> ModelBackend:
    if backend is not None:
        return backend

    if is_hooked_transformer_model(model):
        return TLensBackend(model)

    raise ValueError(
        "A backend must be provided when model is not a transformer_lens.HookedTransformer. "
        "Instantiate HFLLMBackend(model=...) or TLensBackend(model=...) and pass backend=..."
    )


def is_hf_backend(backend: ModelBackend) -> bool:
    return getattr(backend, "kind", "") == "hf"


__all__ = [
    "AdapterResolution",
    "ArchitectureAdapter",
    "BackendConfig",
    "BackendHookSpec",
    "BackendRunInputs",
    "HFLLMBackend",
    "ModelBackend",
    "ProjectionSpec",
    "ProjectionAttentionOperator",
    "ResolutionError",
    "TLensBackend",
    "get_registered_architecture_adapters",
    "inspect_model_architecture",
    "is_hf_backend",
    "register_architecture_adapter",
    "reset_architecture_adapter_registry",
    "resolve_backend",
]
