"""Public package interface for meap.

Stability policy:
- Symbols exported here are the supported package surface.
- High-level APIs in `api.py` are the primary stable entrypoints.
- Internal implementation modules (for example `attribute.py`, `evaluate.py`, `utils.py`)
  are not part of the stability contract and may change between minor releases.
"""

from __future__ import annotations

from typing import Any
import warnings

from .api import (
    AttributionRunResult,
    CircuitEdgeSummary,
    CircuitRunResult,
    TaskSpec,
    attribute_from_dataloader,
    discover_circuit,
    evaluate_baseline_from_dataloader,
    evaluate_graph_from_dataloader,
)
from .attribute import (
    attribute,
)
from .attribute import (
    get_real_edge_scores as _get_real_edge_scores,
)
from .attribute import (
    get_scores_clean_corrupted as _get_scores_clean_corrupted,
)
from .attribute import (
    get_scores_eap as _get_scores_eap,
)
from .attribute import (
    get_scores_eap_ig as _get_scores_eap_ig,
)
from .attribute import (
    get_scores_exact as _get_scores_exact,
)
from .attribute import (
    get_scores_ig_activations as _get_scores_ig_activations,
)
from .attribute import (
    get_scores_smoke as _get_scores_smoke,
)
from .backend import (
    ArchitectureAdapter,
    BackendConfig,
    BackendRunInputs,
    HFLLMBackend,
    TLensBackend,
    inspect_model_architecture,
    register_architecture_adapter,
    resolve_backend,
)
from .batch import (
    PreparedBatch,
    RawPairBatch,
    iter_prepared_batches,
    text_batch_to_prepared_batch,
    validate_prepared_batch,
)
from .catalog import (
    list_official_models,
    list_supported_architectures,
)
from .config import DEFAULT_BACKBONE_MODEL_ID, DEFAULT_MULTIMODAL_MODEL_ID
from .evaluate import evaluate_baseline, evaluate_graph
from .graph import Graph
from .preparer import (
    HFProcessorAdapter,
    prepare_pair_batch_with_processor,
)
from .preparer import (
    build_default_llava_processor as _build_default_llava_processor,
)
from .preparer import (
    prepare_llava_token_pair_batch as _prepare_llava_token_pair_batch,
)

_DEPRECATION_VERSION = "1.0.0"
_REMOVAL_VERSION = "1.2.0"


def _warn_deprecated_public_symbol(symbol: str, replacement: str) -> None:
    warnings.warn(
        (
            f"`meap.{symbol}` is deprecated since {_DEPRECATION_VERSION} "
            f"and will be removed in {_REMOVAL_VERSION}. "
            f"Use `{replacement}` instead."
        ),
        DeprecationWarning,
        stacklevel=2,
    )


def get_real_edge_scores(*args: Any, **kwargs: Any):
    _warn_deprecated_public_symbol("get_real_edge_scores", "meap.attribute.get_real_edge_scores")
    return _get_real_edge_scores(*args, **kwargs)


def get_scores_clean_corrupted(*args: Any, **kwargs: Any):
    _warn_deprecated_public_symbol(
        "get_scores_clean_corrupted", "meap.attribute.get_scores_clean_corrupted"
    )
    return _get_scores_clean_corrupted(*args, **kwargs)


def get_scores_eap(*args: Any, **kwargs: Any):
    _warn_deprecated_public_symbol("get_scores_eap", "meap.attribute.get_scores_eap")
    return _get_scores_eap(*args, **kwargs)


def get_scores_eap_ig(*args: Any, **kwargs: Any):
    _warn_deprecated_public_symbol("get_scores_eap_ig", "meap.attribute.get_scores_eap_ig")
    return _get_scores_eap_ig(*args, **kwargs)


def get_scores_exact(*args: Any, **kwargs: Any):
    _warn_deprecated_public_symbol("get_scores_exact", "meap.attribute.get_scores_exact")
    return _get_scores_exact(*args, **kwargs)


def get_scores_ig_activations(*args: Any, **kwargs: Any):
    _warn_deprecated_public_symbol(
        "get_scores_ig_activations", "meap.attribute.get_scores_ig_activations"
    )
    return _get_scores_ig_activations(*args, **kwargs)


def get_scores_smoke(*args: Any, **kwargs: Any):
    _warn_deprecated_public_symbol("get_scores_smoke", "meap.attribute.get_scores_smoke")
    return _get_scores_smoke(*args, **kwargs)


def build_default_llava_processor(*args: Any, **kwargs: Any):
    _warn_deprecated_public_symbol(
        "build_default_llava_processor", "meap.preparer.build_default_llava_processor"
    )
    return _build_default_llava_processor(*args, **kwargs)


def prepare_llava_token_pair_batch(*args: Any, **kwargs: Any):
    _warn_deprecated_public_symbol(
        "prepare_llava_token_pair_batch", "meap.preparer.prepare_llava_token_pair_batch"
    )
    return _prepare_llava_token_pair_batch(*args, **kwargs)


__all__ = [
    "AttributionRunResult",
    "CircuitEdgeSummary",
    "CircuitRunResult",
    "DEFAULT_BACKBONE_MODEL_ID",
    "DEFAULT_MULTIMODAL_MODEL_ID",
    "Graph",
    "PreparedBatch",
    "RawPairBatch",
    "HFProcessorAdapter",
    "HFLLMBackend",
    "TLensBackend",
    "BackendConfig",
    "BackendRunInputs",
    "ArchitectureAdapter",
    "TaskSpec",
    "attribute",
    "attribute_from_dataloader",
    "discover_circuit",
    "evaluate_graph",
    "evaluate_graph_from_dataloader",
    "evaluate_baseline",
    "evaluate_baseline_from_dataloader",
    "iter_prepared_batches",
    "prepare_pair_batch_with_processor",
    "register_architecture_adapter",
    "inspect_model_architecture",
    "list_official_models",
    "list_supported_architectures",
    "resolve_backend",
    "text_batch_to_prepared_batch",
    "validate_prepared_batch",
    "build_default_llava_processor",
    "prepare_llava_token_pair_batch",
    "get_real_edge_scores",
    "get_scores_clean_corrupted",
    "get_scores_eap",
    "get_scores_eap_ig",
    "get_scores_exact",
    "get_scores_ig_activations",
    "get_scores_smoke",
]
