"""Public package interface for meap.

Stability policy:
- Symbols exported here are the supported package surface.
- High-level APIs in `api.py` are the primary stable entrypoints.
- Internal implementation modules (for example `attribute.py`, `evaluate.py`, `utils.py`)
  are not part of the stability contract and may change between minor releases.
"""

from __future__ import annotations

from .api import (
    AttributionModel,
    AttributionResult,
    PreparedInputLike,
    RouteInfo,
    TaskSpec,
    evaluate_baseline_from_dataloader,
    evaluate_graph_from_dataloader,
)
from .attribute import attribute
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
from .preparer import HFProcessorAdapter, prepare_pair_batch_with_processor

__all__ = [
    "AttributionModel",
    "AttributionResult",
    "DEFAULT_BACKBONE_MODEL_ID",
    "DEFAULT_MULTIMODAL_MODEL_ID",
    "Graph",
    "PreparedBatch",
    "PreparedInputLike",
    "RawPairBatch",
    "RouteInfo",
    "HFProcessorAdapter",
    "HFLLMBackend",
    "TLensBackend",
    "BackendConfig",
    "BackendRunInputs",
    "ArchitectureAdapter",
    "TaskSpec",
    "attribute",
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
]
