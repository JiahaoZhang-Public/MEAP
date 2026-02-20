"""Stable high-level API surface for meap.

Use this module for production usage. Lower-level implementation modules are internal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Literal, Optional, Union

import torch
from torch import Tensor

from .attribute import attribute
from .backend import ModelBackend, resolve_backend
from .batch import BatchLike, PairBatchPreparer, PreparedBatch, iter_prepared_batches
from .evaluate import evaluate_baseline, evaluate_graph
from .graph import Graph
from .preparer import HFProcessorAdapter

MetricFn = Callable[[Tensor, Optional[Tensor], PreparedBatch], Tensor]


@dataclass
class AttributionRunResult:
    graph: Graph
    scores: Tensor


def _resolve_pair_batch_preparer(
    backend_obj: ModelBackend,
    *,
    processor: Optional[Any],
    pair_batch_preparer: Optional[PairBatchPreparer],
    processor_kwargs: Optional[dict[str, Any]],
) -> Optional[PairBatchPreparer]:
    if processor is not None and pair_batch_preparer is not None:
        raise ValueError("Provide either processor or pair_batch_preparer, not both")

    if pair_batch_preparer is not None:
        return pair_batch_preparer

    if processor is None:
        if processor_kwargs is not None:
            raise ValueError("processor_kwargs requires processor=... to be provided")
        return None

    return HFProcessorAdapter(
        processor=processor,
        processor_kwargs=processor_kwargs,
        device=backend_obj.config.device,
    )


def _prepare_batch_stream(
    backend_obj: ModelBackend,
    batches: Iterable[BatchLike],
    *,
    max_length: Optional[int],
    pair_batch_preparer: Optional[PairBatchPreparer],
):
    return iter_prepared_batches(
        tokenization_model=backend_obj.tokenization_model,
        batches=batches,
        max_length=max_length,
        pair_batch_preparer=pair_batch_preparer,
    )


def attribute_from_dataloader(
    model: Any,
    dataloader: Iterable[BatchLike],
    metric: MetricFn,
    *,
    backend: Optional[ModelBackend] = None,
    graph: Optional[Graph] = None,
    processor: Optional[Any] = None,
    pair_batch_preparer: Optional[PairBatchPreparer] = None,
    processor_kwargs: Optional[dict[str, Any]] = None,
    method: Literal[
        "EAP",
        "EAP-IG-inputs",
        "clean-corrupted",
        "EAP-IG-activations",
        "exact",
        "smoke",
    ] = "smoke",
    intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
    aggregation: Literal["sum", "mean"] = "sum",
    ig_steps: Optional[int] = None,
    intervention_dataloader: Optional[Iterable[BatchLike]] = None,
    max_length: Optional[int] = None,
    quiet: bool = False,
) -> AttributionRunResult:
    """High-level attribution API with user-managed data preparation.

    Users should provide either:
    1) dataloader entries that are already PreparedBatch, or
    2) raw clean/corrupt entries and explicit processor=... / pair_batch_preparer=....
    """

    backend_obj = resolve_backend(model, backend)
    if max_length is not None:
        raise ValueError(
            "max_length is not applied by high-level API anymore. "
            "Please truncate inside your processor/preparer."
        )
    run_graph = graph if graph is not None else Graph.from_model(backend_obj.config)

    resolved_preparer = _resolve_pair_batch_preparer(
        backend_obj,
        processor=processor,
        pair_batch_preparer=pair_batch_preparer,
        processor_kwargs=processor_kwargs,
    )

    prepared_batches = _prepare_batch_stream(
        backend_obj,
        dataloader,
        max_length=max_length,
        pair_batch_preparer=resolved_preparer,
    )

    prepared_intervention_batches = None
    if intervention_dataloader is not None:
        prepared_intervention_batches = _prepare_batch_stream(
            backend_obj,
            intervention_dataloader,
            max_length=max_length,
            pair_batch_preparer=resolved_preparer,
        )

    scores = attribute(
        model=model,
        graph=run_graph,
        batches=prepared_batches,
        metric=metric,
        backend=backend_obj,
        method=method,
        intervention=intervention,
        aggregation=aggregation,
        ig_steps=ig_steps,
        intervention_batches=prepared_intervention_batches,
        quiet=quiet,
    )

    return AttributionRunResult(graph=run_graph, scores=scores)


def evaluate_graph_from_dataloader(
    model: Any,
    graph: Graph,
    dataloader: Iterable[BatchLike],
    metrics: Union[MetricFn, List[MetricFn]],
    *,
    backend: Optional[ModelBackend] = None,
    processor: Optional[Any] = None,
    pair_batch_preparer: Optional[PairBatchPreparer] = None,
    processor_kwargs: Optional[dict[str, Any]] = None,
    max_length: Optional[int] = None,
    quiet: bool = False,
    intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
    intervention_dataloader: Optional[Iterable[BatchLike]] = None,
    skip_clean: bool = True,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    backend_obj = resolve_backend(model, backend)
    if max_length is not None:
        raise ValueError(
            "max_length is not applied by high-level API anymore. "
            "Please truncate inside your processor/preparer."
        )
    resolved_preparer = _resolve_pair_batch_preparer(
        backend_obj,
        processor=processor,
        pair_batch_preparer=pair_batch_preparer,
        processor_kwargs=processor_kwargs,
    )

    prepared_batches = _prepare_batch_stream(
        backend_obj,
        dataloader,
        max_length=max_length,
        pair_batch_preparer=resolved_preparer,
    )

    prepared_intervention_batches = None
    if intervention_dataloader is not None:
        prepared_intervention_batches = _prepare_batch_stream(
            backend_obj,
            intervention_dataloader,
            max_length=max_length,
            pair_batch_preparer=resolved_preparer,
        )

    return evaluate_graph(
        model=model,
        graph=graph,
        batches=prepared_batches,
        metrics=metrics,
        backend=backend_obj,
        quiet=quiet,
        intervention=intervention,
        intervention_batches=prepared_intervention_batches,
        skip_clean=skip_clean,
    )


def evaluate_baseline_from_dataloader(
    model: Any,
    dataloader: Iterable[BatchLike],
    metrics: Union[MetricFn, List[MetricFn]],
    *,
    backend: Optional[ModelBackend] = None,
    processor: Optional[Any] = None,
    pair_batch_preparer: Optional[PairBatchPreparer] = None,
    processor_kwargs: Optional[dict[str, Any]] = None,
    max_length: Optional[int] = None,
    run_corrupted: bool = False,
    quiet: bool = False,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    backend_obj = resolve_backend(model, backend)
    if max_length is not None:
        raise ValueError(
            "max_length is not applied by high-level API anymore. "
            "Please truncate inside your processor/preparer."
        )
    resolved_preparer = _resolve_pair_batch_preparer(
        backend_obj,
        processor=processor,
        pair_batch_preparer=pair_batch_preparer,
        processor_kwargs=processor_kwargs,
    )

    prepared_batches = _prepare_batch_stream(
        backend_obj,
        dataloader,
        max_length=max_length,
        pair_batch_preparer=resolved_preparer,
    )

    return evaluate_baseline(
        model=model,
        batches=prepared_batches,
        metrics=metrics,
        backend=backend_obj,
        run_corrupted=run_corrupted,
        quiet=quiet,
    )


__all__ = [
    "AttributionRunResult",
    "MetricFn",
    "attribute_from_dataloader",
    "evaluate_baseline_from_dataloader",
    "evaluate_graph_from_dataloader",
]
