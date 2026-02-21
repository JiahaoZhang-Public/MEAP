"""Stable high-level API surface for meap.

Use this module for production usage. Lower-level implementation modules are internal.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Literal, Optional, Union

import torch

from .attribution_model import (
    AttributionModel,
    AttributionResult,
    PreparedInputLike,
    RouteInfo,
)
from .backend import ModelBackend
from .batch import (
    BatchLike,
    PairBatchPreparer,
    iter_prepared_batches,
)
from .evaluate import evaluate_baseline, evaluate_graph
from .graph import Graph
from .tasking import MetricFn, TaskSpec


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


def evaluate_graph_from_dataloader(
    model: Any,
    graph: Graph,
    dataloader: Iterable[BatchLike],
    metrics: Union[MetricFn, List[MetricFn]],
    *,
    backend: ModelBackend,
    pair_batch_preparer: Optional[PairBatchPreparer] = None,
    max_length: Optional[int] = None,
    quiet: bool = False,
    intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
    intervention_dataloader: Optional[Iterable[BatchLike]] = None,
    skip_clean: bool = True,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    backend_obj = backend
    if max_length is not None:
        raise ValueError(
            "max_length is not applied by high-level API anymore. "
            "Please truncate inside your processor/preparer."
        )
    prepared_batches = _prepare_batch_stream(
        backend_obj,
        dataloader,
        max_length=max_length,
        pair_batch_preparer=pair_batch_preparer,
    )

    prepared_intervention_batches = None
    if intervention_dataloader is not None:
        prepared_intervention_batches = _prepare_batch_stream(
            backend_obj,
            intervention_dataloader,
            max_length=max_length,
            pair_batch_preparer=pair_batch_preparer,
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
    backend: ModelBackend,
    pair_batch_preparer: Optional[PairBatchPreparer] = None,
    max_length: Optional[int] = None,
    run_corrupted: bool = False,
    quiet: bool = False,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    backend_obj = backend
    if max_length is not None:
        raise ValueError(
            "max_length is not applied by high-level API anymore. "
            "Please truncate inside your processor/preparer."
        )
    prepared_batches = _prepare_batch_stream(
        backend_obj,
        dataloader,
        max_length=max_length,
        pair_batch_preparer=pair_batch_preparer,
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
    "AttributionModel",
    "AttributionResult",
    "MetricFn",
    "PreparedInputLike",
    "RouteInfo",
    "TaskSpec",
    "evaluate_baseline_from_dataloader",
    "evaluate_graph_from_dataloader",
]
