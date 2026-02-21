"""Stable high-level API surface for meap.

Use this module for production usage. Lower-level implementation modules are internal.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Iterable, List, Literal, Mapping, Optional, Sequence, Union

import torch
from torch import Tensor
import torch.nn.functional as F

from .attribute import attribute
from .backend import ModelBackend, resolve_backend
from .batch import (
    BatchLike,
    PairBatchPreparer,
    PreparedBatch,
    iter_prepared_batches,
    validate_prepared_batch,
)
from .evaluate import evaluate_baseline, evaluate_graph
from .graph import Graph
from .hf_loader import load_hf_backend_and_processor
from .preparer import HFProcessorAdapter

MetricFn = Callable[[Tensor, Optional[Tensor], PreparedBatch], Tensor]
TaskName = Literal["next_token", "choice_classification"]
AttributionMethod = Literal[
    "EAP",
    "EAP-IG-inputs",
    "clean-corrupted",
    "EAP-IG-activations",
    "exact",
]


@dataclass
class AttributionRunResult:
    graph: Graph
    scores: Tensor


@dataclass
class TaskSpec:
    task: TaskName
    labels: Any | None
    options: dict[str, Any] | None = None


@dataclass
class CircuitEdgeSummary:
    edge_name: str
    score: float
    abs_score: float
    src: str
    dst: str


@dataclass
class CircuitRunResult:
    graph: Graph
    scores: Tensor
    top_edges: list[CircuitEdgeSummary]
    backend_info: dict[str, Any]
    run_info: dict[str, Any]


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


def _is_tensor_input_mapping(samples: Any) -> bool:
    if not isinstance(samples, Mapping):
        return False
    input_ids = samples.get("input_ids")
    inputs_embeds = samples.get("inputs_embeds")
    return torch.is_tensor(input_ids) or torch.is_tensor(inputs_embeds)


def _input_lengths_from_inputs(inputs: Mapping[str, Any]) -> Tensor:
    attention_mask = inputs.get("attention_mask")
    if isinstance(attention_mask, Tensor):
        return attention_mask.sum(dim=-1)

    input_ids = inputs.get("input_ids")
    if isinstance(input_ids, Tensor):
        return torch.full(
            (int(input_ids.shape[0]),),
            int(input_ids.shape[1]),
            dtype=torch.long,
            device=input_ids.device,
        )

    inputs_embeds = inputs.get("inputs_embeds")
    if isinstance(inputs_embeds, Tensor):
        return torch.full(
            (int(inputs_embeds.shape[0]),),
            int(inputs_embeds.shape[1]),
            dtype=torch.long,
            device=inputs_embeds.device,
        )

    raise ValueError("Prepared tensor inputs must include input_ids or inputs_embeds")


def _as_tensor_inputs(clean_inputs: Mapping[str, Any], corrupt_inputs: Mapping[str, Any], labels: Any) -> PreparedBatch:
    clean_dict = dict(clean_inputs)
    corrupt_dict = dict(corrupt_inputs)
    prepared = PreparedBatch(
        clean_inputs=clean_dict,
        corrupt_inputs=corrupt_dict,
        labels=labels,
        input_lengths=_input_lengths_from_inputs(clean_dict),
    )
    validate_prepared_batch(prepared)
    return prepared


def _slice_labels(labels: Any, start: int, end: int, *, total_size: int) -> Any:
    if labels is None:
        return None
    if torch.is_tensor(labels):
        if labels.ndim > 0 and int(labels.shape[0]) == total_size:
            return labels[start:end]
        return labels
    if isinstance(labels, (list, tuple)):
        if len(labels) == total_size:
            return labels[start:end]
        return labels
    if isinstance(labels, Mapping):
        out = {}
        for key, value in labels.items():
            if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == total_size:
                out[key] = value[start:end]
            elif isinstance(value, (list, tuple)) and len(value) == total_size:
                out[key] = value[start:end]
            else:
                out[key] = value
        return out
    return labels


def _is_sliceable_sequence(value: Any, *, expected_size: int) -> bool:
    if torch.is_tensor(value):
        return value.ndim > 0 and int(value.shape[0]) == expected_size
    if isinstance(value, (list, tuple)):
        return len(value) == expected_size
    return False


def _mapping_batch_size(samples: Mapping[str, Any]) -> Optional[int]:
    for value in samples.values():
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
        if isinstance(value, (list, tuple)) and len(value) > 0:
            return len(value)
    return None


def _slice_sample_mapping(samples: Mapping[str, Any], start: int, end: int, *, total_size: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in samples.items():
        if _is_sliceable_sequence(value, expected_size=total_size):
            out[key] = value[start:end]
        else:
            out[key] = value
    return out


def _prepare_discovery_batches(
    *,
    backend_obj: ModelBackend,
    clean_samples: Any,
    corrupt_samples: Any,
    labels: Any,
    processor: Any,
    processor_kwargs: Optional[dict[str, Any]],
    batch_size: Optional[int],
) -> list[PreparedBatch]:
    if _is_tensor_input_mapping(clean_samples) and _is_tensor_input_mapping(corrupt_samples):
        return [_as_tensor_inputs(clean_samples, corrupt_samples, labels)]

    adapter = HFProcessorAdapter(
        processor=processor,
        processor_kwargs=processor_kwargs,
        device=backend_obj.config.device,
    )

    if batch_size is None:
        return [
            adapter.prepare_batch(
                clean_samples=clean_samples,
                corrupt_samples=corrupt_samples,
                labels=labels,
            )
        ]

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive when provided, got {batch_size}")

    if isinstance(clean_samples, Mapping) and isinstance(corrupt_samples, Mapping):
        clean_total = _mapping_batch_size(clean_samples)
        corrupt_total = _mapping_batch_size(corrupt_samples)
        if clean_total is None or corrupt_total is None:
            raise ValueError("Could not infer batch dimension for mapping samples")
        if clean_total != corrupt_total:
            raise ValueError(
                "clean/corrupt sample count mismatch: "
                f"clean={clean_total} corrupt={corrupt_total}"
            )
        prepared_batches: list[PreparedBatch] = []
        for start in range(0, clean_total, batch_size):
            end = min(clean_total, start + batch_size)
            prepared_batches.append(
                adapter.prepare_batch(
                    clean_samples=_slice_sample_mapping(clean_samples, start, end, total_size=clean_total),
                    corrupt_samples=_slice_sample_mapping(
                        corrupt_samples, start, end, total_size=corrupt_total
                    ),
                    labels=_slice_labels(labels, start, end, total_size=clean_total),
                )
            )
        return prepared_batches

    if not isinstance(clean_samples, Sequence) or isinstance(clean_samples, (str, bytes, Mapping)):
        raise ValueError("batch_size slicing requires clean_samples to be a sequence of examples")
    if not isinstance(corrupt_samples, Sequence) or isinstance(corrupt_samples, (str, bytes, Mapping)):
        raise ValueError("batch_size slicing requires corrupt_samples to be a sequence of examples")
    if len(clean_samples) != len(corrupt_samples):
        raise ValueError(
            "clean/corrupt sample count mismatch: "
            f"clean={len(clean_samples)} corrupt={len(corrupt_samples)}"
        )

    total = len(clean_samples)
    sequence_prepared_batches: list[PreparedBatch] = []
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        sequence_prepared_batches.append(
            adapter.prepare_batch(
                clean_samples=clean_samples[start:end],
                corrupt_samples=corrupt_samples[start:end],
                labels=_slice_labels(labels, start, end, total_size=total),
            )
        )
    return sequence_prepared_batches


def _coerce_vector(value: Any, *, batch_size: int, device: torch.device, name: str) -> Tensor:
    vec = torch.as_tensor(value, device=device, dtype=torch.long).reshape(-1)
    if vec.numel() == 1 and batch_size != 1:
        vec = vec.expand(batch_size)
    if vec.numel() != batch_size:
        raise ValueError(f"{name} size mismatch: expected {batch_size}, got {vec.numel()}")
    return vec


def _resolve_positions(batch: PreparedBatch, *, logits: Tensor, explicit_positions: Any) -> Tensor:
    if explicit_positions is not None:
        return _coerce_vector(
            explicit_positions,
            batch_size=int(logits.shape[0]),
            device=logits.device,
            name="target_positions",
        )

    clean_mask = batch.clean_inputs.get("attention_mask")
    if torch.is_tensor(clean_mask):
        positions = clean_mask.to(device=logits.device).sum(dim=-1).long() - 1
    else:
        positions = batch.input_lengths.to(device=logits.device).long() - 1
    return torch.clamp(positions, min=0)


def _build_next_token_metric(task_spec: TaskSpec) -> MetricFn:
    label_payload = task_spec.labels
    explicit_positions = None
    target_token_ids = None

    if isinstance(label_payload, Mapping):
        target_token_ids = label_payload.get("target_token_ids")
        explicit_positions = label_payload.get("target_positions")
    else:
        target_token_ids = label_payload

    if target_token_ids is None:
        raise ValueError(
            "task='next_token' requires labels. Pass token ids directly, "
            "or pass {'target_token_ids': ..., 'target_positions': ...}."
        )

    def metric(logits: Tensor, clean_logits: Optional[Tensor], batch: PreparedBatch) -> Tensor:
        batch_size = int(logits.shape[0])
        targets = _coerce_vector(
            target_token_ids,
            batch_size=batch_size,
            device=logits.device,
            name="target_token_ids",
        )
        positions = _resolve_positions(batch, logits=logits, explicit_positions=explicit_positions)
        batch_idx = torch.arange(batch_size, device=logits.device)

        chosen = logits[batch_idx, positions, targets]
        if clean_logits is not None:
            chosen = chosen - clean_logits[batch_idx, positions, targets]
        return chosen.sum()

    return metric


def _build_choice_classification_metric(task_spec: TaskSpec) -> MetricFn:
    payload = task_spec.labels
    if not isinstance(payload, Mapping):
        raise ValueError(
            "task='choice_classification' requires labels mapping with "
            "{'targets': ..., 'choice_token_ids': ...}"
        )

    targets = payload.get("targets")
    choice_token_ids = payload.get("choice_token_ids")
    explicit_positions = payload.get("target_positions")
    if targets is None or choice_token_ids is None:
        raise ValueError(
            "task='choice_classification' requires labels={'targets': ..., 'choice_token_ids': ...}"
        )

    def metric(logits: Tensor, clean_logits: Optional[Tensor], batch: PreparedBatch) -> Tensor:
        batch_size = int(logits.shape[0])
        target_idx = _coerce_vector(targets, batch_size=batch_size, device=logits.device, name="targets")
        positions = _resolve_positions(batch, logits=logits, explicit_positions=explicit_positions)

        choices = torch.as_tensor(choice_token_ids, device=logits.device, dtype=torch.long)
        if choices.ndim == 1:
            choices = choices.unsqueeze(0).expand(batch_size, -1)
        elif choices.ndim == 2 and int(choices.shape[0]) == 1 and batch_size > 1:
            choices = choices.expand(batch_size, -1)
        elif choices.ndim != 2 or int(choices.shape[0]) != batch_size:
            raise ValueError(
                "choice_token_ids must be shape [num_choices] or [batch, num_choices]. "
                f"Got shape={tuple(choices.shape)} for batch={batch_size}"
            )

        batch_idx = torch.arange(batch_size, device=logits.device).unsqueeze(1)
        pos_idx = positions.unsqueeze(1)
        selected = logits[batch_idx, pos_idx, choices]
        if clean_logits is not None:
            selected = selected - clean_logits[batch_idx, pos_idx, choices]

        return -F.cross_entropy(selected, target_idx, reduction="sum")

    return metric


def _build_task_metric(task_spec: TaskSpec) -> MetricFn:
    if task_spec.task == "next_token":
        return _build_next_token_metric(task_spec)
    if task_spec.task == "choice_classification":
        return _build_choice_classification_metric(task_spec)
    raise ValueError(f"Unsupported task '{task_spec.task}'")


def _summarize_top_edges(graph: Graph, scores: Tensor, *, top_k: int) -> list[CircuitEdgeSummary]:
    if top_k < 0:
        raise ValueError(f"top_k must be >= 0, got {top_k}")

    summaries: list[CircuitEdgeSummary] = []
    for edge in graph.edges.values():
        score = float(scores[edge.matrix_index].item())
        summaries.append(
            CircuitEdgeSummary(
                edge_name=edge.name,
                score=score,
                abs_score=abs(score),
                src=edge.parent.name,
                dst=edge.child.name,
            )
        )

    summaries.sort(key=lambda item: item.abs_score, reverse=True)
    return summaries[: min(top_k, len(summaries))]


def discover_circuit(
    *,
    model_id_or_path: str,
    clean_samples: Any,
    corrupt_samples: Any,
    task: Literal["next_token", "choice_classification"],
    labels: Any | None = None,
    method: AttributionMethod = "EAP",
    top_k: int = 200,
    device: str | torch.device = "auto",
    dtype: str | torch.dtype = "auto",
    ig_steps: int | None = None,
    batch_size: int | None = None,
    processor_kwargs: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
    cache: bool = True,
    strict_arch: bool = True,
) -> CircuitRunResult:
    """One-shot high-level entrypoint for HF model_id-based circuit discovery."""

    t0 = time.perf_counter()
    artifacts = load_hf_backend_and_processor(
        model_id_or_path=model_id_or_path,
        device=device,
        dtype=dtype,
        model_kwargs=model_kwargs,
        strict_arch=strict_arch,
        cache=cache,
    )

    task_spec = TaskSpec(task=task, labels=labels)
    metric = _build_task_metric(task_spec)

    prepared_batches = _prepare_discovery_batches(
        backend_obj=artifacts.backend,
        clean_samples=clean_samples,
        corrupt_samples=corrupt_samples,
        labels=labels,
        processor=artifacts.processor,
        processor_kwargs=processor_kwargs,
        batch_size=batch_size,
    )

    run_graph = Graph.from_model(artifacts.backend.config)
    scores = attribute(
        model=artifacts.model,
        graph=run_graph,
        batches=prepared_batches,
        metric=metric,
        backend=artifacts.backend,
        method=method,
        ig_steps=ig_steps,
        quiet=True,
    )

    elapsed = time.perf_counter() - t0
    top_edges = _summarize_top_edges(run_graph, scores, top_k=top_k)

    backend_info = {
        "adapter_name": getattr(artifacts.backend, "adapter_name", None),
        "backbone_path": getattr(artifacts.backend, "backbone_path", None),
        "arch_kind": getattr(artifacts.backend, "arch_kind", None),
        "from_cache": artifacts.from_cache,
    }
    run_info = {
        "method": method,
        "task": task,
        "model_id_or_path": artifacts.model_id_or_path,
        "device": str(artifacts.device),
        "dtype": str(artifacts.dtype),
        "elapsed_sec": elapsed,
        "n_batches": len(prepared_batches),
        "n_examples": int(sum(batch.batch_size for batch in prepared_batches)),
    }

    return CircuitRunResult(
        graph=run_graph,
        scores=scores,
        top_edges=top_edges,
        backend_info=backend_info,
        run_info=run_info,
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
    "CircuitEdgeSummary",
    "CircuitRunResult",
    "MetricFn",
    "TaskSpec",
    "attribute_from_dataloader",
    "discover_circuit",
    "evaluate_baseline_from_dataloader",
    "evaluate_graph_from_dataloader",
]
