from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Optional

import torch
from torch import Tensor
import torch.nn.functional as F

from .batch import PreparedBatch

MetricFn = Callable[[Tensor, Optional[Tensor], PreparedBatch], Tensor]
TaskName = Literal["next_token", "choice_classification"]


@dataclass
class TaskSpec:
    task: TaskName
    labels: Any | None
    options: dict[str, Any] | None = None


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


def build_task_metric(task_spec: TaskSpec) -> MetricFn:
    if task_spec.task == "next_token":
        return _build_next_token_metric(task_spec)
    if task_spec.task == "choice_classification":
        return _build_choice_classification_metric(task_spec)
    raise ValueError(f"Unsupported task '{task_spec.task}'")


__all__ = [
    "MetricFn",
    "TaskName",
    "TaskSpec",
    "build_task_metric",
]
