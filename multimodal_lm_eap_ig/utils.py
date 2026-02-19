from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Iterable, List, Optional, Union

from einops import einsum
import torch
from torch import Tensor

from .backend import BackendHookSpec, BackendRunInputs, ModelBackend
from .batch import BatchLike, iter_prepared_batches
from .graph import AttentionNode, Graph, LogitNode

HookSpec = BackendHookSpec


@dataclass
class ModelRunInputs:
    run_inputs: BackendRunInputs


def resolve_run_inputs(
    backend: ModelBackend,
    inputs: dict[str, Tensor],
) -> ModelRunInputs:
    return ModelRunInputs(run_inputs=backend.prepare_inputs(inputs))


def forward_with_hooks(
    backend: ModelBackend,
    model_inputs: ModelRunInputs,
    *,
    fwd_hooks: Optional[List[HookSpec]] = None,
    bwd_hooks: Optional[List[HookSpec]] = None,
) -> Tensor:
    return backend.forward(
        model_inputs.run_inputs,
        fwd_hooks=fwd_hooks,
        bwd_hooks=bwd_hooks,
    )


def make_hooks_and_matrices(
    backend: ModelBackend,
    graph: Graph,
    batch_size: int,
    n_pos: int,
    scores: Optional[Tensor],
):
    """Create activation-difference buffers and hooks used by EAP/EAP-IG."""
    cfg = backend.config
    separate_activations = cfg.use_normalization_before_and_after and scores is None

    if separate_activations:
        activation_difference = torch.zeros(
            (2, batch_size, n_pos, graph.n_forward, cfg.d_model),
            device=cfg.device,
            dtype=cfg.dtype,
        )
    else:
        activation_difference = torch.zeros(
            (batch_size, n_pos, graph.n_forward, cfg.d_model),
            device=cfg.device,
            dtype=cfg.dtype,
        )

    fwd_hooks_clean: List[HookSpec] = []
    fwd_hooks_corrupted: List[HookSpec] = []
    bwd_hooks: List[HookSpec] = []

    def activation_hook(index, activations: Tensor, hook, add: bool = True):
        acts = activations.detach()
        if separate_activations:
            activation_difference[0 if add else 1, :, :, index] += acts
        else:
            if add:
                activation_difference[:, :, index] += acts
            else:
                activation_difference[:, :, index] -= acts

    def gradient_hook(prev_index: int, bwd_index: Union[slice, int], gradients: Tensor, hook):
        if scores is None:
            return
        grads = gradients.detach()
        if grads.ndim == 3:
            grads = grads.unsqueeze(2)
        score_update = einsum(
            activation_difference[:, :, :prev_index],
            grads,
            "batch pos forward hidden, batch pos backward hidden -> forward backward",
        )
        score_update = score_update.squeeze(1)
        scores[:prev_index, bwd_index] += score_update

    node = graph.nodes["input"]
    fwd_index = graph.forward_index(node)
    fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
    fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))

    for layer in range(graph.cfg["n_layers"]):
        node = graph.nodes[f"a{layer}.h0"]
        fwd_index = graph.forward_index(node)
        fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
        fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
        prev_index = graph.prev_index(node)
        for i, letter in enumerate("qkv"):
            bwd_index = graph.backward_index(node, qkv=letter)
            bwd_hooks.append((node.qkv_inputs[i], partial(gradient_hook, prev_index, bwd_index)))

        node = graph.nodes[f"m{layer}"]
        fwd_index = graph.forward_index(node)
        bwd_index = graph.backward_index(node)
        prev_index = graph.prev_index(node)
        fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
        fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
        bwd_hooks.append((node.in_hook, partial(gradient_hook, prev_index, bwd_index)))

    node = graph.nodes["logits"]
    prev_index = graph.prev_index(node)
    bwd_index = graph.backward_index(node)
    bwd_hooks.append((node.in_hook, partial(gradient_hook, prev_index, bwd_index)))

    return (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference


def _mean_over_non_padding(activations: Tensor, input_lengths: Tensor) -> Tensor:
    max_len = activations.shape[1]
    pos_mask = (
        torch.arange(max_len, device=activations.device).unsqueeze(0)
        < input_lengths.to(device=activations.device).unsqueeze(1)
    )
    while pos_mask.ndim < activations.ndim:
        pos_mask = pos_mask.unsqueeze(-1)

    weighted = activations * pos_mask
    summed = weighted.sum(dim=1)
    denom = input_lengths.clamp_min(1).to(device=activations.device).view(-1, *([1] * (summed.ndim - 1)))
    return summed / denom


def compute_mean_activations(
    backend: ModelBackend,
    graph: Graph,
    batches: Iterable[BatchLike],
    *,
    per_position: bool = False,
):
    """Compute mean source-node activations across a prepared batch stream."""

    processed_attn_layers = set()
    hook_points_indices = []
    for node in graph.nodes.values():
        if isinstance(node, AttentionNode):
            if node.layer in processed_attn_layers:
                continue
            processed_attn_layers.add(node.layer)

        if not isinstance(node, LogitNode):
            hook_points_indices.append((node.out_hook, graph.forward_index(node)))

    means_initialized = False
    total_items = 0
    first_n_pos: Optional[int] = None

    for batch in iter_prepared_batches(backend.tokenization_model, batches):
        total_items += batch.batch_size
        model_inputs = resolve_run_inputs(backend, batch.clean_inputs)
        run_inputs = model_inputs.run_inputs

        if run_inputs.input_ids is not None:
            n_pos = int(run_inputs.input_ids.shape[1])
            input_device = run_inputs.input_ids.device
        elif run_inputs.inputs_embeds is not None:
            n_pos = int(run_inputs.inputs_embeds.shape[1])
            input_device = run_inputs.inputs_embeds.device
        else:
            raise RuntimeError("BackendRunInputs has neither input_ids nor inputs_embeds")

        if per_position:
            if first_n_pos is None:
                first_n_pos = n_pos
            elif n_pos != first_n_pos:
                raise ValueError(
                    "Per-position mean requires constant sequence length across batches; "
                    f"found {first_n_pos} and {n_pos}"
                )

        if not means_initialized:
            if per_position:
                means = torch.zeros(
                    (n_pos, graph.n_forward, backend.config.d_model),
                    device=backend.config.device,
                    dtype=backend.config.dtype,
                )
            else:
                means = torch.zeros(
                    (graph.n_forward, backend.config.d_model),
                    device=backend.config.device,
                    dtype=backend.config.dtype,
                )
            means_initialized = True

        input_lengths = batch.input_lengths.to(device=input_device)

        def activation_hook(index, activations: Tensor, hook, means_tensor: Tensor, lengths: Tensor):
            acts = activations.detach()
            if per_position:
                means_tensor[:, index] += acts.sum(0)
            else:
                item_means = _mean_over_non_padding(acts, lengths)
                means_tensor[index] += item_means.sum(0)

        add_to_mean_hooks = [
            (hook_point, partial(activation_hook, index, means_tensor=means, lengths=input_lengths))
            for hook_point, index in hook_points_indices
        ]

        with torch.inference_mode():
            _ = forward_with_hooks(backend, model_inputs, fwd_hooks=add_to_mean_hooks)

    if not means_initialized:
        raise ValueError("Cannot compute means on an empty batch iterable")

    means = means / total_items
    return means
