from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

from einops import einsum
import torch
from torch import Tensor
from transformer_lens import HookedTransformer

from .batch import BatchLike, iter_prepared_batches
from .graph import AttentionNode, Graph, LogitNode

HookSpec = Tuple[str, Callable]


@dataclass
class ModelRunInputs:
    tokens: Tensor
    attention_mask: Optional[Tensor]
    extra_fwd_hooks: List[HookSpec]


def _get_model_device(model: HookedTransformer) -> torch.device:
    model_device = getattr(model.cfg, "device", "cpu")
    return torch.device(model_device)


def _get_model_dtype(model: HookedTransformer) -> torch.dtype:
    model_dtype = getattr(model.cfg, "dtype", torch.float32)
    if isinstance(model_dtype, torch.dtype):
        return model_dtype
    if isinstance(model_dtype, str) and hasattr(torch, model_dtype):
        return getattr(torch, model_dtype)
    return torch.float32


def resolve_model_run_inputs(
    model: HookedTransformer,
    inputs: Dict[str, Tensor],
) -> ModelRunInputs:
    device = _get_model_device(model)
    dtype = _get_model_dtype(model)

    tokens = inputs.get("input_ids")
    if tokens is None:
        tokens = inputs.get("tokens")

    input_embeds = inputs.get("inputs_embeds")

    if tokens is None and input_embeds is None:
        raise ValueError("Model inputs must include input_ids/tokens or inputs_embeds")

    if input_embeds is not None:
        input_embeds = input_embeds.to(device=device, dtype=dtype)

    if tokens is None:
        if input_embeds is None or input_embeds.ndim != 3:
            raise ValueError("inputs_embeds must be rank-3 [batch, seq, d_model]")
        pad_token_id = getattr(model.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(model.tokenizer, "eos_token_id", 0)
        tokens = torch.full(
            input_embeds.shape[:2],
            int(pad_token_id),
            device=device,
            dtype=torch.long,
        )
    else:
        tokens = tokens.to(device=device, dtype=torch.long)

    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=device)
    else:
        attention_mask = torch.ones(tokens.shape, device=device, dtype=torch.long)

    if attention_mask.shape != tokens.shape:
        raise ValueError(
            "attention_mask shape must match tokens shape. "
            f"Got attention_mask={tuple(attention_mask.shape)} tokens={tuple(tokens.shape)}"
        )

    extra_fwd_hooks: List[HookSpec] = []
    if input_embeds is not None:

        def _embed_override_hook(activations: Tensor, hook, prepared_embeds: Tensor = input_embeds) -> Tensor:
            return prepared_embeds

        extra_fwd_hooks.append(("hook_embed", _embed_override_hook))

    return ModelRunInputs(
        tokens=tokens,
        attention_mask=attention_mask,
        extra_fwd_hooks=extra_fwd_hooks,
    )


def forward_with_hooks(
    model: HookedTransformer,
    model_inputs: ModelRunInputs,
    *,
    fwd_hooks: Optional[List[HookSpec]] = None,
    bwd_hooks: Optional[List[HookSpec]] = None,
) -> Tensor:
    merged_fwd_hooks = list(model_inputs.extra_fwd_hooks)
    if fwd_hooks:
        merged_fwd_hooks.extend(fwd_hooks)

    kwargs = {"attention_mask": model_inputs.attention_mask}

    if not merged_fwd_hooks and not bwd_hooks:
        return model(model_inputs.tokens, **kwargs)

    hook_kwargs = {"fwd_hooks": merged_fwd_hooks}
    if bwd_hooks is not None:
        hook_kwargs["bwd_hooks"] = bwd_hooks

    with model.hooks(**hook_kwargs):
        return model(model_inputs.tokens, **kwargs)


def make_hooks_and_matrices(
    model: HookedTransformer,
    graph: Graph,
    batch_size: int,
    n_pos: int,
    scores: Optional[Tensor],
):
    """Create activation-difference buffers and hooks used by EAP/EAP-IG."""
    separate_activations = getattr(model.cfg, "use_normalization_before_and_after", False) and scores is None
    dtype = _get_model_dtype(model)
    device = _get_model_device(model)

    if separate_activations:
        activation_difference = torch.zeros(
            (2, batch_size, n_pos, graph.n_forward, model.cfg.d_model),
            device=device,
            dtype=dtype,
        )
    else:
        activation_difference = torch.zeros(
            (batch_size, n_pos, graph.n_forward, model.cfg.d_model),
            device=device,
            dtype=dtype,
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
    model: HookedTransformer,
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

    for batch in iter_prepared_batches(model, batches):
        total_items += batch.batch_size
        model_inputs = resolve_model_run_inputs(model, batch.clean_inputs)
        n_pos = int(model_inputs.tokens.shape[1])

        if per_position:
            if first_n_pos is None:
                first_n_pos = n_pos
            elif n_pos != first_n_pos:
                raise ValueError(
                    "Per-position mean requires constant sequence length across batches; "
                    f"found {first_n_pos} and {n_pos}"
                )

        if not means_initialized:
            dtype = _get_model_dtype(model)
            device = _get_model_device(model)
            if per_position:
                means = torch.zeros((n_pos, graph.n_forward, model.cfg.d_model), device=device, dtype=dtype)
            else:
                means = torch.zeros((graph.n_forward, model.cfg.d_model), device=device, dtype=dtype)
            means_initialized = True

        input_lengths = batch.input_lengths.to(device=model_inputs.tokens.device)

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
            _ = forward_with_hooks(model, model_inputs, fwd_hooks=add_to_mean_hooks)

    if not means_initialized:
        raise ValueError("Cannot compute means on an empty batch iterable")

    means = means / total_items
    return means
