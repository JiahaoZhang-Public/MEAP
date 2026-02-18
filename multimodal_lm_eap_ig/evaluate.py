from __future__ import annotations

from typing import Callable, Iterable, List, Literal, Optional, Union

from einops import einsum
import torch
from torch import Tensor
from tqdm import tqdm
from transformer_lens import HookedTransformer

from .batch import BatchLike, PreparedBatch, iter_prepared_batches
from .graph import AttentionNode, Graph
from .utils import (
    compute_mean_activations,
    forward_with_hooks,
    make_hooks_and_matrices,
    resolve_model_run_inputs,
)

MetricFn = Callable[[Tensor, Optional[Tensor], PreparedBatch], Tensor]


def evaluate_graph(
    model: HookedTransformer,
    graph: Graph,
    batches: Iterable[BatchLike],
    metrics: Union[MetricFn, List[MetricFn]],
    *,
    quiet: bool = False,
    intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
    intervention_batches: Optional[Iterable[BatchLike]] = None,
    skip_clean: bool = True,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Evaluate a pruned circuit under patching/ablation interventions."""

    assert model.cfg.use_attn_result, "Model must enable model.cfg.use_attn_result"
    if model.cfg.n_key_value_heads is not None:
        assert model.cfg.ungroup_grouped_query_attention, (
            "Model must enable model.cfg.ungroup_grouped_query_attention"
        )

    assert intervention in ["patching", "zero", "mean", "mean-positional"]

    means = None
    if "mean" in intervention:
        if intervention_batches is None:
            raise ValueError("intervention_batches must be provided for mean interventions")
        per_position = "positional" in intervention
        means = compute_mean_activations(
            model,
            graph,
            intervention_batches,
            per_position=per_position,
        )
        means = means.unsqueeze(0)
        if not per_position:
            means = means.unsqueeze(0)

    graph.prune()

    in_graph_matrix = graph.in_graph.to(device=model.cfg.device, dtype=model.cfg.dtype)

    if graph.neurons_in_graph is not None:
        neuron_matrix = graph.neurons_in_graph.to(device=model.cfg.device, dtype=model.cfg.dtype)
        node_fully_in_graph = (neuron_matrix.sum(-1) == model.cfg.d_model).to(model.cfg.dtype)
        in_graph_matrix = einsum(
            in_graph_matrix,
            node_fully_in_graph,
            "forward backward, forward -> forward backward",
        )
    else:
        neuron_matrix = None

    in_graph_matrix = 1 - in_graph_matrix
    if neuron_matrix is not None:
        neuron_matrix = 1 - neuron_matrix

    if model.cfg.use_normalization_before_and_after:
        attention_head_mask = torch.zeros(
            (graph.n_forward, model.cfg.n_layers),
            device=model.cfg.device,
            dtype=model.cfg.dtype,
        )
        for node in graph.nodes.values():
            if isinstance(node, AttentionNode):
                attention_head_mask[graph.forward_index(node), node.layer] = 1

        non_attention_head_mask = 1 - attention_head_mask.any(-1).to(dtype=model.cfg.dtype)
        attention_biases = torch.stack([block.attn.b_O for block in model.blocks])

    def make_input_construction_hook(activation_matrix, in_graph_vector, neuron_mask):
        def input_construction_hook(activations, hook):
            if model.cfg.use_normalization_before_and_after:
                activation_differences = activation_matrix[0] - activation_matrix[1]

                clean_attention_results = einsum(
                    activation_matrix[1, :, :, : len(in_graph_vector)],
                    attention_head_mask[: len(in_graph_vector)],
                    "batch pos previous hidden, previous layer -> batch pos layer hidden",
                )

                if neuron_mask is not None:
                    non_attention_update = einsum(
                        activation_differences[:, :, : len(in_graph_vector)],
                        neuron_mask[: len(in_graph_vector)],
                        in_graph_vector,
                        non_attention_head_mask[: len(in_graph_vector)],
                        (
                            "batch pos previous hidden, previous hidden, previous ..., previous -> "
                            "batch pos ... hidden"
                        ),
                    )
                    corrupted_attention_difference = einsum(
                        activation_differences[:, :, : len(in_graph_vector)],
                        neuron_mask[: len(in_graph_vector)],
                        in_graph_vector,
                        attention_head_mask[: len(in_graph_vector)],
                        (
                            "batch pos previous hidden, previous hidden, previous ..., previous layer -> "
                            "batch pos ... layer hidden"
                        ),
                    )
                else:
                    non_attention_update = einsum(
                        activation_differences[:, :, : len(in_graph_vector)],
                        in_graph_vector,
                        non_attention_head_mask[: len(in_graph_vector)],
                        "batch pos previous hidden, previous ..., previous -> batch pos ... hidden",
                    )
                    corrupted_attention_difference = einsum(
                        activation_differences[:, :, : len(in_graph_vector)],
                        in_graph_vector,
                        attention_head_mask[: len(in_graph_vector)],
                        (
                            "batch pos previous hidden, previous ..., previous layer -> "
                            "batch pos ... layer hidden"
                        ),
                    )

                if in_graph_vector.ndim == 2:
                    corrupted_attention_results = clean_attention_results.unsqueeze(2)
                    corrupted_attention_results += corrupted_attention_difference
                    clean_attention_results += attention_biases.unsqueeze(0).unsqueeze(0)
                    corrupted_attention_results += attention_biases.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                else:
                    corrupted_attention_results = clean_attention_results + corrupted_attention_difference
                    clean_attention_results += attention_biases.unsqueeze(0).unsqueeze(0)
                    corrupted_attention_results += attention_biases.unsqueeze(0).unsqueeze(0)

                update = non_attention_update
                valid_layers = attention_head_mask[: len(in_graph_vector)].any(0)
                for i, valid_layer in enumerate(valid_layers):
                    if not valid_layer:
                        break
                    if in_graph_vector.ndim == 2:
                        update -= model.blocks[i].ln1_post(clean_attention_results[:, :, None, i])
                        update += model.blocks[i].ln1_post(corrupted_attention_results[:, :, :, i])
                    else:
                        update -= model.blocks[i].ln1_post(clean_attention_results[:, :, i])
                        update += model.blocks[i].ln1_post(corrupted_attention_results[:, :, i])
            else:
                activation_differences = activation_matrix
                if neuron_mask is not None:
                    update = einsum(
                        activation_differences[:, :, : len(in_graph_vector)],
                        neuron_mask[: len(in_graph_vector)],
                        in_graph_vector,
                        "batch pos previous hidden, previous hidden, previous ... -> batch pos ... hidden",
                    )
                else:
                    update = einsum(
                        activation_differences[:, :, : len(in_graph_vector)],
                        in_graph_vector,
                        "batch pos previous hidden, previous ... -> batch pos ... hidden",
                    )

            activations += update
            return activations

        return input_construction_hook

    def make_input_construction_hooks(activation_differences, in_graph_values, neuron_mask):
        hooks = []
        for layer in range(model.cfg.n_layers):
            if any(
                graph.nodes[f"a{layer}.h{head}"].in_graph for head in range(model.cfg.n_heads)
            ) and not (
                neuron_mask is None
                and all(
                    parent_edge.in_graph
                    for head in range(model.cfg.n_heads)
                    for parent_edge in graph.nodes[f"a{layer}.h{head}"].parent_edges
                )
            ):
                for i, letter in enumerate("qkv"):
                    node = graph.nodes[f"a{layer}.h0"]
                    prev_index = graph.prev_index(node)
                    bwd_index = graph.backward_index(node, qkv=letter, attn_slice=True)
                    input_hook = make_input_construction_hook(
                        activation_differences,
                        in_graph_values[:prev_index, bwd_index],
                        neuron_mask,
                    )
                    hooks.append((node.qkv_inputs[i], input_hook))

            if graph.nodes[f"m{layer}"].in_graph and not (
                neuron_mask is None
                and all(
                    parent_edge.in_graph for parent_edge in graph.nodes[f"m{layer}"].parent_edges
                )
            ):
                node = graph.nodes[f"m{layer}"]
                prev_index = graph.prev_index(node)
                bwd_index = graph.backward_index(node)
                input_hook = make_input_construction_hook(
                    activation_differences,
                    in_graph_values[:prev_index, bwd_index],
                    neuron_mask,
                )
                hooks.append((node.in_hook, input_hook))

        if not (
            neuron_mask is None
            and all(parent_edge.in_graph for parent_edge in graph.nodes["logits"].parent_edges)
        ):
            node = graph.nodes["logits"]
            fwd_index = graph.prev_index(node)
            bwd_index = graph.backward_index(node)
            input_hook = make_input_construction_hook(
                activation_differences,
                in_graph_values[:fwd_index, bwd_index],
                neuron_mask,
            )
            hooks.append((node.in_hook, input_hook))

        return hooks

    metrics_list = metrics if isinstance(metrics, list) else [metrics]
    results: List[List[Tensor]] = [[] for _ in metrics_list]

    batch_iter = iter_prepared_batches(model, batches)
    if not quiet:
        batch_iter = tqdm(batch_iter)

    for batch in batch_iter:
        clean_inputs = resolve_model_run_inputs(model, batch.clean_inputs)
        corrupt_inputs = resolve_model_run_inputs(model, batch.corrupt_inputs)

        if clean_inputs.tokens.shape != corrupt_inputs.tokens.shape:
            raise ValueError("clean/corrupt token shapes must match for evaluation")

        batch_size = batch.batch_size
        n_pos = int(clean_inputs.tokens.shape[1])

        (fwd_hooks_corrupted, fwd_hooks_clean, _), activation_difference = make_hooks_and_matrices(
            model,
            graph,
            batch_size,
            n_pos,
            None,
        )

        if means is not None and means.shape[1] not in (1, n_pos):
            raise ValueError(
                "mean activation length mismatch. "
                f"Expected seq length {n_pos}, got {means.shape[1]}"
            )

        input_construction_hooks = make_input_construction_hooks(
            activation_difference,
            in_graph_matrix,
            neuron_matrix,
        )

        with torch.inference_mode():
            if intervention == "patching":
                _ = forward_with_hooks(model, corrupt_inputs, fwd_hooks=fwd_hooks_corrupted)
            elif means is not None:
                activation_difference += means

            clean_logits = None if skip_clean else forward_with_hooks(model, clean_inputs)
            logits = forward_with_hooks(
                model,
                clean_inputs,
                fwd_hooks=fwd_hooks_clean + input_construction_hooks,
            )

        for i, metric in enumerate(metrics_list):
            r = metric(logits, clean_logits, batch).detach().cpu()
            if r.ndim == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    concatenated = [torch.cat(rs) for rs in results]
    if len(concatenated) == 1:
        return concatenated[0]
    return concatenated


def evaluate_baseline(
    model: HookedTransformer,
    batches: Iterable[BatchLike],
    metrics: Union[MetricFn, List[MetricFn]],
    *,
    run_corrupted: bool = False,
    quiet: bool = False,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Evaluate baseline model behavior without graph interventions."""

    metrics_list = metrics if isinstance(metrics, list) else [metrics]
    results: List[List[Tensor]] = [[] for _ in metrics_list]

    batch_iter = iter_prepared_batches(model, batches)
    if not quiet:
        batch_iter = tqdm(batch_iter)

    for batch in batch_iter:
        clean_inputs = resolve_model_run_inputs(model, batch.clean_inputs)
        corrupt_inputs = resolve_model_run_inputs(model, batch.corrupt_inputs)

        with torch.inference_mode():
            corrupted_logits = forward_with_hooks(model, corrupt_inputs)
            logits = forward_with_hooks(model, clean_inputs)

        for i, metric in enumerate(metrics_list):
            if run_corrupted:
                r = metric(corrupted_logits, logits, batch).detach().cpu()
            else:
                r = metric(logits, corrupted_logits, batch).detach().cpu()
            if r.ndim == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    concatenated = [torch.cat(rs) for rs in results]
    if len(concatenated) == 1:
        return concatenated[0]
    return concatenated
