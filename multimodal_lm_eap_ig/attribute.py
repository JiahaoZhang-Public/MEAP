from __future__ import annotations

from typing import Callable, Iterable, Literal, Optional, Tuple

import torch
from torch import Tensor
from tqdm import tqdm
from transformer_lens import HookedTransformer

from .batch import BatchLike, PreparedBatch, iter_prepared_batches
from .evaluate import evaluate_baseline, evaluate_graph
from .graph import Graph
from .utils import (
    compute_mean_activations,
    forward_with_hooks,
    make_hooks_and_matrices,
    resolve_model_run_inputs,
)

MetricFn = Callable[[Tensor, Optional[Tensor], PreparedBatch], Tensor]


def _new_scores(model: HookedTransformer, graph: Graph) -> Tensor:
    return torch.zeros(
        (graph.n_forward, graph.n_backward),
        device=model.cfg.device,
        dtype=model.cfg.dtype,
    )


def _prepare_means(
    model: HookedTransformer,
    graph: Graph,
    intervention: str,
    intervention_batches: Optional[Iterable[BatchLike]],
) -> Optional[Tensor]:
    if "mean" not in intervention:
        return None

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
    return means


def get_scores_exact(
    model: HookedTransformer,
    graph: Graph,
    batches: Iterable[BatchLike],
    metric: MetricFn,
    *,
    intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
    intervention_batches: Optional[Iterable[BatchLike]] = None,
    quiet: bool = False,
) -> Tensor:
    """Get exact leave-one-edge-out scores by repeated graph evaluation."""

    graph.in_graph |= graph.real_edge_mask
    baseline = evaluate_baseline(model, batches, metric, quiet=quiet).mean().item()

    edges = graph.edges.values() if quiet else tqdm(graph.edges.values())
    for edge in edges:
        edge.in_graph = False
        intervened = evaluate_graph(
            model,
            graph,
            batches,
            metric,
            intervention=intervention,
            intervention_batches=intervention_batches,
            quiet=True,
            skip_clean=True,
        ).mean().item()
        edge.score = intervened - baseline
        edge.in_graph = True

    return graph.scores


def get_scores_eap(
    model: HookedTransformer,
    graph: Graph,
    batches: Iterable[BatchLike],
    metric: MetricFn,
    *,
    intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
    intervention_batches: Optional[Iterable[BatchLike]] = None,
    quiet: bool = False,
) -> Tensor:
    """Get edge attribution scores using EAP."""

    scores = _new_scores(model, graph)
    means = _prepare_means(model, graph, intervention, intervention_batches)

    total_items = 0
    batch_iter = iter_prepared_batches(model, batches)
    if not quiet:
        batch_iter = tqdm(batch_iter)

    for batch in batch_iter:
        clean_inputs = resolve_model_run_inputs(model, batch.clean_inputs)
        corrupt_inputs = resolve_model_run_inputs(model, batch.corrupt_inputs)

        if clean_inputs.tokens.shape != corrupt_inputs.tokens.shape:
            raise ValueError("clean/corrupt token shapes must match")

        batch_size = batch.batch_size
        total_items += batch_size
        n_pos = int(clean_inputs.tokens.shape[1])

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(
            model,
            graph,
            batch_size,
            n_pos,
            scores,
        )

        with torch.inference_mode():
            if intervention == "patching":
                _ = forward_with_hooks(model, corrupt_inputs, fwd_hooks=fwd_hooks_corrupted)
            elif means is not None:
                if means.shape[1] not in (1, n_pos):
                    raise ValueError(
                        "mean activation length mismatch. "
                        f"Expected seq length {n_pos}, got {means.shape[1]}"
                    )
                activation_difference += means

            clean_logits = forward_with_hooks(model, clean_inputs)

        model.zero_grad(set_to_none=True)
        logits = forward_with_hooks(
            model,
            clean_inputs,
            fwd_hooks=fwd_hooks_clean,
            bwd_hooks=bwd_hooks,
        )
        metric_value = metric(logits, clean_logits, batch)
        metric_value.backward()

    if total_items == 0:
        raise ValueError("Cannot score an empty batch iterable")

    scores /= total_items
    return scores


def get_scores_eap_ig(
    model: HookedTransformer,
    graph: Graph,
    batches: Iterable[BatchLike],
    metric: MetricFn,
    *,
    steps: int = 30,
    quiet: bool = False,
) -> Tensor:
    """Get edge attribution scores using EAP integrated gradients over input stream."""

    if steps <= 0:
        raise ValueError("steps must be positive")

    scores = _new_scores(model, graph)

    total_items = 0
    total_steps = 0

    batch_iter = iter_prepared_batches(model, batches)
    if not quiet:
        batch_iter = tqdm(batch_iter)

    for batch in batch_iter:
        clean_inputs = resolve_model_run_inputs(model, batch.clean_inputs)
        corrupt_inputs = resolve_model_run_inputs(model, batch.corrupt_inputs)

        if clean_inputs.tokens.shape != corrupt_inputs.tokens.shape:
            raise ValueError("clean/corrupt token shapes must match")

        batch_size = batch.batch_size
        total_items += batch_size
        n_pos = int(clean_inputs.tokens.shape[1])

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(
            model,
            graph,
            batch_size,
            n_pos,
            scores,
        )

        with torch.inference_mode():
            _ = forward_with_hooks(model, corrupt_inputs, fwd_hooks=fwd_hooks_corrupted)

            input_idx = graph.forward_index(graph.nodes["input"])
            input_acts_corrupt = activation_difference[:, :, input_idx].clone()

            clean_logits = forward_with_hooks(model, clean_inputs, fwd_hooks=fwd_hooks_clean)
            input_acts_clean = input_acts_corrupt - activation_difference[:, :, input_idx]

        for step in range(1, steps + 1):
            total_steps += 1
            alpha = step / steps

            def input_interpolation_hook(activations, hook, interpolation_alpha: float = alpha):
                return input_acts_corrupt + interpolation_alpha * (input_acts_clean - input_acts_corrupt)

            model.zero_grad(set_to_none=True)
            logits = forward_with_hooks(
                model,
                clean_inputs,
                fwd_hooks=[(graph.nodes["input"].out_hook, input_interpolation_hook)],
                bwd_hooks=bwd_hooks,
            )
            metric_value = metric(logits, clean_logits, batch)
            metric_value.backward()

    if total_items == 0:
        raise ValueError("Cannot score an empty batch iterable")

    scores /= total_items
    scores /= total_steps
    return scores


def get_scores_ig_activations(
    model: HookedTransformer,
    graph: Graph,
    batches: Iterable[BatchLike],
    metric: MetricFn,
    *,
    intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
    steps: int = 30,
    intervention_batches: Optional[Iterable[BatchLike]] = None,
    quiet: bool = False,
) -> Tensor:
    """Get edge attribution scores using IG over node activations."""

    if steps <= 0:
        raise ValueError("steps must be positive")

    means = _prepare_means(model, graph, intervention, intervention_batches)
    scores = _new_scores(model, graph)

    total_items = 0
    total_steps = 0

    batch_iter = iter_prepared_batches(model, batches)
    if not quiet:
        batch_iter = tqdm(batch_iter)

    for batch in batch_iter:
        clean_inputs = resolve_model_run_inputs(model, batch.clean_inputs)
        corrupt_inputs = resolve_model_run_inputs(model, batch.corrupt_inputs)

        if clean_inputs.tokens.shape != corrupt_inputs.tokens.shape:
            raise ValueError("clean/corrupt token shapes must match")

        batch_size = batch.batch_size
        total_items += batch_size
        n_pos = int(clean_inputs.tokens.shape[1])

        (_, _, bwd_hooks), activation_difference = make_hooks_and_matrices(
            model,
            graph,
            batch_size,
            n_pos,
            scores,
        )
        (fwd_hooks_corrupted, _, _), activations_corrupted = make_hooks_and_matrices(
            model,
            graph,
            batch_size,
            n_pos,
            scores,
        )
        (fwd_hooks_clean, _, _), activations_clean = make_hooks_and_matrices(
            model,
            graph,
            batch_size,
            n_pos,
            scores,
        )

        if intervention == "patching":
            with torch.inference_mode():
                _ = forward_with_hooks(model, corrupt_inputs, fwd_hooks=fwd_hooks_corrupted)
        elif means is not None:
            if means.shape[1] not in (1, n_pos):
                raise ValueError(
                    "mean activation length mismatch. "
                    f"Expected seq length {n_pos}, got {means.shape[1]}"
                )
            activation_difference += means

        with torch.inference_mode():
            clean_logits = forward_with_hooks(model, clean_inputs, fwd_hooks=fwd_hooks_clean)
            activation_difference += activations_corrupted.detach() - activations_clean.detach()

        def output_interpolation_hook(
            clean_acts: Tensor,
            corrupt_acts: Tensor,
            alpha: float,
        ):
            def hook_fn(activations: Tensor, hook):
                return alpha * clean_acts + (1 - alpha) * corrupt_acts

            return hook_fn

        nodes_list = [graph.nodes["input"]]
        for layer in range(graph.cfg["n_layers"]):
            nodes_list.append(graph.nodes[f"a{layer}.h0"])
            nodes_list.append(graph.nodes[f"m{layer}"])

        for node in nodes_list:
            clean_acts = activations_clean[:, :, graph.forward_index(node)]
            corrupt_acts = activations_corrupted[:, :, graph.forward_index(node)]
            for step in range(1, steps + 1):
                total_steps += 1
                alpha = step / steps
                model.zero_grad(set_to_none=True)
                logits = forward_with_hooks(
                    model,
                    clean_inputs,
                    fwd_hooks=[
                        (
                            node.out_hook,
                            output_interpolation_hook(clean_acts, corrupt_acts, alpha),
                        )
                    ],
                    bwd_hooks=bwd_hooks,
                )
                metric_value = metric(logits, clean_logits, batch)
                metric_value.backward(retain_graph=True)

    if total_items == 0:
        raise ValueError("Cannot score an empty batch iterable")

    scores /= total_items
    scores /= total_steps
    return scores


def get_scores_clean_corrupted(
    model: HookedTransformer,
    graph: Graph,
    batches: Iterable[BatchLike],
    metric: MetricFn,
    *,
    quiet: bool = False,
) -> Tensor:
    """Two-point clean/corrupted approximation of IG."""

    scores = _new_scores(model, graph)

    total_items = 0
    total_steps = 2

    batch_iter = iter_prepared_batches(model, batches)
    if not quiet:
        batch_iter = tqdm(batch_iter)

    for batch in batch_iter:
        clean_inputs = resolve_model_run_inputs(model, batch.clean_inputs)
        corrupt_inputs = resolve_model_run_inputs(model, batch.corrupt_inputs)

        if clean_inputs.tokens.shape != corrupt_inputs.tokens.shape:
            raise ValueError("clean/corrupt token shapes must match")

        batch_size = batch.batch_size
        total_items += batch_size
        n_pos = int(clean_inputs.tokens.shape[1])

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), _ = make_hooks_and_matrices(
            model,
            graph,
            batch_size,
            n_pos,
            scores,
        )

        with torch.inference_mode():
            _ = forward_with_hooks(model, corrupt_inputs, fwd_hooks=fwd_hooks_corrupted)
            clean_logits = forward_with_hooks(model, clean_inputs, fwd_hooks=fwd_hooks_clean)

        model.zero_grad(set_to_none=True)
        logits = forward_with_hooks(model, clean_inputs, bwd_hooks=bwd_hooks)
        metric(logits, clean_logits, batch).backward()

        model.zero_grad(set_to_none=True)
        corrupted_logits = forward_with_hooks(model, corrupt_inputs, bwd_hooks=bwd_hooks)
        metric(corrupted_logits, clean_logits, batch).backward()

    if total_items == 0:
        raise ValueError("Cannot score an empty batch iterable")

    scores /= total_items
    scores /= total_steps
    return scores


def get_real_edge_scores(
    graph: Graph,
    scores: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Return real-edge indices and flattened scores for convenience."""

    if scores is None:
        scores = graph.scores
    indices = graph.real_edge_mask.nonzero(as_tuple=False)
    values = scores[graph.real_edge_mask]
    return indices, values


allowed_aggregations = {"sum", "mean"}


def attribute(
    model: HookedTransformer,
    graph: Graph,
    batches: Iterable[BatchLike],
    metric: MetricFn,
    *,
    method: Literal["EAP", "EAP-IG-inputs", "clean-corrupted", "EAP-IG-activations", "exact"],
    intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
    aggregation: Literal["sum", "mean"] = "sum",
    ig_steps: Optional[int] = None,
    intervention_batches: Optional[Iterable[BatchLike]] = None,
    quiet: bool = False,
) -> Tensor:
    """Unified attribution entrypoint for multimodal prepared batches."""

    assert model.cfg.use_attn_result, "Model must enable model.cfg.use_attn_result"
    assert model.cfg.use_split_qkv_input, "Model must enable model.cfg.use_split_qkv_input"
    assert model.cfg.use_hook_mlp_in, "Model must enable model.cfg.use_hook_mlp_in"
    if model.cfg.n_key_value_heads is not None:
        assert model.cfg.ungroup_grouped_query_attention, (
            "Model must enable model.cfg.ungroup_grouped_query_attention"
        )

    if aggregation not in allowed_aggregations:
        raise ValueError(f"aggregation must be in {allowed_aggregations}, but got {aggregation}")

    steps = ig_steps if ig_steps is not None else 30

    if method == "EAP":
        scores = get_scores_eap(
            model,
            graph,
            batches,
            metric,
            intervention=intervention,
            intervention_batches=intervention_batches,
            quiet=quiet,
        )
    elif method == "EAP-IG-inputs":
        if intervention != "patching":
            raise ValueError("intervention must be 'patching' for EAP-IG-inputs")
        scores = get_scores_eap_ig(model, graph, batches, metric, steps=steps, quiet=quiet)
    elif method == "clean-corrupted":
        if intervention != "patching":
            raise ValueError("intervention must be 'patching' for clean-corrupted")
        scores = get_scores_clean_corrupted(model, graph, batches, metric, quiet=quiet)
    elif method == "EAP-IG-activations":
        scores = get_scores_ig_activations(
            model,
            graph,
            batches,
            metric,
            intervention=intervention,
            steps=steps,
            intervention_batches=intervention_batches,
            quiet=quiet,
        )
    elif method == "exact":
        scores = get_scores_exact(
            model,
            graph,
            batches,
            metric,
            intervention=intervention,
            intervention_batches=intervention_batches,
            quiet=quiet,
        )
    else:
        raise ValueError(
            "method must be one of ['EAP', 'EAP-IG-inputs', 'clean-corrupted', "
            "'EAP-IG-activations', 'exact']"
        )

    if aggregation == "mean":
        scores = scores / model.cfg.d_model

    graph.scores[:] = scores.to(graph.scores.device)
    return graph.scores
