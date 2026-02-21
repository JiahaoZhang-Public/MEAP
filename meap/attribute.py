"""Internal attribution method implementations.

This module is intentionally low-level and may change between minor versions.
Prefer `meap.api` and package-level public APIs for stable usage.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Literal, Optional, Tuple

import torch
from torch import Tensor
from tqdm import tqdm

from .backend import ModelBackend, resolve_backend
from .batch import BatchLike, PreparedBatch, iter_prepared_batches
from .evaluate import evaluate_baseline, evaluate_graph
from .graph import Graph
from .utils import (
    compute_mean_activations,
    forward_with_hooks,
    make_hooks_and_matrices,
    resolve_run_inputs,
)

MetricFn = Callable[[Tensor, Optional[Tensor], PreparedBatch], Tensor]


def _new_scores(backend: ModelBackend, graph: Graph) -> Tensor:
    return torch.zeros(
        (graph.n_forward, graph.n_backward),
        device=backend.config.device,
        dtype=backend.config.dtype,
    )


def _resolve_backend_for_method(
    model: Any,
    backend: Optional[ModelBackend],
    method: str,
) -> ModelBackend:
    del method
    return resolve_backend(model, backend)


def _prepare_means(
    backend: ModelBackend,
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
        backend,
        graph,
        intervention_batches,
        per_position=per_position,
    )
    means = means.unsqueeze(0)
    if not per_position:
        means = means.unsqueeze(0)
    return means


def get_scores_exact(
    model: Any,
    graph: Graph,
    batches: Iterable[BatchLike],
    metric: MetricFn,
    *,
    backend: Optional[ModelBackend] = None,
    intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
    intervention_batches: Optional[Iterable[BatchLike]] = None,
    quiet: bool = False,
) -> Tensor:
    """Get exact leave-one-edge-out scores by repeated graph evaluation."""

    backend_obj = _resolve_backend_for_method(model, backend, method="exact")
    prepared_batches = list(iter_prepared_batches(backend_obj.tokenization_model, batches))
    if len(prepared_batches) == 0:
        raise ValueError("Cannot score an empty batch iterable")
    prepared_intervention_batches = None
    if intervention_batches is not None:
        prepared_intervention_batches = list(
            iter_prepared_batches(backend_obj.tokenization_model, intervention_batches)
        )

    graph_for_eval = Graph.from_model(
        graph.cfg,
        neuron_level=graph.neurons_in_graph is not None,
        node_scores=graph.nodes_scores is not None,
    )

    baseline = evaluate_baseline(
        model,
        prepared_batches,
        metric,
        backend=backend_obj,
        quiet=quiet,
    ).mean().item()

    edges = graph.edges.values() if quiet else tqdm(graph.edges.values())
    for edge in edges:
        graph_for_eval.reset(empty=False)
        graph_for_eval.edges[edge.name].in_graph = False
        intervened = evaluate_graph(
            model,
            graph_for_eval,
            prepared_batches,
            metric,
            backend=backend_obj,
            intervention=intervention,
            intervention_batches=prepared_intervention_batches,
            quiet=True,
            skip_clean=True,
        ).mean().item()
        edge.score = intervened - baseline

    return graph.scores


def get_scores_smoke(
    model: Any,
    graph: Graph,
    batches: Iterable[BatchLike],
    metric: MetricFn,
    *,
    backend: Optional[ModelBackend] = None,
    quiet: bool = False,
) -> Tensor:
    """Run a forward-only multimodal sanity pass and return zero scores."""
    del metric

    backend_obj = _resolve_backend_for_method(model, backend, method="smoke")

    scores = _new_scores(backend_obj, graph)
    total_items = 0

    batch_iter = iter_prepared_batches(backend_obj.tokenization_model, batches)
    if not quiet:
        batch_iter = tqdm(batch_iter)

    for batch in batch_iter:
        clean_inputs = resolve_run_inputs(backend_obj, batch.clean_inputs)
        corrupt_inputs = resolve_run_inputs(backend_obj, batch.corrupt_inputs)

        run_clean = clean_inputs.run_inputs
        run_corrupt = corrupt_inputs.run_inputs

        clean_shape = (
            run_clean.input_ids.shape if run_clean.input_ids is not None else run_clean.inputs_embeds.shape[:2]
        )
        corrupt_shape = (
            run_corrupt.input_ids.shape
            if run_corrupt.input_ids is not None
            else run_corrupt.inputs_embeds.shape[:2]
        )
        if clean_shape != corrupt_shape:
            raise ValueError("clean/corrupt token shapes must match")

        total_items += batch.batch_size
        with torch.inference_mode():
            _ = forward_with_hooks(backend_obj, corrupt_inputs)
            _ = forward_with_hooks(backend_obj, clean_inputs)

    if total_items == 0:
        raise ValueError("Cannot score an empty batch iterable")

    return scores


def get_scores_eap(
    model: Any,
    graph: Graph,
    batches: Iterable[BatchLike],
    metric: MetricFn,
    *,
    backend: Optional[ModelBackend] = None,
    intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
    intervention_batches: Optional[Iterable[BatchLike]] = None,
    quiet: bool = False,
) -> Tensor:
    """Get edge attribution scores using EAP."""

    backend_obj = _resolve_backend_for_method(model, backend, method="EAP")

    scores = _new_scores(backend_obj, graph)
    means = _prepare_means(backend_obj, graph, intervention, intervention_batches)

    total_items = 0
    batch_iter = iter_prepared_batches(backend_obj.tokenization_model, batches)
    if not quiet:
        batch_iter = tqdm(batch_iter)

    for batch in batch_iter:
        clean_inputs = resolve_run_inputs(backend_obj, batch.clean_inputs)
        corrupt_inputs = resolve_run_inputs(backend_obj, batch.corrupt_inputs)

        run_clean = clean_inputs.run_inputs
        run_corrupt = corrupt_inputs.run_inputs
        clean_shape = (
            run_clean.input_ids.shape if run_clean.input_ids is not None else run_clean.inputs_embeds.shape[:2]
        )
        corrupt_shape = (
            run_corrupt.input_ids.shape
            if run_corrupt.input_ids is not None
            else run_corrupt.inputs_embeds.shape[:2]
        )
        if clean_shape != corrupt_shape:
            raise ValueError("clean/corrupt token shapes must match")

        batch_size = batch.batch_size
        total_items += batch_size
        n_pos = int(clean_shape[1])

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(
            backend_obj,
            graph,
            batch_size,
            n_pos,
            scores,
        )

        with torch.inference_mode():
            if intervention == "patching":
                _ = forward_with_hooks(backend_obj, corrupt_inputs, fwd_hooks=fwd_hooks_corrupted)
            elif means is not None:
                if means.shape[1] not in (1, n_pos):
                    raise ValueError(
                        "mean activation length mismatch. "
                        f"Expected seq length {n_pos}, got {means.shape[1]}"
                    )
                activation_difference += means

            clean_logits = forward_with_hooks(backend_obj, clean_inputs)

        backend_obj.zero_grad()
        logits = forward_with_hooks(
            backend_obj,
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
    model: Any,
    graph: Graph,
    batches: Iterable[BatchLike],
    metric: MetricFn,
    *,
    backend: Optional[ModelBackend] = None,
    steps: int = 30,
    quiet: bool = False,
) -> Tensor:
    """Get edge attribution scores using EAP integrated gradients over input stream."""

    backend_obj = _resolve_backend_for_method(model, backend, method="EAP-IG-inputs")

    if steps <= 0:
        raise ValueError("steps must be positive")

    scores = _new_scores(backend_obj, graph)

    total_items = 0
    total_steps = 0

    batch_iter = iter_prepared_batches(backend_obj.tokenization_model, batches)
    if not quiet:
        batch_iter = tqdm(batch_iter)

    for batch in batch_iter:
        clean_inputs = resolve_run_inputs(backend_obj, batch.clean_inputs)
        corrupt_inputs = resolve_run_inputs(backend_obj, batch.corrupt_inputs)

        run_clean = clean_inputs.run_inputs
        run_corrupt = corrupt_inputs.run_inputs
        clean_shape = (
            run_clean.input_ids.shape if run_clean.input_ids is not None else run_clean.inputs_embeds.shape[:2]
        )
        corrupt_shape = (
            run_corrupt.input_ids.shape
            if run_corrupt.input_ids is not None
            else run_corrupt.inputs_embeds.shape[:2]
        )
        if clean_shape != corrupt_shape:
            raise ValueError("clean/corrupt token shapes must match")

        batch_size = batch.batch_size
        total_items += batch_size
        n_pos = int(clean_shape[1])

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(
            backend_obj,
            graph,
            batch_size,
            n_pos,
            scores,
        )

        with torch.inference_mode():
            _ = forward_with_hooks(backend_obj, corrupt_inputs, fwd_hooks=fwd_hooks_corrupted)

            input_idx = graph.forward_index(graph.nodes["input"])
            input_acts_corrupt = activation_difference[:, :, input_idx].clone()

            clean_logits = forward_with_hooks(backend_obj, clean_inputs, fwd_hooks=fwd_hooks_clean)
            input_acts_clean = input_acts_corrupt - activation_difference[:, :, input_idx]

        total_steps = 0
        for step in range(0, steps):
            total_steps += 1
            alpha = step / steps

            def input_interpolation_hook(activations, hook, interpolation_alpha: float = alpha):
                new_input = input_acts_corrupt + interpolation_alpha * (
                    input_acts_clean - input_acts_corrupt
                )
                new_input.requires_grad_(True)
                return new_input

            logits = forward_with_hooks(
                backend_obj,
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
    model: Any,
    graph: Graph,
    batches: Iterable[BatchLike],
    metric: MetricFn,
    *,
    backend: Optional[ModelBackend] = None,
    intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
    steps: int = 30,
    intervention_batches: Optional[Iterable[BatchLike]] = None,
    quiet: bool = False,
) -> Tensor:
    """Get edge attribution scores using IG over node activations."""

    backend_obj = _resolve_backend_for_method(model, backend, method="EAP-IG-activations")

    if steps <= 0:
        raise ValueError("steps must be positive")

    means = _prepare_means(backend_obj, graph, intervention, intervention_batches)
    scores = _new_scores(backend_obj, graph)

    total_items = 0
    total_steps = 0

    batch_iter = iter_prepared_batches(backend_obj.tokenization_model, batches)
    if not quiet:
        batch_iter = tqdm(batch_iter)

    for batch in batch_iter:
        clean_inputs = resolve_run_inputs(backend_obj, batch.clean_inputs)
        corrupt_inputs = resolve_run_inputs(backend_obj, batch.corrupt_inputs)

        run_clean = clean_inputs.run_inputs
        run_corrupt = corrupt_inputs.run_inputs
        clean_shape = (
            run_clean.input_ids.shape if run_clean.input_ids is not None else run_clean.inputs_embeds.shape[:2]
        )
        corrupt_shape = (
            run_corrupt.input_ids.shape
            if run_corrupt.input_ids is not None
            else run_corrupt.inputs_embeds.shape[:2]
        )
        if clean_shape != corrupt_shape:
            raise ValueError("clean/corrupt token shapes must match")

        batch_size = batch.batch_size
        total_items += batch_size
        n_pos = int(clean_shape[1])

        (_, _, bwd_hooks), activation_difference = make_hooks_and_matrices(
            backend_obj,
            graph,
            batch_size,
            n_pos,
            scores,
        )
        (fwd_hooks_corrupted, _, _), activations_corrupted = make_hooks_and_matrices(
            backend_obj,
            graph,
            batch_size,
            n_pos,
            scores,
        )
        (fwd_hooks_clean, _, _), activations_clean = make_hooks_and_matrices(
            backend_obj,
            graph,
            batch_size,
            n_pos,
            scores,
        )

        if intervention == "patching":
            with torch.inference_mode():
                _ = forward_with_hooks(backend_obj, corrupt_inputs, fwd_hooks=fwd_hooks_corrupted)
        elif means is not None:
            if means.shape[1] not in (1, n_pos):
                raise ValueError(
                    "mean activation length mismatch. "
                    f"Expected seq length {n_pos}, got {means.shape[1]}"
                )
            activation_difference += means

        with torch.inference_mode():
            clean_logits = forward_with_hooks(backend_obj, clean_inputs, fwd_hooks=fwd_hooks_clean)
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

        batch_steps = 0
        for node in nodes_list:
            clean_acts = activations_clean[:, :, graph.forward_index(node)]
            corrupt_acts = activations_corrupted[:, :, graph.forward_index(node)]
            for step in range(1, steps + 1):
                batch_steps += 1
                alpha = step / steps
                backend_obj.zero_grad()
                logits = forward_with_hooks(
                    backend_obj,
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
        # Parity-first with vendor/eap-ig: total_steps is tracked per batch and
        # overwritten, not accumulated across batches.
        total_steps = batch_steps

    if total_items == 0:
        raise ValueError("Cannot score an empty batch iterable")

    scores /= total_items
    scores /= total_steps
    return scores


def get_scores_clean_corrupted(
    model: Any,
    graph: Graph,
    batches: Iterable[BatchLike],
    metric: MetricFn,
    *,
    backend: Optional[ModelBackend] = None,
    quiet: bool = False,
) -> Tensor:
    """Two-point clean/corrupted approximation of IG."""

    backend_obj = _resolve_backend_for_method(model, backend, method="clean-corrupted")

    scores = _new_scores(backend_obj, graph)

    total_items = 0
    total_steps = 2

    batch_iter = iter_prepared_batches(backend_obj.tokenization_model, batches)
    if not quiet:
        batch_iter = tqdm(batch_iter)

    for batch in batch_iter:
        clean_inputs = resolve_run_inputs(backend_obj, batch.clean_inputs)
        corrupt_inputs = resolve_run_inputs(backend_obj, batch.corrupt_inputs)

        run_clean = clean_inputs.run_inputs
        run_corrupt = corrupt_inputs.run_inputs
        clean_shape = (
            run_clean.input_ids.shape if run_clean.input_ids is not None else run_clean.inputs_embeds.shape[:2]
        )
        corrupt_shape = (
            run_corrupt.input_ids.shape
            if run_corrupt.input_ids is not None
            else run_corrupt.inputs_embeds.shape[:2]
        )
        if clean_shape != corrupt_shape:
            raise ValueError("clean/corrupt token shapes must match")

        batch_size = batch.batch_size
        total_items += batch_size
        n_pos = int(clean_shape[1])

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), _ = make_hooks_and_matrices(
            backend_obj,
            graph,
            batch_size,
            n_pos,
            scores,
        )

        with torch.inference_mode():
            _ = forward_with_hooks(backend_obj, corrupt_inputs, fwd_hooks=fwd_hooks_corrupted)
            clean_logits = forward_with_hooks(backend_obj, clean_inputs, fwd_hooks=fwd_hooks_clean)

        backend_obj.zero_grad()
        logits = forward_with_hooks(backend_obj, clean_inputs, bwd_hooks=bwd_hooks)
        metric(logits, clean_logits, batch).backward()

        backend_obj.zero_grad()
        corrupted_logits = forward_with_hooks(backend_obj, corrupt_inputs, bwd_hooks=bwd_hooks)
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
_backward_methods = {"EAP", "EAP-IG-inputs", "clean-corrupted", "EAP-IG-activations"}


def _assert_autograd_ready_for_attribution(backend: ModelBackend) -> None:
    if not torch.is_grad_enabled():
        raise RuntimeError(
            "Attribution requires autograd to be enabled. "
            "Do not wrap attribute() in torch.no_grad() or torch.inference_mode()."
        )

    if not any(param.requires_grad for param in backend.parameters()):
        raise RuntimeError(
            "Attribution requires at least one model parameter with requires_grad=True. "
            "Enable grads before scoring (for example: model.requires_grad_(True))."
        )


def attribute(
    model: Any,
    graph: Graph,
    batches: Iterable[BatchLike],
    metric: MetricFn,
    *,
    backend: Optional[ModelBackend] = None,
    method: Literal["EAP", "EAP-IG-inputs", "clean-corrupted", "EAP-IG-activations", "exact", "smoke"],
    intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
    aggregation: Literal["sum", "mean"] = "sum",
    ig_steps: Optional[int] = None,
    intervention_batches: Optional[Iterable[BatchLike]] = None,
    quiet: bool = False,
) -> Tensor:
    """Unified attribution entrypoint for multimodal prepared batches."""

    backend_obj = _resolve_backend_for_method(model, backend, method=method)
    cfg = backend_obj.config

    assert cfg.use_attn_result, "Model must enable use_attn_result"
    assert cfg.use_split_qkv_input, "Model must enable use_split_qkv_input"
    assert cfg.use_hook_mlp_in, "Model must enable use_hook_mlp_in"
    if cfg.n_key_value_heads is not None:
        # GQA ungrouping analogue; presence alone is fine in smoke stage.
        pass

    if aggregation not in allowed_aggregations:
        raise ValueError(f"aggregation must be in {allowed_aggregations}, but got {aggregation}")

    steps = ig_steps if ig_steps is not None else 30
    if method in _backward_methods:
        _assert_autograd_ready_for_attribution(backend_obj)

    if method == "smoke":
        scores = get_scores_smoke(model, graph, batches, metric, backend=backend_obj, quiet=quiet)
    elif method == "EAP":
        scores = get_scores_eap(
            model,
            graph,
            batches,
            metric,
            backend=backend_obj,
            intervention=intervention,
            intervention_batches=intervention_batches,
            quiet=quiet,
        )
    elif method == "EAP-IG-inputs":
        if intervention != "patching":
            raise ValueError("intervention must be 'patching' for EAP-IG-inputs")
        scores = get_scores_eap_ig(
            model,
            graph,
            batches,
            metric,
            backend=backend_obj,
            steps=steps,
            quiet=quiet,
        )
    elif method == "clean-corrupted":
        if intervention != "patching":
            raise ValueError("intervention must be 'patching' for clean-corrupted")
        scores = get_scores_clean_corrupted(
            model,
            graph,
            batches,
            metric,
            backend=backend_obj,
            quiet=quiet,
        )
    elif method == "EAP-IG-activations":
        scores = get_scores_ig_activations(
            model,
            graph,
            batches,
            metric,
            backend=backend_obj,
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
            backend=backend_obj,
            intervention=intervention,
            intervention_batches=intervention_batches,
            quiet=quiet,
        )
    else:
        raise ValueError(
            "method must be one of ['smoke', 'EAP', 'EAP-IG-inputs', 'clean-corrupted', "
            "'EAP-IG-activations', 'exact']"
        )

    if aggregation == "mean":
        scores = scores / cfg.d_model

    graph.scores[:] = scores.to(graph.scores.device)
    return graph.scores
