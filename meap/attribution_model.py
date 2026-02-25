from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from typing import Any, Iterable, Literal, Mapping, Sequence, TypeAlias, Union
import warnings

import torch
from torch import Tensor

from .attribute import attribute
from .backend import HFLLMBackend
from .batch import PreparedBatch, RawPairBatch, validate_prepared_batch
from .evaluate import evaluate_graph as evaluate_graph_impl
from .graph import Graph
from .tasking import MetricFn, TaskSpec, build_task_metric

AttributionMethod = Literal[
    "EAP",
    "EAP-IG-inputs",
    "clean-corrupted",
    "EAP-IG-activations",
    "exact",
    "smoke",
]
InterventionMethod = Literal["patching", "zero", "mean", "mean-positional"]
AggregationMethod = Literal["sum", "mean"]


@dataclass(frozen=True)
class RouteInfo:
    model_id_or_path: str | None
    adapter_name: str
    arch_kind: str
    language_trunk_path: str
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class ComponentSpec:
    layers: Sequence[int] | None = None
    heads: Sequence[int] | None = None
    include: Sequence[Literal["attn", "mlp", "input", "logits"]] | None = None
    qkv: Sequence[Literal["q", "k", "v"]] | None = None


@dataclass
class AttributionResult:
    graph: Graph
    scores: Tensor
    route_info: RouteInfo
    run_info: dict[str, Any]


PreparedInputLike: TypeAlias = Union[PreparedBatch, Mapping[str, Any]]


def _normalize_model_ref(model_id_or_path: str) -> str:
    if os.path.isdir(model_id_or_path):
        return os.path.abspath(model_id_or_path)
    return model_id_or_path


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    raise TypeError(f"device must be str|torch.device, got {type(device)}")


def _resolve_dtype(dtype: str | torch.dtype, device: torch.device) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if not isinstance(dtype, str):
        raise TypeError(f"dtype must be str|torch.dtype, got {type(dtype)}")

    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype}'. Expected one of {sorted(mapping)} or 'auto'.")
    return mapping[dtype]


def _resolve_optional_device(device: str | torch.device | None) -> torch.device | None:
    if device is None:
        return None
    return _resolve_device(device)


def _resolve_optional_dtype(
    dtype: str | torch.dtype | None,
    *,
    device: torch.device | None,
) -> torch.dtype | None:
    if dtype is None:
        return None
    if isinstance(dtype, str) and dtype == "auto" and device is None:
        return None
    resolved_device = device if device is not None else torch.device("cpu")
    return _resolve_dtype(dtype, resolved_device)


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


def _mapping_tensor_device(mapping: Mapping[str, Any]) -> torch.device:
    for value in mapping.values():
        if torch.is_tensor(value):
            return value.device
    return torch.device("cpu")


class AttributionModel:
    _FROM_PRETRAINED_CACHE: dict[str, "AttributionModel"] = {}

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        backend: HFLLMBackend,
        model_id_or_path: str | None,
    ):
        self._model = model
        self._backend = backend
        self._model_id_or_path = model_id_or_path

    @classmethod
    def _cache_key(
        cls,
        *,
        model_id_or_path: str,
        device: torch.device,
        dtype: torch.dtype,
        adapter_name: str | None,
        language_trunk_path: str | None,
        strict_arch: bool,
        model_kwargs: dict[str, Any] | None,
    ) -> str:
        kwargs_key = json.dumps(model_kwargs or {}, sort_keys=True, default=str)
        return "|".join(
            [
                model_id_or_path,
                str(device),
                str(dtype),
                str(adapter_name),
                str(language_trunk_path),
                str(strict_arch),
                kwargs_key,
            ]
        )

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        *,
        device: str | torch.device = "auto",
        dtype: str | torch.dtype = "auto",
        model_kwargs: dict[str, Any] | None = None,
        adapter_name: str | None = None,
        language_trunk_path: str | None = None,
        strict_arch: bool = True,
        cache: bool = True,
    ) -> "AttributionModel":
        normalized_ref = _normalize_model_ref(model_id_or_path)
        resolved_device = _resolve_device(device)
        resolved_dtype = _resolve_dtype(dtype, resolved_device)
        key = cls._cache_key(
            model_id_or_path=normalized_ref,
            device=resolved_device,
            dtype=resolved_dtype,
            adapter_name=adapter_name,
            language_trunk_path=language_trunk_path,
            strict_arch=strict_arch,
            model_kwargs=model_kwargs,
        )
        if cache and key in cls._FROM_PRETRAINED_CACHE:
            return cls._FROM_PRETRAINED_CACHE[key]

        from transformers import AutoModel, AutoModelForCausalLM

        resolved_model_kwargs = model_kwargs or {}
        load_errors: list[str] = []
        model = None
        for loader_name, loader in (
            ("AutoModel", AutoModel),
            ("AutoModelForCausalLM", AutoModelForCausalLM),
        ):
            try:
                model = loader.from_pretrained(normalized_ref, **resolved_model_kwargs)
                break
            except Exception as exc:  # noqa: BLE001
                load_errors.append(f"{loader_name}: {exc}")

        if model is None:
            errors_joined = " | ".join(load_errors)
            raise RuntimeError(
                "AttributionModel.from_pretrained failed to load model. "
                "Tried AutoModel and AutoModelForCausalLM. "
                "You can load the model in user code and call "
                "AttributionModel.from_model(model=...) as a fallback. "
                f"model_id_or_path='{normalized_ref}', errors={errors_joined}"
            )

        model.eval()
        backend = HFLLMBackend(
            model,
            tokenizer=None,
            device=resolved_device,
            dtype=resolved_dtype,
            adapter_name=adapter_name,
            language_trunk_path=language_trunk_path,
            strict_arch=strict_arch,
        )
        out = cls(model=model, backend=backend, model_id_or_path=normalized_ref)
        if cache:
            cls._FROM_PRETRAINED_CACHE[key] = out
        return out

    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        *,
        tokenizer: Any | None = None,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype | None = None,
        adapter_name: str | None = None,
        language_trunk_path: str | None = None,
        strict_arch: bool = True,
    ) -> "AttributionModel":
        resolved_device = _resolve_optional_device(device)
        resolved_dtype = _resolve_optional_dtype(dtype, device=resolved_device)
        model.eval()
        backend = HFLLMBackend(
            model,
            tokenizer=tokenizer,
            device=resolved_device,
            dtype=resolved_dtype,
            adapter_name=adapter_name,
            language_trunk_path=language_trunk_path,
            strict_arch=strict_arch,
        )
        return cls(model=model, backend=backend, model_id_or_path=None)

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def backend(self) -> HFLLMBackend:
        return self._backend

    @property
    def route_info(self) -> RouteInfo:
        return RouteInfo(
            model_id_or_path=self._model_id_or_path,
            adapter_name=self._backend.adapter_name,
            arch_kind=self._backend.arch_kind,
            language_trunk_path=self._backend.language_trunk_path,
            diagnostics=self._backend.resolution_diagnostics,
        )

    def build_graph(
        self,
        *,
        components: ComponentSpec | None = None,
        neuron_level: bool = False,
        node_scores: bool = False,
    ) -> Graph:
        if components is not None:
            warnings.warn(
                "ComponentSpec filtering is deferred in AttributionModel MVP. "
                "build_graph currently returns a full graph.",
                UserWarning,
                stacklevel=2,
            )
        return Graph.from_model(
            self._backend.config,
            neuron_level=neuron_level,
            node_scores=node_scores,
        )

    def _iter_inputs(self, batches: Iterable[PreparedInputLike] | PreparedInputLike):
        if isinstance(batches, PreparedBatch) or isinstance(batches, Mapping):
            yield batches
            return
        for batch in batches:
            yield batch

    def _mapping_to_prepared(self, payload: Mapping[str, Any], *, index: int) -> PreparedBatch:
        required = ("clean_inputs", "corrupt_inputs")
        missing = [k for k in required if k not in payload]
        if missing:
            raise ValueError(
                f"batches[{index}] must include keys {required}. Missing: {missing}"
            )

        clean_inputs = payload["clean_inputs"]
        corrupt_inputs = payload["corrupt_inputs"]
        if not isinstance(clean_inputs, Mapping) or not isinstance(corrupt_inputs, Mapping):
            raise ValueError(
                f"batches[{index}] clean_inputs/corrupt_inputs must be mappings of model tensors"
            )

        clean_dict = dict(clean_inputs)
        corrupt_dict = dict(corrupt_inputs)
        labels = payload.get("labels")
        input_lengths = payload.get("input_lengths")
        if input_lengths is None:
            input_lengths = _input_lengths_from_inputs(clean_dict)
        else:
            device = _mapping_tensor_device(clean_dict)
            input_lengths = torch.as_tensor(input_lengths, dtype=torch.long, device=device).reshape(-1)

        meta = payload.get("meta")
        prepared = PreparedBatch(
            clean_inputs=clean_dict,
            corrupt_inputs=corrupt_dict,
            labels=labels,
            input_lengths=input_lengths,
            meta=meta,
        )
        validate_prepared_batch(prepared)
        return prepared

    def validate_batches(
        self,
        batches: Iterable[PreparedInputLike] | PreparedInputLike,
    ) -> list[PreparedBatch]:
        prepared_batches: list[PreparedBatch] = []
        for index, batch in enumerate(self._iter_inputs(batches)):
            if isinstance(batch, PreparedBatch):
                validate_prepared_batch(batch)
                prepared_batches.append(batch)
                continue
            if isinstance(batch, RawPairBatch):
                raise TypeError(
                    "AttributionModel core API does not accept RawPairBatch. "
                    "Provide PreparedBatch or nested prepared tensor mappings."
                )
            if isinstance(batch, Mapping):
                prepared_batches.append(self._mapping_to_prepared(batch, index=index))
                continue
            raise TypeError(
                "AttributionModel batches must be PreparedBatch or nested prepared tensor mappings. "
                f"Got {type(batch)} at batches[{index}]."
            )

        if len(prepared_batches) == 0:
            raise ValueError("AttributionModel requires at least one prepared batch.")
        return prepared_batches

    def _resolve_metric(
        self,
        *,
        metric: MetricFn | None,
        task: TaskSpec | None,
        api_name: str,
    ) -> MetricFn:
        if (metric is None and task is None) or (metric is not None and task is not None):
            raise ValueError(f"{api_name} requires exactly one of metric or task.")
        if metric is not None:
            return metric
        assert task is not None
        return build_task_metric(task)

    def attribute(
        self,
        *,
        batches: Iterable[PreparedInputLike] | PreparedInputLike,
        method: AttributionMethod = "EAP",
        metric: MetricFn | None = None,
        task: TaskSpec | None = None,
        graph: Graph | None = None,
        components: ComponentSpec | None = None,
        ig_steps: int | None = None,
        intervention: InterventionMethod = "patching",
        aggregation: AggregationMethod = "sum",
        intervention_batches: Iterable[PreparedInputLike] | PreparedInputLike | None = None,
        quiet: bool = False,
    ) -> AttributionResult:
        metric_fn = self._resolve_metric(metric=metric, task=task, api_name="AttributionModel.attribute")
        run_graph = graph if graph is not None else self.build_graph(components=components)
        prepared_batches = self.validate_batches(batches)
        prepared_intervention = None
        if intervention_batches is not None:
            prepared_intervention = self.validate_batches(intervention_batches)

        scores = attribute(
            model=self._model,
            graph=run_graph,
            batches=prepared_batches,
            metric=metric_fn,
            backend=self._backend,
            method=method,
            intervention=intervention,
            aggregation=aggregation,
            ig_steps=ig_steps,
            intervention_batches=prepared_intervention,
            quiet=quiet,
        )
        run_info = {
            "method": method,
            "intervention": intervention,
            "aggregation": aggregation,
            "ig_steps": ig_steps,
            "n_batches": len(prepared_batches),
            "n_examples": int(sum(batch.batch_size for batch in prepared_batches)),
            "components": (None if components is None else asdict(components)),
        }
        return AttributionResult(
            graph=run_graph,
            scores=scores,
            route_info=self.route_info,
            run_info=run_info,
        )

    def evaluate_graph(
        self,
        *,
        graph: Graph,
        batches: Iterable[PreparedInputLike] | PreparedInputLike,
        metric: MetricFn | None = None,
        task: TaskSpec | None = None,
        intervention: InterventionMethod = "patching",
        intervention_batches: Iterable[PreparedInputLike] | PreparedInputLike | None = None,
        skip_clean: bool = True,
        quiet: bool = False,
    ) -> Tensor:
        metric_fn = self._resolve_metric(
            metric=metric,
            task=task,
            api_name="AttributionModel.evaluate_graph",
        )
        prepared_batches = self.validate_batches(batches)
        prepared_intervention = None
        if intervention_batches is not None:
            prepared_intervention = self.validate_batches(intervention_batches)

        results = evaluate_graph_impl(
            model=self._model,
            graph=graph,
            batches=prepared_batches,
            metrics=metric_fn,
            backend=self._backend,
            quiet=quiet,
            intervention=intervention,
            intervention_batches=prepared_intervention,
            skip_clean=skip_clean,
        )
        assert torch.is_tensor(results)
        return results


__all__ = [
    "AggregationMethod",
    "AttributionMethod",
    "AttributionModel",
    "AttributionResult",
    "ComponentSpec",
    "InterventionMethod",
    "PreparedInputLike",
    "RouteInfo",
]
