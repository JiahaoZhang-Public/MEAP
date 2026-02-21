# AttributionModel Public API Design (v2 MVP)

Status: Implemented (MVP)  
Audience: end users of `meap` public API and maintainers

## 1. Problem Definition

`meap` should optimize for exactly two core responsibilities:

1. `model -> language trunk -> graph/hook plan`
2. `prepared model inputs -> attribution run`

For multimodal research, users often already own the preprocessing stack (prompt template, processor, modality packing, truncation policy).  
Core `meap` should not own raw clean/corrupt sample preparation.

## 2. Scope and Non-Goals (MVP)

### In Scope

- Resolve language-model trunk from an HF model (automatic or explicit).
- Build attribution graph from resolved trunk/backend config.
- Accept prepared inputs only, validate, run attribution.
- Expose component-level targeting types for attribution (layers/heads/modules/qkv).

### Out of Scope (Core API)

- Raw data processing (`clean_samples`, `corrupt_samples`).
- Processor orchestration (`processor=...`) in core entrypoints.
- Prompt-template decisions for modality placeholders.

These can remain in optional helper modules (`contrib`) or user code.

## 3. Design Principles

1. One object-centric entrypoint: `AttributionModel`.
2. Core accepts `PreparedBatch` (or equivalent tensor mapping only).
3. Explicit contracts over implicit behavior:
- no implicit truncation,
- no hidden processor logic,
- no hidden clean/corrupt construction.
4. Backward compatibility through deprecation wrappers, not dual semantics in core.

## 4. Stable Public Surface (MVP)

Primary module:
- `meap.api` (or `meap` package top-level re-export)

### 4.1 Core class

```python
class AttributionModel:
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
    ) -> "AttributionModel": ...

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
    ) -> "AttributionModel": ...

    @property
    def route_info(self) -> "RouteInfo": ...

    def build_graph(
        self,
        *,
        components: "ComponentSpec | None" = None,  # accepted in MVP
        neuron_level: bool = False,
        node_scores: bool = False,
    ) -> Graph: ...

    def attribute(
        self,
        *,
        batches: Iterable["PreparedInputLike"],
        method: AttributionMethod = "EAP",
        metric: MetricFn | None = None,
        task: "TaskSpecVNext | None" = None,
        graph: Graph | None = None,
        components: "ComponentSpec | None" = None,
        ig_steps: int | None = None,
        intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
        aggregation: Literal["sum", "mean"] = "sum",
        intervention_batches: Iterable["PreparedInputLike"] | None = None,
        quiet: bool = False,
    ) -> "AttributionResult": ...

    def evaluate_graph(
        self,
        *,
        graph: Graph,
        batches: Iterable["PreparedInputLike"],
        metric: MetricFn | None = None,
        task: "TaskSpecVNext | None" = None,
        intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
        intervention_batches: Iterable["PreparedInputLike"] | None = None,
        skip_clean: bool = True,
        quiet: bool = False,
    ) -> torch.Tensor: ...

    def validate_batches(
        self,
        batches: Iterable["PreparedInputLike"],
    ) -> list[PreparedBatch]: ...
```

### 4.2 Stable data objects

```python
@dataclass
class RouteInfo:
    model_id_or_path: str | None
    adapter_name: str
    arch_kind: str
    language_trunk_path: str
    diagnostics: dict[str, Any]
```

```python
@dataclass
class ComponentSpec:
    layers: Sequence[int] | None = None
    heads: Sequence[int] | None = None
    include: Sequence[Literal["attn", "mlp", "input", "logits"]] | None = None
    qkv: Sequence[Literal["q", "k", "v"]] | None = None
```

```python
@dataclass
class AttributionResult:
    graph: Graph
    scores: torch.Tensor
    route_info: RouteInfo
    run_info: dict[str, Any]
```

```python
PreparedInputLike = PreparedBatch | Mapping[str, Any]
```

If `PreparedInputLike` is mapping, canonical schema is:

```python
{
  "clean_inputs": Dict[str, TensorLike],
  "corrupt_inputs": Dict[str, TensorLike],
  "labels": Any,                     # optional
  "input_lengths": TensorLike,       # optional (derived from attention_mask when absent)
  "meta": Dict[str, Any] | None      # optional
}
```

Core runtime normalizes this into `PreparedBatch` and runs `validate_prepared_batch(...)`.

## 5. Execution Contract (Implemented)

### 5.1 Input contract

- Core APIs do not accept:
  - `clean_samples`
  - `corrupt_samples`
  - `processor`
  - `pair_batch_preparer`
- Core APIs accept only prepared tensors (`PreparedBatch` or equivalent mapping).

### 5.2 Metric/task contract

`AttributionModel.attribute(...)` requires exactly one of:

- `metric` (custom metric callback), or
- `task` (built-in task spec, e.g. next-token / choice-classification).

Providing both or neither is an error.

### 5.3 Component targeting contract

If `components` is provided:

- graph contains only selected attribution components, and
- hook plan is compiled only for selected components.

If omitted:

- default full graph behavior is used.

## 6. Ideal User Flows

### Flow A: Multimodal user-owned preprocessing (recommended)

```python
from transformers import AutoProcessor
from meap import AttributionModel, PreparedBatch

model = AttributionModel.from_pretrained("Qwen/Qwen2-VL-2B")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B")

# User-owned preprocessing (outside core meap):
clean_inputs = processor(text=[clean_prompt], images=[clean_image], return_tensors="pt", padding=True)
corrupt_inputs = processor(text=[corrupt_prompt], images=[corrupt_image], return_tensors="pt", padding=True)

batch = PreparedBatch(
    clean_inputs=dict(clean_inputs),
    corrupt_inputs=dict(corrupt_inputs),
    labels=labels,
    input_lengths=dict(clean_inputs)["attention_mask"].sum(dim=-1),
)

result = model.attribute(batches=[batch], method="EAP", metric=my_metric)
```

### Flow B: Explicit architecture route

```python
model = AttributionModel.from_pretrained(
    "/path/to/local/model",
    adapter_name="llama_like",
    language_trunk_path="model.language_model.model",
)
print(model.route_info)
```

### Flow C: Component-scoped attribution

```python
graph = model.build_graph(
    components=ComponentSpec(
        layers=[8, 9, 10],
        include=["attn", "mlp"],
        heads=[0, 1, 2],
        qkv=["q", "k", "v"],
    )
)
result = model.attribute(batches=[batch], graph=graph, method="EAP", metric=my_metric)
```

## 7. Public Error Model

Suggested stable error categories:

- `RouteResolutionError`: trunk/adapter resolution failed.
- `PreparedInputValidationError`: prepared batch schema or alignment invalid.
- `ComponentSelectionError`: invalid/unsupported layer-head-qkv selection.
- `AttributionRuntimeError`: runtime failure during hook execution/backprop.

Each should include actionable context (model id/path, adapter, trunk path, batch index, hook name).

## 8. Migration Plan from v2

### Legacy API removal status

- `discover_circuit(...)` removed from `meap.api` in `v1.4.0`.
- `attribute_from_dataloader(...)` removed from `meap.api` in `v1.4.0`.

### Move to optional helper lane

- `HFProcessorAdapter` and raw clean/corrupt flows remain available as convenience utilities,
  but not part of core attribution contract.

## 9. Summary

The v2 MVP public API makes `meap` explicit and composable:

1. `AttributionModel` resolves language trunk and owns graph/hook runtime.
2. Users own multimodal preprocessing and pass prepared model tensors.
3. Core validates and runs attribution only.

## 10. MVP Limits

1. `ComponentSpec` is currently a public type contract only.
- In MVP, passing `components` does not yet apply hook-level filtering.
- `build_graph(...)` returns the full graph and emits a warning when `components` is provided.
2. `AttributionModel.from_pretrained(...)` uses `AutoModel`.
- If auto loading fails for a specific model class or local checkpoint setup, load model in user code and use `AttributionModel.from_model(...)`.
