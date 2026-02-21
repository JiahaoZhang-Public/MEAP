# AttributionModel API Design (v1.4.0)

Status: Implemented (MVP)

This page explains why the current API is shaped this way and how to use it effectively.
For normative stability guarantees, see `docs/docs/API_STABILITY.md`.

## 1. Design Goal

`meap` v1.4.0 centers on two responsibilities:

1. `model -> language trunk -> graph/hook runtime`
2. `prepared model inputs -> attribution`

This keeps core attribution logic explicit and keeps modality-specific preprocessing in user code.

## 2. Scope Boundaries

### In scope (core)
- HF model route resolution to language trunk.
- Graph construction from resolved trunk/backend config.
- Prepared-input validation and attribution runtime.

## 3. Object Model

Core object: `AttributionModel`

Key public data objects:
- `RouteInfo`
- `AttributionResult`
- `PreparedInputLike`
- `ComponentSpec` (MVP placeholder; see limits)

Construction paths:
- `AttributionModel.from_pretrained(...)`
- `AttributionModel.from_model(...)`

Execution methods:
- `build_graph(...)`
- `validate_batches(...)`
- `attribute(...)`
- `evaluate_graph(...)`

## 4. Prepared Input Contract

`PreparedInputLike` can be:
- `PreparedBatch`, or
- nested mapping equivalent to:

```python
{
  "clean_inputs": Dict[str, TensorLike],
  "corrupt_inputs": Dict[str, TensorLike],
  "labels": Any,                     # optional
  "input_lengths": TensorLike,       # optional
  "meta": Dict[str, Any] | None      # optional
}
```

Runtime behavior:
- mapping input is normalized to `PreparedBatch`
- missing `input_lengths` is derived (usually from `attention_mask`)
- schema/alignment failures raise validation errors (`ValueError` / `TypeError`)

## 5. Execution Semantics

### 5.1 Metric/task requirement

`AttributionModel.attribute(...)` and `AttributionModel.evaluate_graph(...)` require exactly one of:
- `metric`, or
- `task`

Providing both or neither is an error.

### 5.2 Route resolution

Route selection can be:
- fully automatic,
- adapter-constrained (`adapter_name=...`),
- trunk-constrained (`language_trunk_path=...`).

Use `route_info` for resolved adapter/trunk diagnostics.

### 5.3 from_pretrained loading behavior

MVP currently loads via `transformers.AutoModel`.
If model auto-loading is not suitable for your checkpoint/class, load the model yourself and use:
- `AttributionModel.from_model(model=...)`

This is the recommended path for many multimodal setups.

## 6. Recommended User Flows

### Flow A: text model, quick start

```python
from meap import AttributionModel

am = AttributionModel.from_pretrained("gpt2")
result = am.attribute(batches=[prepared_batch], metric=my_metric, method="EAP")
```

### Flow B: multimodal, user-owned preprocessing (recommended)

```python
from meap import AttributionModel, PreparedBatch

# user loads model and prepares tensors in their own stack
am = AttributionModel.from_model(model=my_mm_model, tokenizer=my_tokenizer)
batch = PreparedBatch(
    clean_inputs=clean_inputs,
    corrupt_inputs=corrupt_inputs,
    labels=labels,
    input_lengths=input_lengths,
)
result = am.attribute(batches=[batch], metric=my_metric, method="EAP")
```

### Flow C: explicit route override

```python
am = AttributionModel.from_pretrained(
    "/path/to/model",
    adapter_name="llama_like",
    language_trunk_path="model.language_model.model",
)
print(am.route_info)
```

## 7. MVP Limits (Current)

1. `ComponentSpec` is public as a type/interface, but hook-level component filtering is deferred.
- Passing `components` currently does not prune hook compilation.
- `build_graph(...)` returns full graph and warns when `components` is provided.

2. Core API is prepared-input-first.
- Raw sample preparation remains outside core.

## 8. Relationship with Other Docs

- Stability guarantees:
  - `docs/docs/API_STABILITY.md`
- End-to-end runnable examples:
  - `examples/README.md`
- Supported-model smoke matrix:
  - `docs/docs/SUPPORTED_MODELS.md`
- Historical changes:
  - `CHANGELOG.md`
  - `releases/v1.4.0.md`
