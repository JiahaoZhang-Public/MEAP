# multimodal_lm_eap_ig Package Guide

This package implements attribution methods (EAP / EAP-IG variants / exact / smoke) on top of model backends, with a focus on Hugging Face models and language-trunk attribution for multimodal systems.

## Design Goal

- User controls data preparation.
- Package core focuses on:
1. inferring model architecture/modules from backend models,
2. registering hook points,
3. running attribution methods.

## Module Map

- `api.py`: high-level dataloader APIs (`attribute_from_dataloader`, `evaluate_*_from_dataloader`).
- `attribute.py`: attribution method implementations.
- `evaluate.py`: baseline and graph evaluation logic.
- `graph.py`: graph/node/edge representation and indexing.
- `batch.py`: `PreparedBatch` contract and batch iteration/validation.
- `preparer.py`: optional user-facing helper (`HFProcessorAdapter`) for raw clean/corrupt -> `PreparedBatch`.
- `backend/`:
  - `hf_backend.py`: HF backend hook/runtime implementation.
  - `tlens_backend.py`: TransformerLens adapter backend.
  - `registry.py`: architecture adapter registry and diagnostics.
  - `adapters/*`: architecture-specific module mapping (`llama_like`, `gpt2_like`, `opt_like`, `falcon_like`, `mpt_like`).
- `utils.py`: hook wiring and execution helpers.

## Input Contract

### Preferred Mode A: user provides `PreparedBatch`

Your dataloader yields `PreparedBatch` objects directly.

### Mode B: user provides raw clean/corrupt + explicit preparer

Your dataloader yields raw pair data, and you pass one of:
- `processor=...` (wrapped into `HFProcessorAdapter`), or
- `pair_batch_preparer=...` (custom preparer).

`processor` and `pair_batch_preparer` are mutually exclusive.

## `PreparedBatch` Requirements

A valid `PreparedBatch` must satisfy:
- clean/corrupt batch size match,
- clean/corrupt sequence length match,
- `attention_mask` semantics match between clean/corrupt when present,
- `input_lengths == attention_mask.sum(-1)` when mask is present.

Validation is enforced by `validate_prepared_batch` in `batch.py`.

## High-level API Notes

`attribute_from_dataloader` / `evaluate_*_from_dataloader`:
- always require an explicit backend for non-TransformerLens models,
- no implicit sequence truncation; `max_length` is not applied by API,
- truncation should be performed inside your processor/preparer.

## Backend Architecture Inference

`HFLLMBackend` resolves architecture by adapter registry:
1. find decoder-like backbone candidates,
2. select matching adapter,
3. extract modules for hook points (`hook_embed`, `attn.hook_result`, q/k/v input hooks, mlp hooks, resid post).

Diagnostic helpers:
- `inspect_model_architecture(model)`
- `register_architecture_adapter(...)`

## Methods

Supported method names:
- `smoke`
- `EAP`
- `EAP-IG-inputs`
- `clean-corrupted`
- `EAP-IG-activations`
- `exact`

Method execution routes are implemented in `attribute.py`.

### Text Parity Status (PR2)

Tier A models with strict vendor/TLens/HF parity target:
- `gpt2-small` (HF id: `gpt2`)
- `Qwen/Qwen2-0.5B`

For `EAP-IG-activations`, implementation follows vendor parity-first semantics, including
the per-batch step normalization behavior from upstream `vendor/eap-ig`.

## Known Limits

- Attribution currently targets language-model trunk for multimodal models.
- Falcon with interleaved fused QKV (`new_decoder_architecture=True`) is not yet supported for intervention path.
- `exact` can be expensive on large graphs; parity scripts default to skip when
  `edge_count > --max-exact-edges` and report an explicit skip reason.

## Minimal Usage

### A) Prebuilt `PreparedBatch`

```python
from multimodal_lm_eap_ig import attribute_from_dataloader, HFLLMBackend

backend = HFLLMBackend(model)
result = attribute_from_dataloader(
    model=model,
    backend=backend,
    dataloader=[prepared_batch],
    metric=my_metric,
    method="EAP",
)
```

### B) Raw clean/corrupt with explicit processor

```python
from multimodal_lm_eap_ig import attribute_from_dataloader, HFLLMBackend

backend = HFLLMBackend(model, tokenizer=getattr(processor, "tokenizer", None))
result = attribute_from_dataloader(
    model=model,
    backend=backend,
    dataloader=[{"clean": clean_samples, "corrupt": corrupt_samples, "labels": labels}],
    processor=processor,
    metric=my_metric,
    method="smoke",
)
```

## Development Checklist

When adding new model architecture support:
1. implement adapter under `backend/adapters/`.
2. register adapter in registry.
3. validate hook scaffold via backend tests.
4. run smoke and parity scripts.
