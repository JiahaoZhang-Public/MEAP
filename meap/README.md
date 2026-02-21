# meap Package Guide

This package implements attribution methods (EAP / EAP-IG variants / exact / smoke) on top of model backends, with a focus on Hugging Face models and language-trunk attribution for multimodal systems.

## Upstream Scaffold

The initial scaffold of this codebase was directly copied from:
- [hannamw/EAP-IG](https://github.com/hannamw/EAP-IG)

This package then added backend abstraction, HF-first runtime, multimodal input paths, matrix tooling, and release engineering workflow.

## Installation

Runtime install:

```bash
pip install meap
```

Development install:

```bash
pip install -e ".[dev,multimodal,viz,docs]"
```

## Stable API Contract (v1)

Use these as stable entrypoints:

- High-level API (`meap.api`):
  - `AttributionRunResult`
  - `TaskSpec`
  - `CircuitEdgeSummary`
  - `CircuitRunResult`
  - `discover_circuit`
  - `attribute_from_dataloader`
  - `evaluate_graph_from_dataloader`
  - `evaluate_baseline_from_dataloader`
- Core public objects:
  - `HFLLMBackend`, `TLensBackend`
  - `PreparedBatch`, `RawPairBatch`, `DictPairBatch`
  - `HFProcessorAdapter`
  - `Graph`
  - `register_architecture_adapter`, `inspect_model_architecture`, `resolve_backend`

Internal (not stability-guaranteed) modules:
- `meap.attribute`
- `meap.evaluate`
- `meap.utils`

### Deprecated Top-Level Symbols

Deprecated since `1.0.0`, planned removal in `1.2.0` from package top-level:

- `get_real_edge_scores`
- `get_scores_clean_corrupted`
- `get_scores_eap`
- `get_scores_eap_ig`
- `get_scores_exact`
- `get_scores_ig_activations`
- `get_scores_smoke`
- `build_default_llava_processor`
- `prepare_llava_token_pair_batch`

Migration:
- Import these from their implementation modules (`meap.attribute` / `meap.preparer`) if needed.
- Prefer high-level APIs in `meap.api` for long-term compatibility.

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

## API/UX Rules (PR3)

### Allowed Entrypoints

Only two input styles are supported:
1. `PreparedBatch` (fully user-prepared tensors), or
2. raw clean/corrupt samples + explicit `processor` or explicit `pair_batch_preparer`.

### Explicitly Disallowed Implicit Behavior

- No implicit truncation in high-level APIs.
- `max_length` is rejected in:
  - `attribute_from_dataloader`
  - `evaluate_graph_from_dataloader`
  - `evaluate_baseline_from_dataloader`
- If truncation is needed, do it explicitly in your own preprocessing:
  - pass `processor_kwargs={"truncation": True, "max_length": ...}`, or
  - truncate in your custom `pair_batch_preparer`, or
  - build `PreparedBatch` directly with already-truncated tensors.

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

### One-Shot HF Entry (Recommended)

Use `discover_circuit` when you want model loading + processor + attribution orchestration in one call.

```python
from meap.api import discover_circuit

result = discover_circuit(
    model_id_or_path=\"openai-community/gpt2\",  # HF hub id or local model directory
    clean_samples={\"text\": [\"The capital of France is\"]},
    corrupt_samples={\"text\": [\"The capital of Germany is\"]},
    task=\"next_token\",
    labels=[357],  # target token id(s) or {'target_token_ids': ..., 'target_positions': ...}
    method=\"EAP\",
    top_k=20,
)

print(result.backend_info)
print(result.top_edges[:3])
```

Local path example:

```python
result = discover_circuit(
    model_id_or_path=\"/path/to/local/hf/model\",
    clean_samples={\"text\": [\"Question: 2+2=\"]},
    corrupt_samples={\"text\": [\"Question: 2+3=\"]},
    task=\"next_token\",
    labels=[16],
)
```

Error diagnostics example:

```python
try:
    discover_circuit(
        model_id_or_path=\"/path/to/unsupported/model\",
        clean_samples={\"text\": [\"hello\"]},
        corrupt_samples={\"text\": [\"world\"]},
        task=\"next_token\",
        labels=[1],
    )
except ValueError as exc:
    # Includes candidate backbones, selection_error, and original backend error.
    print(exc)
```

### A) Prebuilt `PreparedBatch`

```python
from meap import attribute_from_dataloader, HFLLMBackend

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
from meap import attribute_from_dataloader, HFLLMBackend

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

## Minimal End-to-End Examples (Text / Image / Audio)

Unified script:
- `scripts/api_minimal_examples.py`

### Text: `openai-community/gpt2`

```bash
python scripts/api_minimal_examples.py \
  --example text-gpt2 \
  --method smoke \
  --device cpu \
  --dtype float32
```

### Text: `Qwen/Qwen2-0.5B`

```bash
python scripts/api_minimal_examples.py \
  --example text-qwen2 \
  --method smoke \
  --device cpu \
  --dtype float32
```

### Image-Text: `Qwen/Qwen2-VL-2B`

```bash
python scripts/api_minimal_examples.py \
  --example image-qwen2vl \
  --method smoke \
  --device cpu \
  --dtype float32
```

### Audio: `fixie-ai/ultravox-v0_5-llama-3_2-1b`

```bash
python scripts/api_minimal_examples.py \
  --example audio-ultravox \
  --method smoke \
  --device cpu \
  --dtype float32 \
  --hf-token <your_hf_token_if_needed>
```

Notes:
- Ultravox example uses custom `pair_batch_preparer` (non-standard audio pipeline preprocessing).
- If `--audio-path` is omitted, the script downloads a default demo mp3 (`glass-breaking-151256.mp3`) and uses prompt `Generate the caption in English:`.
- This model may require access to upstream gated dependencies; use `--hf-token` with proper permissions.

## PR4 Test Matrix

Unified matrix entrypoint:
- `scripts/test_stage_matrix.py`

Default text parity models (vendor/TLens aligned):
- `gpt2`
- `Qwen/Qwen2-0.5B`
- `facebook/opt-125m`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

Default multimodal smoke models:
- `Qwen/Qwen2-VL-2B`
- `llava-hf/llava-1.5-7b-hf`
- `HuggingFaceTB/SmolVLM-Instruct`
- `Qwen/Qwen2-Audio-7B` (audio fallback supported via `--audio-fallback-model`)

The stage matrix writes a unified JSON report with:
- model-level results,
- method-level results,
- error classification summary.

Example:

```bash
python scripts/test_stage_matrix.py \
  --device cpu \
  --dtype float32 \
  --strict \
  --output reports/stage_matrix.json
```

## Development Checklist

When adding new model architecture support:
1. implement adapter under `backend/adapters/`.
2. register adapter in registry.
3. validate hook scaffold via backend tests.
4. run smoke and parity scripts.

## PR5 Quick Onboarding

For the standardized "new model in 5 minutes" path (adapter template, registration steps, must-run commands, acceptance thresholds), use:
- `docs/docs/NEW_MODEL_ONBOARDING.md`

This includes explicit instructions for:
- adding new text models and validating strict parity,
- adding new VLM models (LM trunk assumption) and validating smoke diagnostics.
