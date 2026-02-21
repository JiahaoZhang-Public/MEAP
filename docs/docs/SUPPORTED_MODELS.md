# Supported Models (V2 Core Set)

This page tracks the official Hugging Face support set for `HFLLMBackend`.

Source of truth:
- runtime catalog: `meap.catalog`
- public query APIs:
  - `list_supported_architectures()`
  - `list_official_models()`

## Scope

- New architecture onboarding is smoke-first.
- Text models can be parity-validated with vendor/TLens workflows.
- Multimodal attribution remains language-trunk only in this stage.

## Official Architecture Routes

| Adapter Name | Arch Kind | Modalities | Tier |
| --- | --- | --- | --- |
| `gpt2_like` | `gpt2_like` | text | core |
| `opt_like` | `opt_like` | text | core |
| `llama_like` | `llama_like` | text,multimodal | core |

## Official Model IDs (Core)

| Model ID | Adapter | Modality | Tier |
| --- | --- | --- | --- |
| `gpt2` | `gpt2_like` | text | core |
| `distilgpt2` | `gpt2_like` | text | core |
| `facebook/opt-125m` | `opt_like` | text | core |
| `Qwen/Qwen2-0.5B` | `llama_like` | text | core |
| `Qwen/Qwen2-VL-2B` | `llama_like` | multimodal | core |
| `llava-hf/llava-1.5-7b-hf` | `llama_like` | multimodal | core |

## Explicit Adapter/Trunk Selection

V2 supports:
- `adapter_name`
- `language_trunk_path`

Meaning:
- `language_trunk_path`: path used to locate the language-model `nn.Module` inside your HF model.
- `adapter_name`: structure adapter used after trunk selection to interpret that module stack.

This enables local finetuned models to reuse official architecture routes:

```python
from meap import AttributionModel, TaskSpec

am = AttributionModel.from_pretrained(
    "/path/to/local/model",
    adapter_name="llama_like",
    language_trunk_path="model.language_model.model",
)

result = am.attribute(
    batches=[prepared_batch],
    task=TaskSpec(task="next_token", labels=[1]),
    method="smoke",
)
```

## Standard Commands

### 1) Unified smoke matrix (defaults use catalog core set)

```bash
python scripts/smoke_hf_matrix.py \
  --device cpu \
  --dtype float32 \
  --output reports/smoke_hf_matrix.json
```

### 2) Stage matrix wrapper (lint + unit tests + matrix scripts)

```bash
python scripts/test_stage_matrix.py \
  --device cpu \
  --dtype float32 \
  --output reports/stage_matrix.json
```

### 3) Adapter diagnostics for a specific model

```bash
python scripts/inspect_adapter_registry.py \
  --model-id Qwen/Qwen2-0.5B \
  --output reports/adapter_registry_qwen2.json
```

### 4) Route + graph/hook matrix for text and multimodal models

```bash
python scripts/route_graph_matrix.py \
  --text-models gpt2,facebook/opt-125m,Qwen/Qwen2-0.5B \
  --multimodal-models Qwen/Qwen2-VL-2B,llava-hf/llava-1.5-7b-hf,fixie-ai/ultravox-v0_5-llama-3_2-1b \
  --run-attribute-smoke \
  --device cpu \
  --dtype float32 \
  --output reports/route_graph_matrix.json
```

## Common Failures and Fixes

1. `Unsupported HF architecture for HFLLMBackend`
- Cause: selected trunk/adapter does not match model modules.
- Fix: inspect diagnostics (`candidate_backbones`, `adapter_attempts`, `selection_error`) and adjust `adapter_name` / `language_trunk_path`.

2. `Image features and image tokens do not match`
- Cause: processor prompt template or modality placeholders not aligned.
- Fix: keep clean/corrupt modality layout identical and use processor-specific prompt construction.

3. OOM on multimodal models
- Cause: model size or dtype/device mismatch.
- Fix: prefer GPU + `bfloat16`/`float16`, reduce batch to 1, start with smoke-only runs.
