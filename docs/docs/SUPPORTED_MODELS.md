# Supported Models (Smoke-First)

This page tracks the current Hugging Face model support under the `HFLLMBackend` path.

Quick onboarding guide:
- `docs/docs/NEW_MODEL_ONBOARDING.md`
- `docs/docs/COMPATIBILITY_MATRIX.md`
- `docs/docs/REPORT_SCHEMAS.md`

## Scope

- New architecture onboarding is smoke-first.
- Text Tier A parity target remains `HF ~= TLens ~= vendor` under strict thresholds.
- VLM attribution remains language trunk only in this stage.

## Text Models

| Model | Architecture Route | Smoke Status | Suggested Device |
| --- | --- | --- | --- |
| `gpt2` | GPT2-like adapter | Pass | CPU/GPU |
| `distilgpt2` | GPT2-like adapter | Pass | CPU/GPU |
| `facebook/opt-125m` | OPT-like adapter | Pass | CPU/GPU |
| `Qwen/Qwen2-0.5B` | LLaMA-like adapter (GQA ungroup) | Pass | GPU recommended |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | LLaMA-like adapter | Expected* | GPU recommended |
| `tiiuae/falcon-rw-1b` | Falcon-like adapter | Expected* | GPU recommended |
| `mosaicml/mpt-1b-redpajama-200b` | MPT-like adapter | Expected* | GPU recommended |
| `google/gemma-2b` | LLaMA-like adapter | Expected* | GPU recommended |

`Expected*`: compatible by structure and backend contract; validate in your environment via `scripts/smoke_hf_matrix.py`.

## Multimodal Models

| Model | Language Trunk Route | Smoke Status | Suggested Device |
| --- | --- | --- | --- |
| `Qwen/Qwen2-VL-2B` | `language_model.layers` | Pass | GPU recommended (CPU functional) |
| `llava-hf/llava-1.5-7b-hf` | `model.layers` + image projector path | Expected* | GPU required |
| `HuggingFaceTB/SmolVLM-Instruct` | VLM wrapper + decoder trunk | Expected* | GPU recommended |

If `HuggingFaceTB/SmolVLM-Instruct` is not compatible with your installed `transformers` version, use `Qwen/Qwen2-VL-2B-Instruct` as fallback and record the substitution in your run report.

## Standard Commands

### 1) Unified smoke matrix

```bash
python scripts/smoke_hf_matrix.py \
  --text-models gpt2,distilgpt2,facebook/opt-125m,Qwen/Qwen2-0.5B,TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --multimodal-models Qwen/Qwen2-VL-2B,llava-hf/llava-1.5-7b-hf,HuggingFaceTB/SmolVLM-Instruct \
  --device cpu \
  --dtype bfloat16 \
  --output reports/smoke_hf_matrix.json
```

### 2) Stage matrix wrapper (lint + unit tests + smoke matrix)

```bash
python scripts/test_stage_matrix.py \
  --text-models gpt2,distilgpt2,facebook/opt-125m \
  --multimodal-models Qwen/Qwen2-VL-2B \
  --device cpu \
  --dtype bfloat16 \
  --output reports/stage_matrix.json
```

### 3) Adapter diagnostics for a specific model

```bash
python scripts/inspect_adapter_registry.py \
  --model-id Qwen/Qwen2-0.5B \
  --output reports/adapter_registry_qwen2.json
```

## Add New Adapter (5 Minutes)

1. Copy template: `meap/backend/adapters/_template.py`.
2. Implement `match`, layer accessors, attention projection mapping, and `projection_spec`.
3. Register adapter in `meap/backend/registry.py` default adapter list (or call `register_architecture_adapter`).
4. Add tests:
- adapter registry selection test (`tests/test_adapter_registry.py`)
- backend smoke test for the new architecture (`tests/test_backend_<arch>.py`)
5. Run:

```bash
ruff check meap tests scripts
pytest -q
python scripts/smoke_hf_matrix.py --text-models <new-model-id> --multimodal-models \"\" --device cpu --dtype float32
```

## Common Failures and Fixes

1. `Image features and image tokens do not match`
- Cause: processor prompt template or modality placeholders not aligned.
- Fix: use processor-specific chat template path, keep clean/corrupt modality layout identical.

2. `Unrecognized configuration class ... for AutoModelForCausalLM`
- Cause: VLM loaded with text-only auto class.
- Fix: use `AutoModelForImageTextToText` / `AutoModelForVision2Seq`.

3. `Unsupported HF architecture for HFLLMBackend`
- Cause: decoder backbone modules do not match registered adapters.
- Fix: add/extend architecture adapter and include missing module-path diagnostics.

4. OOM on multimodal models
- Cause: model size or dtype/device mismatch.
- Fix: prefer GPU + `bfloat16`/`float16`, reduce batch to 1, start with smoke-only runs.
