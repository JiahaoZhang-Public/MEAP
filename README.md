# mm-eap

`mm-eap` (Multimodal Edge Attribution Patching) is a Python package for running the EAP-family attribution methods on Hugging Face language-model backbones, including multimodal models via language-trunk attribution.

Import path:

```python
import multimodal_lm_eap_ig
```

PyPI package name:

```bash
pip install mm-eap
```

## Quick Start (10 Minutes)

### 1) Install

```bash
conda create -n mm-eap-ig python=3.10 -y
conda activate mm-eap-ig
pip install -e ".[dev,multimodal,viz,docs]"
```

### 2) Run one smoke example (text)

```bash
python scripts/api_minimal_examples.py \
  --example text-gpt2 \
  --method smoke \
  --device cpu \
  --dtype float32
```

### 3) Optional: save a JSON report

```bash
python scripts/api_minimal_examples.py \
  --example text-gpt2 \
  --method smoke \
  --device cpu \
  --dtype float32 \
  --output reports/example_text_gpt2.json
```

## Stable API

Stable high-level API lives in `multimodal_lm_eap_ig.api`.

- `attribute_from_dataloader`
- `evaluate_graph_from_dataloader`
- `evaluate_baseline_from_dataloader`
- `AttributionRunResult`

API stability policy and deprecations:
- `docs/docs/API_STABILITY.md`

## Two Supported Input Entrypoints

### Entrypoint A: `PreparedBatch` (fully user-prepared inputs)

```python
from multimodal_lm_eap_ig import HFLLMBackend, attribute_from_dataloader

backend = HFLLMBackend(model, tokenizer=tokenizer)
result = attribute_from_dataloader(
    model=model,
    backend=backend,
    dataloader=[prepared_batch],  # PreparedBatch
    metric=metric_fn,
    method="EAP",
)
```

### Entrypoint B: raw clean/corrupt + `processor` or `pair_batch_preparer`

```python
from multimodal_lm_eap_ig import HFLLMBackend, attribute_from_dataloader

backend = HFLLMBackend(model, tokenizer=getattr(processor, "tokenizer", None))
result = attribute_from_dataloader(
    model=model,
    backend=backend,
    dataloader=[{"clean": clean_samples, "corrupt": corrupt_samples, "labels": labels}],
    processor=processor,  # or pair_batch_preparer=...
    metric=metric_fn,
    method="smoke",
)
```

Notes:
- `processor` and `pair_batch_preparer` are mutually exclusive.
- High-level API does not do implicit truncation.

## Minimal Executable Examples

All examples share unified arguments:
- `--example`
- `--method`
- `--device`
- `--dtype`
- `--hf-token` (optional)
- `--output` (optional)

### Text: GPT-2 (`PreparedBatch` path)

```bash
python scripts/api_minimal_examples.py \
  --example text-gpt2 \
  --method smoke \
  --device cpu \
  --dtype float32
```

### Text: Qwen2-0.5B (`PreparedBatch` path)

```bash
python scripts/api_minimal_examples.py \
  --example text-qwen2 \
  --method smoke \
  --device cpu \
  --dtype float32
```

### Image-text: Qwen2-VL-2B (`processor` path)

```bash
python scripts/api_minimal_examples.py \
  --example image-qwen2vl \
  --method smoke \
  --device cpu \
  --dtype float32
```

### Audio: Ultravox (`pair_batch_preparer` path)

```bash
python scripts/api_minimal_examples.py \
  --example audio-ultravox \
  --audio-path /path/to/audio.wav \
  --method smoke \
  --device cpu \
  --dtype float32
```

## Supported Methods

- `smoke`
- `EAP`
- `EAP-IG-inputs`
- `clean-corrupted`
- `EAP-IG-activations`
- `exact`

## Current Limits

- Multimodal attribution currently targets language-model trunk only.
- Some architectures still require adapter extension for full method parity.
- `exact` can be expensive; parity scripts support skip-by-edge-threshold policy.

## Docs Map

Active docs:
- Package guide: `multimodal_lm_eap_ig/README.md`
- API stability: `docs/docs/API_STABILITY.md`
- Compatibility matrix: `docs/docs/COMPATIBILITY_MATRIX.md`
- Supported models: `docs/docs/SUPPORTED_MODELS.md`
- Report schemas: `docs/docs/REPORT_SCHEMAS.md`
- New model onboarding (5 min): `docs/docs/NEW_MODEL_ONBOARDING.md`
- Scripts guide: `scripts/README.md`

Archived historical docs (cache):
- Staged implementation history: `docs/cache/STAGED_IMPLEMENTATION.md`
- Early refactor design context: `docs/cache/REFACTOR_DESIGN.md`

## Development

```bash
ruff check multimodal_lm_eap_ig tests scripts
pytest -q
mypy
```
