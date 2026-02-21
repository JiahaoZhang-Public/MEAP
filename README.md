# meap

`meap` (Multimodal Edge Attribution Patching) is a Python package for running the EAP-family attribution methods on Hugging Face language-model backbones, including multimodal models via language-trunk attribution.

## Upstream Reference

This repository's initial code scaffold was directly copied from the original EAP-IG repository and then refactored/extended for multimodal and Hugging Face backend support:

- [hannamw/EAP-IG](https://github.com/hannamw/EAP-IG)

Import path:

```python
import meap
```

PyPI package name:

```bash
pip install meap
```

## Quick Start (10 Minutes)

### 1) Install

```bash
conda create -n meap-ig python=3.10 -y
conda activate meap-ig
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

Stable high-level API lives in `meap.api`.

- `AttributionModel` (primary)
- `AttributionResult`
- `RouteInfo`
- `PreparedInputLike`
- `attribute_from_dataloader`
- `evaluate_graph_from_dataloader`
- `evaluate_baseline_from_dataloader`
- `AttributionRunResult`

API stability policy and deprecations:
- `docs/docs/API_STABILITY.md`

## Primary Entrypoint (Prepared Inputs)

Core v2 usage is object-oriented and prepared-input-first:

```python
from meap import AttributionModel, PreparedBatch

model = AttributionModel.from_pretrained("openai-community/gpt2")
result = model.attribute(
    batches=[prepared_batch],  # PreparedBatch or nested prepared mapping
    metric=metric_fn,
    method="EAP",
)
```

## Compatibility Entrypoints (Deprecated)

The following APIs are retained for migration and emit `DeprecationWarning`:
- `discover_circuit(...)`
- `attribute_from_dataloader(...)`

### Compatibility A: `PreparedBatch` (fully user-prepared inputs)

```python
from meap import HFLLMBackend, attribute_from_dataloader

backend = HFLLMBackend(model, tokenizer=tokenizer)
result = attribute_from_dataloader(
    model=model,
    backend=backend,
    dataloader=[prepared_batch],  # PreparedBatch
    metric=metric_fn,
    method="EAP",
)
```

### Compatibility B: raw clean/corrupt + `pair_batch_preparer`

```python
from meap import HFLLMBackend, HFProcessorAdapter, RawPairBatch, attribute_from_dataloader

backend = HFLLMBackend(model, tokenizer=getattr(processor, "tokenizer", None))
pair_batch_preparer = HFProcessorAdapter(processor=processor, device=backend.config.device)
result = attribute_from_dataloader(
    model=model,
    backend=backend,
    dataloader=[
        RawPairBatch(clean=clean_samples, corrupt=corrupt_samples, labels=labels)
    ],
    pair_batch_preparer=pair_batch_preparer,
    metric=metric_fn,
    method="smoke",
)
```

Notes:
- V2 高层 API 仅接受 `PreparedBatch` 或 `RawPairBatch`。
- High-level API does not do implicit truncation.

## HF Adapter Selection (V2)

Primary lane:
- `AttributionModel.from_pretrained(...)` handles model loading + HF route selection.

Compatibility lane (deprecated):
- `discover_circuit(...)`
- `attribute_from_dataloader(...)`

Route terms:
- `language_trunk_path`: 目的是找到你 Hugging Face model (`nn.Module`) 中语言模型部分的路径。
- `adapter_name`: 识别该 language trunk 内部结构的适配器名称。

Lane A supports explicit adapter/trunk selection:

```python
from meap.api import discover_circuit

result = discover_circuit(
    model_id_or_path="Qwen/Qwen2-VL-2B",
    adapter_name="llama_like",
    language_trunk_path="model.language_model.model",
    clean_samples={"text": ["Describe the image"]},
    corrupt_samples={"text": ["Transcribe the text"]},
    task="next_token",
    labels=[42],
)
```

默认情况下（`adapter_name=None`, `language_trunk_path=None`）会自动解析。

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
  --method smoke \
  --device cpu \
  --dtype float32
```

Optional overrides for the default audio smoke input:
- `--audio-path /path/to/audio.wav`
- `--audio-url https://.../your_audio.mp3` (used when `--audio-path` is empty)
- `--audio-prompt "Generate the caption in English:"`

## Real Attribution + Graph Visualization

Run one command to execute real attribution experiments for text/image/audio and export graph
artifacts (`full.json`, `topn.json`, `topn.png`) per modality:

```bash
python scripts/real_attribution_modalities.py \
  --modalities text,image,audio \
  --method EAP \
  --device cpu \
  --dtype float32 \
  --topn 200 \
  --output-dir reports/real_attribution
```

## Per-Modality Example Scripts

Detailed per-model examples are available under `examples/`:

- `examples/text/gpt2.py`
- `examples/image/Qwen2-VL-2B.py`
- `examples/audio/ultravox.py`

Each script explicitly demonstrates:
1. raw data -> process -> model inputs
2. attribution
3. graph visualization export

See `examples/README.md` for usage and outputs.

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

## Release Engineering (v1.1.0)

Changelog:
- `CHANGELOG.md`

Scripted release flow:
- `scripts/release/release.py`
- `scripts/release/README.md`
- `docs/docs/RELEASE_PROCESS.md`

Typical release flow:

```bash
python scripts/release/release.py set-version --version 1.1.0
python scripts/release/release.py gate --clean-dist
python scripts/release/release.py notes --version v1.1.0
python scripts/release/release.py tag --version v1.1.0 --push
```

For RC releases, use tags like `v1.1.0-rc1`.

## Docs Map

Active docs:
- Package guide: `meap/README.md`
- API stability: `docs/docs/API_STABILITY.md`
- Compatibility matrix: `docs/docs/COMPATIBILITY_MATRIX.md`
- Supported models: `docs/docs/SUPPORTED_MODELS.md`
- Report schemas: `docs/docs/REPORT_SCHEMAS.md`
- Release process: `docs/docs/RELEASE_PROCESS.md`
- New model onboarding (5 min): `docs/docs/NEW_MODEL_ONBOARDING.md`
- Scripts guide: `scripts/README.md`

## Development

```bash
ruff check meap tests scripts
pytest -q
mypy
```
