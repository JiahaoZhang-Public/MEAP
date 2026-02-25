# meap Package Guide

`meap` provides EAP-family attribution on Hugging Face language-model trunks (including multimodal models routed to their language trunk).

## Install

Runtime:

```bash
pip install meap
```

Development:

```bash
pip install -e ".[dev,multimodal,viz,docs]"
```

## Primary Public API

Use `AttributionModel` as the default user entrypoint:

- `AttributionModel`
- `AttributionResult`
- `RouteInfo`
- `PreparedInputLike`
- `PreparedBatch`
- `HFLLMBackend`
- `Graph`

Canonical flow:
1. `model -> language trunk/graph`
2. `prepared inputs -> attribution`

```python
from meap import AttributionModel

am = AttributionModel.from_pretrained("gpt2")
result = am.attribute(batches=[prepared_batch], metric=metric_fn, method="EAP")
```

## Input Contract

Core API accepts:
- `PreparedBatch`
- nested prepared mapping with:
  - `clean_inputs`
  - `corrupt_inputs`
  - optional `labels`, `input_lengths`, `meta`

Core API does not accept:
- raw `clean_samples` / `corrupt_samples`
- `pair_batch_preparer`

## Dataloader Evaluation APIs

The following dataloader-based evaluation wrappers remain available:
- `evaluate_graph_from_dataloader(...)`
- `evaluate_baseline_from_dataloader(...)`

Historical migration notes are tracked in:
- `CHANGELOG.md`
- `releases/v1.4.0.md`
- `releases/v1.5.0.md`

## Module Map

- `attribution_model.py`: primary object API
- `api.py`: stable high-level surface + dataloader-based evaluation wrappers
- `batch.py`: `PreparedBatch` contract/validation
- `graph.py`: graph structure
- `backend/`: HF/TLens runtime backends and adapter registry
- `attribute.py`, `evaluate.py`: low-level execution internals

## Examples

- Examples guide:
  - `examples/README.md`
- Minimal prepared-input smoke:
  - `python examples/text/attribution_model_prepared.py --device cpu --dtype float32 --method smoke`
- Per-modality walkthroughs:
  - `python examples/text/gpt2.py --device cpu --dtype float32 --method EAP`
  - `python examples/image/Qwen2-VL-2B.py --device cpu --dtype float32 --method EAP`
  - `python examples/audio/ultravox.py --device cpu --dtype float32 --method EAP`

## Known Limits

- Multimodal attribution is language-trunk-centric.
- `ComponentSpec` is currently a contract placeholder (filtering deferred).

## Versioning

- Current release baseline: `1.5.0`
