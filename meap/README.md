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
- `processor`
- `pair_batch_preparer`

## Legacy Compatibility Lane

The following high-level wrappers remain available for migration but are no longer primary:
- `discover_circuit(...)`
- `attribute_from_dataloader(...)`
- `evaluate_graph_from_dataloader(...)`
- `evaluate_baseline_from_dataloader(...)`

## Module Map

- `attribution_model.py`: primary object API
- `api.py`: stable high-level surface + legacy wrappers
- `batch.py`: `PreparedBatch` contract/validation
- `graph.py`: graph structure
- `backend/`: HF/TLens runtime backends and adapter registry
- `attribute.py`, `evaluate.py`: low-level execution internals

## Examples

- Minimal prepared-input example:
  - `/Users/jiahaozhang/Repo/project/multimodal-lm-eap-ig/examples/text/attribution_model_prepared.py`
- Per-modality walkthroughs:
  - `/Users/jiahaozhang/Repo/project/multimodal-lm-eap-ig/examples/text/gpt2.py`
  - `/Users/jiahaozhang/Repo/project/multimodal-lm-eap-ig/examples/image/Qwen2-VL-2B.py`
  - `/Users/jiahaozhang/Repo/project/multimodal-lm-eap-ig/examples/audio/ultravox.py`

## Known Limits

- Multimodal attribution is language-trunk-centric.
- `ComponentSpec` is currently a contract placeholder (filtering deferred).
