# meap

`meap` (Multimodal Edge Attribution Patching) runs EAP-family attribution on Hugging Face language-model trunks, including multimodal models through language-trunk routing.

## Install

Runtime:

```bash
pip install meap
```

TestPyPI (optional):

```bash
pip install -i https://test.pypi.org/simple/ meap
```

Development:

```bash
pip install -e ".[dev,multimodal,viz,docs]"
```

## Primary API (v1.4.0)

Use `AttributionModel` as the default user entrypoint.

Core contract:
1. `model -> language trunk/graph`
2. `prepared inputs -> attribution`

```python
from meap import AttributionModel

am = AttributionModel.from_pretrained("gpt2")
result = am.attribute(
    batches=[prepared_batch],
    metric=metric_fn,
    method="EAP",
)
```

## Input Contract

Core API accepts:
- `PreparedBatch`
- nested prepared tensor mappings with:
  - `clean_inputs`
  - `corrupt_inputs`
  - optional `labels`, `input_lengths`, `meta`

Core API does not accept:
- raw `clean_samples` / `corrupt_samples`
- `pair_batch_preparer`

## Dataloader Evaluation APIs

The following evaluation wrappers remain available for dataloader-based workflows:
- `evaluate_graph_from_dataloader(...)`
- `evaluate_baseline_from_dataloader(...)`

## Quick Commands

Examples guide (recommended starting point):

```bash
cat examples/README.md
```

Minimal prepared-input example:

```bash
python examples/text/attribution_model_prepared.py --device cpu --dtype float32 --method smoke
```

Per-modality walkthroughs:

```bash
python examples/text/gpt2.py --device cpu --dtype float32 --method EAP
python examples/image/Qwen2-VL-2B.py --device cpu --dtype float32 --method EAP
python examples/audio/ultravox.py --device cpu --dtype float32 --method EAP
```

Model route + graph/hook matrix:

```bash
python scripts/route_graph_matrix.py \
  --text-models gpt2,facebook/opt-125m,Qwen/Qwen2-0.5B \
  --multimodal-models Qwen/Qwen2-VL-2B,llava-hf/llava-1.5-7b-hf,fixie-ai/ultravox-v0_5-llama-3_2-1b \
  --run-attribute-smoke \
  --device cpu \
  --dtype float32
```

Release gate:

```bash
python scripts/release/release.py gate --clean-dist
```

## Documentation

- `meap/README.md`
- `docs/docs/API_STABILITY.md`
- `docs/docs/SUPPORTED_MODELS.md`
- `docs/docs/ATTRIBUTION_MODEL_API_DESIGN.md`
- `scripts/README.md`

## Versioning

- Current release baseline: `1.4.0`
- Historical migration notes: `CHANGELOG.md` and `releases/v1.4.0.md`

## Upstream Reference

Initial scaffold reference:
- [hannamw/EAP-IG](https://github.com/hannamw/EAP-IG)
