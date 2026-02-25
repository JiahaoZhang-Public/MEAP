# Scripts Guide

This directory is split into:

- active scripts (top-level `scripts/`)

## Active Scripts

1. `scripts/api_minimal_examples.py`
- Purpose: runnable minimal examples for text/image/audio attribution with a unified CLI.
- Typical use:
```bash
python scripts/api_minimal_examples.py --example text-gpt2 --method smoke --device cpu --dtype float32
```

2. `scripts/inspect_adapter_registry.py`
- Purpose: inspect adapter matching/diagnostics for one HF model.
- Typical use:
```bash
python scripts/inspect_adapter_registry.py --model-id Qwen/Qwen2-0.5B --output reports/inspect_qwen2.json
```

3. `scripts/smoke_hf_matrix.py`
- Purpose: model-level smoke matrix for text and multimodal HF models.
- Output schema: `report_type=smoke_hf_matrix`, `schema_version=1.0.0`.
- Typical use:
```bash
python scripts/smoke_hf_matrix.py --text-models gpt2 --multimodal-models Qwen/Qwen2-VL-2B --device cpu --dtype float32 --output reports/smoke.json
```

4. `scripts/test_text_vendor_parity.py`
- Purpose: method-level text parity (`vendor` vs `TLens` vs `HF`) for EAP-family methods.
- Output schema: `report_type=text_vendor_parity`, `schema_version=1.0.0`.

5. `scripts/test_stage_matrix.py`
- Purpose: unified staged matrix wrapper (quality gates + text parity + multimodal smoke).
- Output schema: `report_type=stage_matrix`, `schema_version=1.0.0`.

6. `scripts/release/release.py`
- Purpose: release engineering utility (gate/build/check/notes/tag).
- See `scripts/release/README.md` for command usage.

7. `scripts/real_attribution_modalities.py`
- Purpose: run real attribution experiments for `text`, `image`, and `audio` and export graph artifacts.
- Exports per modality:
  - `<modality>_graph_full.json`
  - `<modality>_graph_topn.json`
  - `<modality>_graph_topn.png`
  - `<modality>_summary.json`
- Exports unified report:
  - `real_attribution_modalities.json`
- Typical use:
```bash
python scripts/real_attribution_modalities.py \
  --modalities text,image,audio \
  --method EAP \
  --device cpu \
  --dtype float32 \
  --topn 200 \
  --output-dir reports/real_attribution
```

8. `scripts/route_graph_matrix.py`
- Purpose: verify `model -> language trunk -> graph/hook` route resolution across text and multimodal model sets.
- Optional: add `--run-attribute-smoke` to run one minimal `PreparedBatch` attribution smoke (`method=smoke`) per model.
- Output schema: `report_type=route_graph_matrix`, `schema_version=1.0.0`.
- Defaults: `tier=core` model set from `meap.catalog`.
- Typical use:
```bash
python scripts/route_graph_matrix.py \
  --text-models gpt2,facebook/opt-125m,Qwen/Qwen2-0.5B,tiiuae/falcon-rw-1b \
  --multimodal-models Qwen/Qwen2-VL-2B,HuggingFaceTB/SmolVLM-Instruct,fixie-ai/ultravox-v0_5-llama-3_2-1b \
  --run-attribute-smoke \
  --device cpu \
  --dtype float32 \
  --output reports/route_graph_matrix.json
```

9. `scripts/download_hf_models.py`
- Purpose: one-click download of Hugging Face model snapshots.
- Default preset downloads all known models (official + route matrix + v1.5 candidates).
- Output schema: `report_type=hf_model_download`, `schema_version=1.0.0`.
- Typical use:
```bash
python scripts/download_hf_models.py --preset all-known --output reports/hf_model_download.json
```
- Dry-run (list only, no download):
```bash
python scripts/download_hf_models.py --preset all-known --dry-run
```

For per-model, per-modality walkthrough scripts, use:
- `examples/text/attribution_model_prepared.py`
- `examples/text/gpt2.py`
- `examples/image/Qwen2-VL-2B.py`
- `examples/audio/ultravox.py`

Use active scripts for current CI/reproducible runs.
