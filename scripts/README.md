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

8. `scripts/discover_circuit_minimal.py`
- Purpose: one-shot `discover_circuit` demo with HF model id or local path.
- Typical use:
```bash
python scripts/discover_circuit_minimal.py --model-ref openai-community/gpt2 --task next_token --method EAP
```

For per-model, per-modality walkthrough scripts, use:
- `examples/text/gpt2.py`
- `examples/image/Qwen2-VL-2B.py`
- `examples/audio/ultravox.py`

Use active scripts for current CI/reproducible runs.
