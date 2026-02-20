# Scripts Guide

This directory is split into:

- active scripts (top-level `scripts/`)
- archived/legacy scripts (`scripts/cache/`)

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

## Archived Scripts

Archived scripts are kept for historical/debug reference and are not part of the current recommended workflow:

- `scripts/cache/e2e_llava_attribution.py`
- `scripts/cache/e2e_qwen2_vl_smoke.py`
- `scripts/cache/backend_unified_legacy.py`

Use active scripts for current CI/reproducible runs.
