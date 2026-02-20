# Compatibility Matrix

This page fixes the compatibility contract for `meap` v1 PR4.

## Installation Modes

- Development (recommended for contributors):
  - `pip install -e ".[dev,multimodal,viz,docs]"`
- Release install (post `v1.0.0`):
  - `pip install meap`

## Runtime Compatibility

From package metadata (`pyproject.toml`):

- Python: `>=3.10,<3.13`
- `torch`: `>=2.1`
- `transformers`: `>=4.49`
- `transformer-lens`: `>=2.11`

CI baseline:
- Python `3.10`
- CPU tiny smoke matrix

## CI Layering

Always-on CI (`.github/workflows/ci.yml`):
- `ruff` + `mypy` + `pytest`
- package build + `twine check`
- tiny smoke (`hf-internal-testing/tiny-random-gpt2`)

Manual/regression CI (`.github/workflows/regression-matrix.yml`):
- full stage matrix (`scripts/test_stage_matrix.py`)
- text parity + multimodal smoke
- JSON artifacts uploaded for reproducible regression tracking

## Text Parity Matrix (Vendor/TLens/HF)

Primary parity models (required):
- `gpt2` (aka `gpt2-small` in TLens naming)
- `Qwen/Qwen2-0.5B`

Extended text parity targets:
- `facebook/opt-125m`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (environment-dependent)

Methods:
- `EAP`
- `EAP-IG-inputs`
- `clean-corrupted`
- `EAP-IG-activations`
- `exact` (allowed `skip` by edge threshold policy)

Strict thresholds:
- `max_abs <= 1e-3`
- `cosine >= 0.995`
- `top-k overlap >= 0.90` (`k=200`)

## Multimodal Smoke Matrix

Current smoke targets:
- `Qwen/Qwen2-VL-2B`
- `llava-hf/llava-1.5-7b-hf`
- `HuggingFaceTB/SmolVLM-Instruct`
- `Qwen/Qwen2-Audio-7B` (fallback supported: `fixie-ai/ultravox-v0_5-llama-3_2-1b`)

Smoke acceptance:
- `status == "pass"`
- graph stats present (`n_forward`, `n_backward`, `n_edges`)
- failures provide structured diagnostics (`error_type`, `error_message`, `resolution_error_hint`)

## Known Limits (Release Notes Mirror)

- Multimodal attribution targets language-model trunk only.
- Some architecture-specific intervention paths remain staged and adapter-dependent.
- `exact` is compute-heavy and commonly run with edge-threshold skip policy.
