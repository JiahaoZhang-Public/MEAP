# Staged Implementation and Acceptance

This document records the staged rollout requested for backend migration and multimodal input unification.

## PR1: Phase 1 + 2 (Unified Input Layer + Processor Adapter)

### Scope
- Introduce a unified raw-pair input contract alongside existing `PreparedBatch`.
- Keep legacy text tuple support unchanged.
- Add processor-based preparation for non-text or multimodal samples.

### Implemented
- `multimodal_lm_eap_ig.batch`
  - Added `RawPairBatch` and dict/tuple raw-pair ingestion support in `iter_prepared_batches`.
  - Added `PairBatchPreparer` protocol for pluggable preprocessing.
- `multimodal_lm_eap_ig.preparer`
  - Added `HFProcessorAdapter`.
  - Added `prepare_pair_batch_with_processor`.
  - Kept `prepare_llava_token_pair_batch` as backward-compatible wrapper over the new adapter.

### Acceptance
- `iter_prepared_batches` accepts:
  - `PreparedBatch`
  - legacy `(clean_text, corrupt_text, labels)` tuples
  - raw pair mappings (`clean`, `corrupt`, `labels`) with `pair_batch_preparer`
- Validation invariants in `validate_prepared_batch` remain enforced.

## PR2: Phase 3 (Qwen2-VL HF Backend)

### Scope
- Ensure HF backend can execute multimodal forwards (extra model kwargs) while preserving hook contracts.
- Support decoder extraction from language-model wrappers used by multimodal models.

### Implemented
- `multimodal_lm_eap_ig.backend`
  - `HFLLMBackend.prepare_inputs` now forwards arbitrary tensor/non-tensor kwargs (for example `pixel_values`, `image_grid_thw`).
  - Added decoder backbone resolution for `model.language_model` wrappers.
- Added script:
  - `scripts/e2e_qwen2_vl_smoke.py`

### Acceptance
- HF backend can run with multimodal kwargs present.
- Hook scaffold and graph construction continue to work for decoder backbone.

## PR3: Phase 4 (High-Level API)

### Scope
- User-facing API where caller provides `metric` + `dataloader`; preprocessing route is resolved internally.

### Implemented
- Added `multimodal_lm_eap_ig.api`:
  - `attribute_from_dataloader`
  - `evaluate_graph_from_dataloader`
  - `evaluate_baseline_from_dataloader`
  - `AttributionRunResult`
- API uses user-managed preparation with explicit control:
  - provide `PreparedBatch` in dataloader, or
  - provide `processor=...` explicitly, or
  - provide `pair_batch_preparer=...` explicitly.
  - `processor` and `pair_batch_preparer` are mutually exclusive.
  - sequence truncation (`max_length`) is no longer applied by API; truncation must be done inside user preprocessing.

### Acceptance
- Same dataloader interface can carry text-only, multimodal raw pairs, or pre-built `PreparedBatch`.
- Existing low-level APIs remain available.

## PR4: Phase 5 + 6 (Test Matrix + Docs)

### Scope
- Add staged validation matrix and document the rollout.

### Implemented
- Added script:
  - `scripts/smoke_hf_matrix.py`
  - `scripts/test_stage_matrix.py` (wrapper now calls `smoke_hf_matrix.py`)
- Added tests:
  - `tests/test_input_layer.py`
  - `tests/test_backend_multimodal_passthrough.py`
  - `tests/test_api_high_level.py`
  - `tests/test_smoke_matrix_script.py`
- Added model support documentation:
  - `docs/docs/SUPPORTED_MODELS.md`
- Added this documentation file.

### Acceptance
- `ruff` + `pytest` pass.
- Text backend matrix script remains runnable.
- Optional Qwen2-VL smoke can be invoked via matrix script.

## Architecture Extension Update (Current)

### Implemented
- Backend split from monolithic file into package layout:
  - `multimodal_lm_eap_ig/backend/base.py`
  - `multimodal_lm_eap_ig/backend/registry.py`
  - `multimodal_lm_eap_ig/backend/hf_backend.py`
  - `multimodal_lm_eap_ig/backend/tlens_backend.py`
  - `multimodal_lm_eap_ig/backend/adapters/*`
- Added architecture adapters:
  - `llama_like`, `gpt2_like`, `opt_like`, `falcon_like`, `mpt_like`
- Added adapter registration/diagnostics APIs:
  - `register_architecture_adapter`
  - `inspect_model_architecture`
- Added template for new adapters:
  - `multimodal_lm_eap_ig/backend/adapters/_template.py`
- Added tests:
  - `tests/test_adapter_registry.py`
  - `tests/test_adapter_template_contract.py`
  - `tests/test_backend_falcon_like.py`
  - `tests/test_backend_mpt_like.py`
- Script updates:
  - `scripts/smoke_hf_matrix.py` now reports `adapter_name`
  - `scripts/test_text_vendor_parity.py` now reports top-k overlap and strict-mode threshold checks
  - new diagnostics script: `scripts/inspect_adapter_registry.py`

### Known Limits
- Falcon `new_decoder_architecture=True` interleaved fused QKV layout is diagnosed but not yet supported for intervention hooks.
- VLM path still attributes language trunk only in this stage.

## Recommended Validation Commands

```bash
ruff check multimodal_lm_eap_ig tests scripts
pytest -q
python scripts/smoke_hf_matrix.py --text-models gpt2,distilgpt2,facebook/opt-125m --multimodal-models Qwen/Qwen2-VL-2B --device cpu --dtype bfloat16
python scripts/test_stage_matrix.py --text-models gpt2 --multimodal-models Qwen/Qwen2-VL-2B --device cpu --dtype bfloat16
```

## PR5: New Model 5-Minute Onboarding

### Scope
- Freeze a repeatable onboarding contract for new text/VLM models.
- Consolidate adapter template usage, registration steps, required commands, and acceptance thresholds.

### Implemented
- Added onboarding guide:
  - `docs/docs/NEW_MODEL_ONBOARDING.md`
- Strengthened adapter template documentation:
  - `multimodal_lm_eap_ig/backend/adapters/_template.py`
- Linked onboarding entry from package docs:
  - `multimodal_lm_eap_ig/README.md`
  - `docs/docs/SUPPORTED_MODELS.md`

### Acceptance
- New contributors can add an adapter with a fixed checklist.
- Text-model onboarding has explicit strict parity thresholds.
- VLM onboarding has explicit smoke + diagnostics requirements.
