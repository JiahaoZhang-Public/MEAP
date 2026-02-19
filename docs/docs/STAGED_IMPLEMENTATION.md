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
- Added support for either:
  - `processor=...` (auto-wrap into `HFProcessorAdapter`), or
  - `pair_batch_preparer=...`.

### Acceptance
- Same dataloader interface can carry text-only, multimodal raw pairs, or pre-built `PreparedBatch`.
- Existing low-level APIs remain available.

## PR4: Phase 5 + 6 (Test Matrix + Docs)

### Scope
- Add staged validation matrix and document the rollout.

### Implemented
- Added script:
  - `scripts/test_stage_matrix.py`
- Added tests:
  - `tests/test_input_layer.py`
  - `tests/test_backend_multimodal_passthrough.py`
  - `tests/test_api_high_level.py`
- Added this documentation file.

### Acceptance
- `ruff` + `pytest` pass.
- Text backend matrix script remains runnable.
- Optional Qwen2-VL smoke can be invoked via matrix script.

## Recommended Validation Commands

```bash
ruff check multimodal_lm_eap_ig tests scripts
pytest -q
python scripts/test_backend_unified.py --levels model,method --models gpt2-small,Qwen/Qwen2-0.5B --methods smoke,EAP-IG-inputs
python scripts/test_stage_matrix.py --run-qwen2-vl-smoke --device cuda --dtype bfloat16
```
