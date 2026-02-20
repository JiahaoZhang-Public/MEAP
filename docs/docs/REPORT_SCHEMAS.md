# Report Schemas (Frozen)

The following script outputs are schema-versioned and treated as stable integration artifacts.

## 1) `smoke_hf_matrix.py`

Top-level keys:
- `report_type` = `"smoke_hf_matrix"`
- `schema_version` = `"1.0.0"`
- `generated_at_utc`
- `config`
- `all_passed`
- `results`

Per-result keys:
- `model_id`, `modality`, `adapter_name`, `backbone_path`, `arch_kind`
- `status`, `seconds`
- `error_type`, `error_message`, `resolution_error_hint`
- `graph_stats` (`null` on failure)

## 2) `test_text_vendor_parity.py`

Top-level keys:
- `report_type` = `"text_vendor_parity"`
- `schema_version` = `"1.0.0"`
- `generated_at_utc`
- `config`
- `by_adapter`
- `all_passed`
- `results`

Per-result keys:
- `model_tlens`, `model_hf`, `adapter_name`
- `method`, `status`, `seconds`, `note`, `skip_reason`
- `vendor_vs_ours_tlens`
- `vendor_vs_ours_hf`
- `ours_tlens_vs_ours_hf`

## 3) `test_stage_matrix.py`

Top-level keys:
- `report_type` = `"stage_matrix"`
- `schema_version` = `"1.0.0"`
- `generated_at_utc`
- `config`
- `checks`
- `model_level`
- `method_level`
- `error_summary`
- `all_passed`

These schemas are consumed by tests and should only change with an explicit schema version bump.
