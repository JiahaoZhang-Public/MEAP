# API Stability (v2, AttributionModel MVP)

This page defines the supported package surface for `meap` v2.

## Stable Public API

Primary module:
- `meap.api`

Primary stable entrypoint:
- `AttributionModel`

Stable symbols:
- `AttributionModel`
- `AttributionResult`
- `RouteInfo`
- `PreparedInputLike`
- `AttributionRunResult`
- `TaskSpec`
- `CircuitEdgeSummary`
- `CircuitRunResult`
- `evaluate_graph_from_dataloader`
- `evaluate_baseline_from_dataloader`
- compatibility wrappers (deprecated):
  - `attribute_from_dataloader`
  - `discover_circuit`

Stable package-level objects:
- `HFLLMBackend`, `TLensBackend`
- `PreparedBatch`, `RawPairBatch`
- `HFProcessorAdapter`
- `Graph`
- `list_supported_architectures`, `list_official_models`
- `register_architecture_adapter`, `inspect_model_architecture`, `resolve_backend`

## V2 MVP Contract

Core path:
1. `model -> language trunk/graph`
2. `prepared inputs -> attribution`

Core `AttributionModel` APIs accept:
- `PreparedBatch`, or
- nested prepared mapping:
  - `clean_inputs`
  - `corrupt_inputs`
  - optional `labels` / `input_lengths` / `meta`

Core `AttributionModel` APIs do not accept:
- `clean_samples`
- `corrupt_samples`
- `processor`
- `pair_batch_preparer`

## Compatibility Lane (Deprecated)

1. High-level dataloader input is now strict:
- accepted: `PreparedBatch`, `RawPairBatch`
- removed: dict/tuple dataloader entries and `DictPairBatch`

2. High-level API no longer accepts direct `processor=...`:
- use `pair_batch_preparer=...`
- recommended helper: `HFProcessorAdapter(processor=...)`

Deprecated compatibility wrappers:
- `discover_circuit(...)`
- `attribute_from_dataloader(...)`

These wrappers preserve behavior for migration and emit `DeprecationWarning`.

## HF Resolution Contract

Definitions:
- `language_trunk_path`: path used to locate the language-model `nn.Module` inside a Hugging Face model.
- `adapter_name`: architecture adapter used to interpret structure inside the selected language trunk.

Resolution order:
1. if `language_trunk_path` is provided: only that trunk candidate is used (and `adapter_name`, if provided, is enforced)
2. else if `adapter_name` is provided: only that adapter is tried across candidate trunks
3. else: automatic adapter + trunk selection

Where route selection happens:
- `AttributionModel.from_pretrained(...)` (primary), or
- `discover_circuit(...)` (deprecated compatibility), or
- `HFLLMBackend(model, adapter_name=..., language_trunk_path=...)` initialization.

Diagnostics fields are stable:
- `selected`
- `candidate_backbones`
- `adapter_attempts`
- `selection_error`

## Internal Modules (Not Stability-Guaranteed)

The following modules are implementation detail APIs and may change in minor releases:

- `meap.attribute`
- `meap.evaluate`
- `meap.utils`

## Deprecation Policy

Deprecated top-level symbols emit `DeprecationWarning`.

Current deprecations:
- Deprecated since: `1.0.0`
- Planned top-level removal: `1.2.0`

Symbols:
- `get_real_edge_scores`
- `get_scores_clean_corrupted`
- `get_scores_eap`
- `get_scores_eap_ig`
- `get_scores_exact`
- `get_scores_ig_activations`
- `get_scores_smoke`
- `build_default_llava_processor`
- `prepare_llava_token_pair_batch`

Migration guidance:
- Import these from internal modules only if you need low-level control.
- Prefer high-level APIs in `meap.api` for forward compatibility.
