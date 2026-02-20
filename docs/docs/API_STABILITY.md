# API Stability (v1)

This page defines the supported package surface for `meap` v1.

## Stable Public API

Primary module:
- `meap.api`

Stable symbols:
- `AttributionRunResult`
- `attribute_from_dataloader`
- `evaluate_graph_from_dataloader`
- `evaluate_baseline_from_dataloader`

Stable package-level objects:
- `HFLLMBackend`, `TLensBackend`
- `PreparedBatch`, `RawPairBatch`, `DictPairBatch`
- `HFProcessorAdapter`
- `Graph`
- `register_architecture_adapter`, `inspect_model_architecture`, `resolve_backend`

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
