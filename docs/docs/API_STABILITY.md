# API Stability (v2, v1.4 Cleanup)

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
- `TaskSpec`

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

## Legacy Compatibility Lane

The following APIs remain available for migration but are no longer primary:
- `discover_circuit(...)`
- `attribute_from_dataloader(...)`
- `evaluate_graph_from_dataloader(...)`
- `evaluate_baseline_from_dataloader(...)`

Compatibility behavior:
- high-level dataloader input is strict:
- accepted: `PreparedBatch`, `RawPairBatch`
- removed: dict/tuple dataloader entries and `DictPairBatch`

- high-level API no longer accepts direct `processor=...`:
- use `pair_batch_preparer=...`
- recommended helper: `HFProcessorAdapter(processor=...)`

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
- `discover_circuit(...)` (legacy compatibility), or
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

Top-level deprecated helpers from the pre-v2 era were removed in `v1.4.0`:
- score helper re-exports (`get_scores_*`, `get_real_edge_scores`)
- preparer helper re-exports (`build_default_llava_processor`, `prepare_llava_token_pair_batch`)

Migration guidance:
- import low-level helpers from implementation modules directly if needed
- prefer `AttributionModel` for stable user-facing workflows
