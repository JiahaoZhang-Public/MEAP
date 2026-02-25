# API Stability (v1.5.0)

This page is the normative stability contract for `meap` public APIs.

If a statement here conflicts with other docs, treat this page as the source of truth.

## Stable Entry Points

Primary stable modules:
- `meap`
- `meap.api`

Primary stable class:
- `AttributionModel`

Stable core symbols:
- `AttributionModel`
- `AttributionResult`
- `RouteInfo`
- `PreparedInputLike`
- `TaskSpec`

Stable dataloader evaluation APIs:
- `evaluate_graph_from_dataloader(...)`
- `evaluate_baseline_from_dataloader(...)`

## Core Contract (AttributionModel)

Canonical path:
1. `model -> language trunk/graph`
2. `prepared inputs -> attribution`

`AttributionModel` core methods accept:
- `PreparedBatch`, or
- nested prepared mapping with:
  - `clean_inputs`
  - `corrupt_inputs`
  - optional `labels`, `input_lengths`, `meta`

`AttributionModel` core methods do **not** accept:
- `clean_samples`
- `corrupt_samples`
- `pair_batch_preparer`
- `RawPairBatch`

## Dataloader Evaluation Contract

The dataloader wrappers support evaluation-oriented workflows:
- `evaluate_graph_from_dataloader(...)`
- `evaluate_baseline_from_dataloader(...)`

For these wrappers:
- accepted dataloader entries: `PreparedBatch`, `RawPairBatch`
- use `pair_batch_preparer=...` when raw pairs are provided

## HF Route Resolution Contract

Stable route concepts:
- `language_trunk_path`: selected language-model trunk path inside the HF model
- `adapter_name`: architecture adapter used on that trunk

Stable diagnostics keys in `RouteInfo.diagnostics`:
- `selected`
- `candidate_backbones`
- `adapter_attempts`
- `selection_error`

## Internal Modules (Not Stability-Guaranteed)

The following modules are implementation details and may change in minor releases:
- `meap.attribute`
- `meap.evaluate`
- `meap.utils`

## Version History References

For historical removals and migration notes, see:
- `CHANGELOG.md`
- `releases/v1.4.0.md`
- `releases/v1.5.0.md`
