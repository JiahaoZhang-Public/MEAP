# Changelog

All notable changes to this project are documented here.

## [1.4.0] - 2026-02-21

### Breaking
- Removed long-deprecated top-level helper re-exports from `meap` package root:
  - score helpers (`get_real_edge_scores`, `get_scores_*`)
  - preparer helpers (`build_default_llava_processor`, `prepare_llava_token_pair_batch`)

### Features
- Standardized `scripts/api_minimal_examples.py` to run through `AttributionModel` core path.
- Added optional attribution smoke (`--run-attribute-smoke`) in route matrix validation workflow for release checks.

### Cleanup
- Removed redundant example/script variants:
  - `examples/image/Qwen2-VL-2B_nonempty.py`
  - `examples/audio/ultravox_nonempty.py`
  - `scripts/discover_circuit_minimal.py`
- Removed obsolete docs page:
  - `docs/docs/COMPATIBILITY_MATRIX.md`

### Docs
- Reorganized and simplified top-level docs/README surfaces to be prepared-input-first and release-oriented.
- Updated onboarding/support/release docs to prefer `AttributionModel` language-trunk workflow.

## [1.3.0] - 2026-02-21

### Features
- Added `AttributionModel` as the primary object-oriented public API for:
  - `model -> language trunk/graph`
  - `prepared inputs -> attribution`
- Added new public data types:
  - `AttributionResult`
  - `RouteInfo`
  - `PreparedInputLike`
  - `ComponentSpec` (MVP contract placeholder)
- Added nested prepared mapping input normalization in `AttributionModel`:
  - required: `clean_inputs`, `corrupt_inputs`
  - optional: `labels`, `input_lengths`, `meta`
- Added `scripts/route_graph_matrix.py` for matrix validation of:
  - `model -> language trunk -> graph/hook` route discovery
  - text + multimodal official model coverage
  - optional one-shot `PreparedBatch` attribution smoke per model (`--run-attribute-smoke`)

### Compatibility / Deprecation
- `discover_circuit(...)` and `attribute_from_dataloader(...)` are retained as compatibility wrappers and now emit `DeprecationWarning` with `AttributionModel` migration hints.

### Docs
- Added `docs/docs/ATTRIBUTION_MODEL_API_DESIGN.md` implementation notes and MVP limits.
- Updated API stability and package docs to mark `AttributionModel` as primary entrypoint.

## [1.2.0] - 2026-02-21

### Breaking
- High-level dataloader contract is now strict:
  - accepted: `PreparedBatch`, `RawPairBatch`
  - removed: dict/tuple-compatible high-level dataloader forms (`DictPairBatch` removed from public surface)
- High-level API no longer accepts direct `processor=...`:
  - use `pair_batch_preparer=...`
  - recommended helper: `HFProcessorAdapter(processor=...)`

### Features
- High-level HF selection now supports:
  - `adapter_name`
  - `language_trunk_path`
- HF architecture diagnostics standardized with:
  - `selected`
  - `candidate_backbones`
  - `adapter_attempts`
  - `selection_error`
- Added official runtime catalog module:
  - `meap.catalog`
  - `list_supported_architectures()`
  - `list_official_models()`
- Official core support set defined as:
  - text: `gpt2`, `distilgpt2`, `facebook/opt-125m`, `Qwen/Qwen2-0.5B`
  - multimodal: `Qwen/Qwen2-VL-2B`, `llava-hf/llava-1.5-7b-hf`

### Docs
- Updated API stability page to v2 contract.
- Updated supported models page to catalog-aligned core set.
- Updated onboarding and governance docs for runtime registration vs official support paths.

## [1.1.0] - 2026-02-20

### Breaking
- Project identity renamed:
  - GitHub repository: `MEAP`
  - PyPI package: `meap`
  - Python import path: `meap` (old `multimodal_lm_eap_ig` removed)

### Features
- Public API surface stabilized around `meap.api` and top-level `meap` exports.
- Hugging Face backend architecture adapter registry with diagnostics (`llama_like`, `gpt2_like`, `opt_like`, `falcon_like`, `mpt_like`).
- Unified input layer supporting:
  - prebuilt `PreparedBatch`
  - raw clean/corrupt pairs via `processor`
  - raw clean/corrupt pairs via `pair_batch_preparer`
- Multimodal language-trunk attribution support (smoke-first).
- Unified matrix tooling for text parity (vendor/TLens/HF) and multimodal smoke.
- Versioned report schemas and release gating utilities.

### Fixes
- Qwen2 alignment between backend paths improved and parity workflow hardened.
- HF backend diagnostics now expose selected backbone path, adapter, and failed-attempt hints.
- CI quality gates hardened with lint/type/test/package checks.

## [0.3.0] - 2026-02-20

### Features
- New model onboarding guide (`docs/docs/NEW_MODEL_ONBOARDING.md`) and adapter template hardening.
- Documentation refactor for stable API and package-level usage.

## [0.2.1-user-managed-prep]

### Features
- User-managed preprocessing contract introduced (`PreparedBatch` or explicit `processor`/`pair_batch_preparer`).

## [0.2.0-hf-unified-input-arch]

### Features
- HF backend path introduced as default direction with unified input flow.

## [0.1.0-llava-smoke-hooked-transformer]

### Features
- Initial LLaVA smoke attribution support via HookedTransformer backend.
