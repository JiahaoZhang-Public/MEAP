# Changelog

All notable changes to this project are documented here.

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
