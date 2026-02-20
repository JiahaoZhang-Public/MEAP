# Changelog

All notable changes to this project are documented here.

## [1.0.0-rc1] - Planned

### Breaking
- Package identity moved to `mm-eap` (PyPI) while keeping import path `multimodal_lm_eap_ig`.
- Public API surface is now explicitly stabilized around `multimodal_lm_eap_ig.api` and top-level exports.
- Internal modules are marked unstable (`attribute.py`, `evaluate.py`, `utils.py`).
- Top-level low-level helpers are deprecated (since `1.0.0`, planned removal in `1.2.0`).

### Features
- Hugging Face backend architecture adapter registry with diagnostics (`llama_like`, `gpt2_like`, `opt_like`, `falcon_like`, `mpt_like`).
- Unified input layer supporting:
  - prebuilt `PreparedBatch`
  - raw clean/corrupt pairs via `processor`
  - raw clean/corrupt pairs via `pair_batch_preparer`
- Multimodal language-trunk attribution support (smoke-first).
- Unified stage matrix tooling for:
  - text parity (vendor/TLens/HF)
  - multimodal smoke matrix
- Versioned report schemas for parity/smoke/stage outputs.

### Fixes
- Qwen2 alignment issues between backend paths reduced and parity workflow hardened.
- HF backend diagnostics improved with selected backbone path, adapter, and failed attempt hints.
- CI hardened with lint/type/test/package gates and tiny smoke gate.

## [1.0.0] - Planned

### Release
- Promote `v1.0.0-rc1` after regression validation.
- Keep API/deprecation contracts unchanged from `v1.0.0-rc1`.

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
