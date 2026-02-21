# Governance

This document defines contribution and release governance for `meap`.

## Goals

- Traceable changes (clear rationale, reviewable diffs, reproducible runs).
- Stable public API evolution under semantic versioning.
- Reliable quality gates for attribution correctness and backend compatibility.

## Branch and PR Model

- Do not push directly to `main`.
- Use short-lived branches (`feat/*`, `fix/*`, `docs/*`, `test/*`, `chore/*`, `codex/*`).
- Merge through PR only.

Every PR should include:
- scope of change,
- test commands and results,
- known limitations,
- next-stage risks (if staged work).

## Quality Gates

Required before merge:

```bash
ruff check meap tests scripts
pytest -q
mypy
```

For packaging/release-related PRs:

```bash
python -m build
twine check dist/*
```

## API Stability Policy

- Stable APIs are documented in `docs/docs/API_STABILITY.md`.
- Internal modules may change in minor releases.
- V2 high-level input contract is strict:
  - accepted: `PreparedBatch`, `RawPairBatch`
  - rejected: dict/tuple dataloader forms in high-level APIs
- Deprecated symbols must emit `DeprecationWarning` with:
  - deprecation version
  - planned removal version
  - migration path

## Model Support Policy

- Text models: parity-first for required methods.
- Multimodal models: smoke-first on language trunk.
- Official support set must be declared in:
  - `meap/catalog.py`
  - `docs/docs/SUPPORTED_MODELS.md`
- Community adapter path:
  - runtime registration via `register_architecture_adapter(...)`
- Official adapter path:
  - upstream PR with adapter + catalog + tests + docs
- New model onboarding must follow:
  - `docs/docs/NEW_MODEL_ONBOARDING.md`

## Release Policy

- Use semantic versioning.
- Tag each release (`vX.Y.Z`).
- Release notes should summarize:
  - API changes,
  - model support changes,
  - known limitations.
- Update `CHANGELOG.md` for every release.
- Use scripted release flow in `docs/docs/RELEASE_PROCESS.md`.
