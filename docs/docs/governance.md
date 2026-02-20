# Governance

This document defines contribution and release governance for `mm-eap`.

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
ruff check multimodal_lm_eap_ig tests scripts
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
- Deprecated symbols must emit `DeprecationWarning` with:
  - deprecation version
  - planned removal version
  - migration path

## Model Support Policy

- Text models: parity-first for required methods.
- Multimodal models: smoke-first on language trunk.
- New model onboarding must follow:
  - `docs/docs/NEW_MODEL_ONBOARDING.md`

## Release Policy

- Use semantic versioning.
- Tag each release (`vX.Y.Z`).
- Release notes should summarize:
  - API changes,
  - model support changes,
  - known limitations.
