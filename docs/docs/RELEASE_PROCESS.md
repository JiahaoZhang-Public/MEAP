# Release Process (v1.0.0)

This document defines the scripted release flow for `meap`.

## Release Stages

1. `v1.0.0-rc1` (pre-release candidate)
2. `v1.0.0` (stable release)

## Step-by-Step

### 1) Prepare

- Ensure working tree is clean.
- Update:
  - `CHANGELOG.md`
  - docs (compatibility, known limits, installation notes)

### 2) Run release gate

```bash
python scripts/release/release.py set-version --version 1.0.0rc1
python scripts/release/release.py gate --clean-dist
```

This runs:
- `ruff`
- `mypy`
- `pytest`
- `python -m build`
- `twine check`

### 3) Run manual/regression matrix

```bash
python scripts/test_stage_matrix.py --strict --output reports/stage_matrix_release.json
```

Or use GitHub Actions manual workflow:
- `.github/workflows/regression-matrix.yml`

### 4) Draft release notes

```bash
python scripts/release/release.py notes --version v1.0.0-rc1
```

Template:
- `.github/RELEASE_TEMPLATE.md`

### 5) Tag

```bash
python scripts/release/release.py tag --version v1.0.0-rc1 --push
```

After RC validation, repeat for stable:

```bash
python scripts/release/release.py set-version --version 1.0.0
python scripts/release/release.py tag --version v1.0.0 --push
```

## Post-Release Documentation Sync

After final `v1.0.0`:
- verify install command in `README.md`
- verify compatibility versions in `docs/docs/COMPATIBILITY_MATRIX.md`
- verify known limits in `README.md` and `docs/docs/SUPPORTED_MODELS.md`
- record release highlights in `CHANGELOG.md`
