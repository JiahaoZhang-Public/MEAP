# Release Process (v1.x)

This document defines the scripted release flow for `meap`.

## Release Stages

1. `vX.Y.Z-rc1` (pre-release candidate, optional)
2. `vX.Y.Z` (stable release)

## Step-by-Step

### 1) Prepare

- Ensure working tree is clean.
- Update:
  - `CHANGELOG.md`
  - docs (compatibility, known limits, installation notes)

### 2) Run release gate

```bash
python scripts/release/release.py set-version --version X.Y.Zrc1
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
python scripts/test_stage_matrix.py --strict --output artifacts/stage_matrix_release.json
```

Or use GitHub Actions manual workflow:
- `.github/workflows/regression-matrix.yml`

### 4) Draft release notes

```bash
python scripts/release/release.py notes --version vX.Y.Z-rc1
```

Template:
- `.github/RELEASE_TEMPLATE.md`

### 5) Tag

```bash
python scripts/release/release.py tag --version vX.Y.Z-rc1 --push
```

After RC validation, repeat for stable:

```bash
python scripts/release/release.py set-version --version X.Y.Z
python scripts/release/release.py tag --version vX.Y.Z --push
```

## Post-Release Documentation Sync

After final `vX.Y.Z`:
- verify install command in `README.md`
- verify API/runtime versions in `pyproject.toml` and `docs/docs/API_STABILITY.md`
- verify known limits in `README.md` and `docs/docs/SUPPORTED_MODELS.md`
- record release highlights in `CHANGELOG.md`
