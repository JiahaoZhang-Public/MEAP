# meap Documentation

Welcome to `meap` (Multimodal Edge Attribution Patching).

This documentation is organized by **what a new user should read first**.

Current baseline:
- API contract: **v2**
- Primary API: `AttributionModel`
- Core contract:
  - `model -> language trunk/graph`
  - `prepared inputs -> attribution`
- Compatibility lane (deprecated): `discover_circuit(...)`, `attribute_from_dataloader(...)`

## Start Here

1. Package quickstart:
- `README.md` (repository root)

Primary entrypoint:
- `AttributionModel` in `meap.api` / `meap`

2. Stable API contract:
- `docs/docs/API_STABILITY.md`

AttributionModel API design and scope:
- `docs/docs/ATTRIBUTION_MODEL_API_DESIGN.md`

3. Runtime compatibility matrix:
- `docs/docs/COMPATIBILITY_MATRIX.md`

4. Model support and smoke matrix:
- `docs/docs/SUPPORTED_MODELS.md`

5. Report schema contract:
- `docs/docs/REPORT_SCHEMAS.md`

6. Release process:
- `docs/docs/RELEASE_PROCESS.md`

7. Add a new model in 5 minutes:
- `docs/docs/NEW_MODEL_ONBOARDING.md`

8. Scripts guide:
- `scripts/README.md`

## Documentation Status

Active documents:
- `API_STABILITY.md`
- `ATTRIBUTION_MODEL_API_DESIGN.md`
- `COMPATIBILITY_MATRIX.md`
- `SUPPORTED_MODELS.md`
- `REPORT_SCHEMAS.md`
- `RELEASE_PROCESS.md`
- `NEW_MODEL_ONBOARDING.md`
- `governance.md`
