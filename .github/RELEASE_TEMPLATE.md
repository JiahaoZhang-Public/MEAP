# {{VERSION}}

Date (UTC): {{DATE_UTC}}
Tag: `{{TAG}}`

## Summary

- 

## Breaking Changes

- 

## Features

- 

## Fixes

- 

## Compatibility

- Python: `>=3.10,<3.13`
- torch: `>=2.1`
- transformers: `>=4.49`
- transformer-lens: `>=2.11`

## Validation

- [ ] `python scripts/release/release.py gate --clean-dist`
- [ ] `python scripts/test_stage_matrix.py --strict --output reports/stage_matrix_release.json` (manual/regression)
- [ ] package artifacts built and checked

## Known Limits

- Multimodal attribution currently targets language-model trunk only.
- Architecture-specific limitations are tracked in `docs/docs/SUPPORTED_MODELS.md`.

## Upgrade Notes

- 
