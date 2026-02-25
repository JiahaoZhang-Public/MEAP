# gpt2_like

## Official Models

- `gpt2` (`core`)
- `distilgpt2` (`core`)

## Validation Status

- Route + graph/hook: pass
- Attribute smoke: pass on `gpt2`
- Strict parity (methods: `EAP`, `EAP-IG-inputs`, `clean-corrupted`, `EAP-IG-activations`):
  - `gpt2`: pass
  - `distilgpt2`: pass

## Notes

- This family is currently the strongest parity baseline for strict regression.
