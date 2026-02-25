# gpt2_like

## Official Models

- `gpt2` (`core`)
- `distilgpt2` (`core`)

## Validation Status

- Route + graph/hook: pass (`reports/route_graph_matrix_v150_full.json`)
- Attribute smoke: pass on `gpt2` (`reports/route_graph_matrix_core_with_smoke_v150.json`)
- Strict parity (methods: `EAP`, `EAP-IG-inputs`, `clean-corrupted`, `EAP-IG-activations`):
  - `gpt2`: pass (`reports/text_parity_gpt2.json`)
  - `distilgpt2`: pass (`reports/text_parity_distilgpt2_strict_no_exact.json`)

## Notes

- This family is currently the strongest parity baseline for strict regression.
