# opt_like

## Official Models

- `facebook/opt-125m` (`core`)

## Validation Status

- Route + graph/hook: pass (`reports/route_graph_matrix_v150_full.json`)
- Attribute smoke: pass (`reports/route_graph_matrix_core_with_smoke_v150.json`)
- Strict parity (methods: `EAP`, `EAP-IG-inputs`, `clean-corrupted`, `EAP-IG-activations`):
  - `facebook/opt-125m`: skip (`reports/text_parity_facebook_opt_125m.json`)

## Notes

- Current skip reason:
  - `known parity runtime limit: The size of tensor a (156) must match the size of tensor b (768) at non-singleton dimension 1`
