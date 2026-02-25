# opt_like

## Official Models

- `facebook/opt-125m` (`core`)

## Validation Status

- Route + graph/hook: pass
- Attribute smoke: pass
- Strict parity (methods: `EAP`, `EAP-IG-inputs`, `clean-corrupted`, `EAP-IG-activations`):
  - `facebook/opt-125m`: skip

## Notes

- Current skip reason:
  - `known parity runtime limit: The size of tensor a (156) must match the size of tensor b (768) at non-singleton dimension 1`
