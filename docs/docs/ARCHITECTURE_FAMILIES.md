# Architecture Families

Last updated: 2026-02-25 (UTC)

This section tracks per-family support status and validation state.

## Summary Matrix

| Family | Route + Graph/Hook | Attribute Smoke | Strict Parity (no `exact`) | Notes |
| --- | --- | --- | --- | --- |
| `gpt2_like` | pass | pass (`gpt2`) | pass (`gpt2`, `distilgpt2`) | strict parity fully green for 2/2 models |
| `opt_like` | pass | pass | skip (`facebook/opt-125m`) | known parity runtime limit in vendor/TLens path |
| `llama_like` | pass | pass | partial (`Qwen/Qwen2-0.5B`) | strict parity: 3 pass + 1 fail (`EAP-IG-activations`) |
| `falcon_like` | pass | pass | not in strict vendor parity lane | TLens vendor parity path unavailable |
| `gemma_like` | pass | pass | not in strict vendor parity lane | TLens vendor parity path unavailable |
| `phi_like` | pass | pass | not in strict vendor parity lane | TLens vendor parity path unavailable |

## Validation Sources

- Route/graph/hook matrix checks
- Core and nightly attribute smoke checks
- Strict parity checks for selected text families

## Family Pages

- `gpt2_like`: `architectures/gpt2_like.md`
- `opt_like`: `architectures/opt_like.md`
- `llama_like`: `architectures/llama_like.md`
- `falcon_like`: `architectures/falcon_like.md`
- `gemma_like`: `architectures/gemma_like.md`
- `phi_like`: `architectures/phi_like.md`
