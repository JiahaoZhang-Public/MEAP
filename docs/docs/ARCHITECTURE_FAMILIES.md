# Architecture Families

Last updated: 2026-02-25 (UTC)

This section tracks per-family support status and validation evidence.

## Summary Matrix

| Family | Route + Graph/Hook | Attribute Smoke | Strict Parity (no `exact`) | Notes |
| --- | --- | --- | --- | --- |
| `gpt2_like` | pass | pass (`gpt2`) | pass (`gpt2`, `distilgpt2`) | strict parity fully green for 2/2 models |
| `opt_like` | pass | pass | skip (`facebook/opt-125m`) | known parity runtime limit in vendor/TLens path |
| `llama_like` | pass | pass | partial (`Qwen/Qwen2-0.5B`) | strict parity: 3 pass + 1 fail (`EAP-IG-activations`) |
| `falcon_like` | pass | pass | not in strict vendor parity lane | TLens vendor parity path unavailable |
| `gemma_like` | pass | pass | not in strict vendor parity lane | TLens vendor parity path unavailable |
| `phi_like` | pass | pass | not in strict vendor parity lane | TLens vendor parity path unavailable |

## Evidence Reports

- Route/graph/hook full matrix:
  - `reports/route_graph_matrix_v150_full.json`
- Core smoke matrix:
  - `reports/route_graph_matrix_core_with_smoke_v150.json`
- Nightly smoke matrix:
  - `reports/route_graph_matrix_nightly_with_smoke_v150.json`
- Strict parity reports:
  - `reports/text_parity_gpt2.json`
  - `reports/text_parity_distilgpt2_strict_no_exact.json`
  - `reports/text_parity_facebook_opt_125m.json`
  - `reports/text_parity_qwen2_0_5b_strict_full.json`

## Family Pages

- `gpt2_like`: `architectures/gpt2_like.md`
- `opt_like`: `architectures/opt_like.md`
- `llama_like`: `architectures/llama_like.md`
- `falcon_like`: `architectures/falcon_like.md`
- `gemma_like`: `architectures/gemma_like.md`
- `phi_like`: `architectures/phi_like.md`
