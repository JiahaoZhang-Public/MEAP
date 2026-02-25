# llama_like

## Official Models

Text:
- `Qwen/Qwen2-0.5B` (`core`)
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (`core`)

Multimodal:
- `Qwen/Qwen2-VL-2B` (`core`)
- `HuggingFaceTB/SmolVLM-Instruct` (`core`)
- `fixie-ai/ultravox-v0_5-llama-3_2-1b` (`core`)
- `llava-hf/llava-1.5-7b-hf` (`nightly`)
- `Qwen/Qwen2-Audio-7B` (`nightly`)
- `HuggingFaceM4/idefics2-8b` (`nightly`)

## Validation Status

- Route + graph/hook: pass for all official models
- Attribute smoke:
  - core set: pass
  - nightly set: pass
- Strict parity (text, methods without `exact`):
  - `Qwen/Qwen2-0.5B`: partial
  - method-level: 3 pass, 1 fail (`EAP-IG-activations`, `max_abs=1.1458e-03`, threshold `1e-3`)

## Notes

- Most multimodal support in v1.5.0 is routed through this family’s language trunk.
