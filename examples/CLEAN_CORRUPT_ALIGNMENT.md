# Clean/Corrupt Alignment Guide

This document explains how to build clean/corrupt sample pairs that are valid for attribution in `meap`.

If clean/corrupt are not aligned, common failures include:
- `attention_mask semantics must match between clean and corrupt`
- `Sequence length mismatch`
- modality placeholder errors (image/audio tokens missing or inconsistent)

Important note for small topn exports:
- For multimodal models, `topn=200` with standard prune can still produce an empty pruned graph even when scores are non-zero.
- Use non-empty variants (`examples/image/Qwen2-VL-2B_nonempty.py`, `examples/audio/ultravox_nonempty.py`) when you need stable non-empty top200 circuits.

## Core Rules

1. Keep the same task across clean/corrupt.
2. Keep batch size aligned.
3. Align sequence tensors (`input_ids` same length, `attention_mask` semantics same).
4. For multimodal inputs, placeholder/layout must match model expectations.

Details:
- Image models require valid image placeholders.
- Audio-chat models require valid audio placeholders.

## Text Example (GPT-2)

Real example: `examples/text/gpt2.py`

Clean/corrupt pair:
- clean: `The capital of France is`
- corrupt: `The capital of Germany is`

Alignment pattern:
- tokenize both
- pad to same length
- use `shared_mask = min(clean_mask, corrupt_mask)`
- build one `PreparedBatch`

Why this is valid:
- same task format, one semantic change
- token/mask alignment enforced before attribution

## Image Example (Qwen2-VL-2B)

Real example: `examples/image/Qwen2-VL-2B.py`
Non-empty variant: `examples/image/Qwen2-VL-2B_nonempty.py`

Clean/corrupt pair:
- clean image: solid white RGB image
- corrupt image: solid black RGB image
- prompt (both): `what is the color of the image?`
- metric labels: single-token ids for `white` and `black`

Alignment pattern:
- ensure prompt template includes valid image placeholders
- process clean/corrupt separately
- pad `input_ids` to same length
- align masks via shared mask

Why this is valid:
- question is identical, only visual evidence changes
- low resolution (`--image-size`) controls dynamic image token count

## Audio Example (Ultravox)

Real example: `examples/audio/ultravox.py`
Non-empty variant: `examples/audio/ultravox_nonempty.py`

Clean/corrupt pair:
- audio waveform: same clip for both
- clean prompt: summarize speech
- corrupt prompt: transcribe speech
- user turns normalized to contain exactly one audio placeholder

Alignment pattern:
- run pipeline preprocessing for each side
- pad `input_ids` and `attention_mask` to same length
- align masks with shared mask

Why this is valid:
- same audio evidence, controlled prompt perturbation
- avoids Ultravox placeholder count errors

## Practical Checklist

Before running attribution, verify:
- `clean_inputs["input_ids"].shape == corrupt_inputs["input_ids"].shape`
- `torch.equal(clean_inputs["attention_mask"], corrupt_inputs["attention_mask"])`
- `batch.input_lengths == attention_mask.sum(-1)`
- modality placeholders are present and counts match
