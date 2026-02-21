# Examples

This directory keeps only core, release-facing examples.

## Primary Example (Prepared-Input First)

- `examples/text/attribution_model_prepared.py`

```bash
python examples/text/attribution_model_prepared.py --device cpu --dtype float32 --method smoke
```

## Per-Modality Walkthroughs

- Text: `examples/text/gpt2.py`
- Image: `examples/image/Qwen2-VL-2B.py`
- Audio: `examples/audio/ultravox.py`

Each walkthrough follows:
1. raw data -> processing -> model inputs
2. attribution
3. graph export

## Notes

- `--input-mode` defaults to `prepared` in modality scripts.
- `pygraphviz` is required for PNG graph export.
- Clean/corrupt alignment guidance: `examples/CLEAN_CORRUPT_ALIGNMENT.md`.
