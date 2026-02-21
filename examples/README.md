# Examples (Prepared-Input-First)

This directory demonstrates the **current meap API**:

1. `model -> language trunk/graph` (via `AttributionModel`)
2. `prepared inputs -> attribution` (via `PreparedBatch`)

The core API does not accept raw `clean_samples` / `corrupt_samples` directly.
Each script shows how to turn raw modality data into `PreparedBatch` first.

## Prerequisites

Install runtime + multimodal + graph export extras:

```bash
pip install -e ".[multimodal,viz]"
```

Optional for private/gated models:

```bash
huggingface-cli login
```

## Example Index

- Minimal API quickstart: `examples/text/attribution_model_prepared.py`
- Text walkthrough (GPT-2): `examples/text/gpt2.py`
- Image walkthrough (Qwen2-VL-2B): `examples/image/Qwen2-VL-2B.py`
- Audio walkthrough (Ultravox): `examples/audio/ultravox.py`

## Quick Commands

Minimal smoke test:

```bash
python examples/text/attribution_model_prepared.py --device cpu --dtype float32 --method smoke
```

Text attribution + graph export:

```bash
python examples/text/gpt2.py --device cpu --dtype float32 --method EAP
```

Image attribution + graph export:

```bash
python examples/image/Qwen2-VL-2B.py --device cpu --dtype float32 --method EAP
```

Audio attribution + graph export:

```bash
python examples/audio/ultravox.py --device cpu --dtype float32 --method EAP
```

## Output Artifacts

The 3 walkthrough scripts write outputs under each modality directory:

- `outputs/<run_name_or_timestamp>/run_summary.json`
- `outputs/<run_name_or_timestamp>/model_input_summary.json`
- `outputs/<run_name_or_timestamp>/graph_full.json`
- `outputs/<run_name_or_timestamp>/graph_topn.json`
- `outputs/<run_name_or_timestamp>/graph_topn.png`

`run_summary.json` includes route resolution (`adapter_name`, `language_trunk_path`) and graph stats.

## PreparedBatch Contract

All scripts construct this shape before calling `attribute(...)`:

```python
PreparedBatch(
    clean_inputs={...},
    corrupt_inputs={...},
    labels=...,            # optional for some metrics
    input_lengths=...,     # usually attention_mask.sum(-1)
    meta=...,              # optional
)
```

For clean/corrupt alignment rules, see:
`examples/CLEAN_CORRUPT_ALIGNMENT.md`

## Method Choices

Supported `--method` values:

- `smoke` (fast sanity check)
- `EAP`
- `EAP-IG-inputs`
- `clean-corrupted`
- `EAP-IG-activations`
- `exact` (slow, debugging/reference)

## Troubleshooting

- `ModuleNotFoundError: pygraphviz`: install `.[viz]`.
- HF auth / gated model errors (`401`/`403`): run `huggingface-cli login` and pass `--hf-token` if needed.
- OOM on GPU: switch to `--device cpu` or smaller model.
- Multimodal prompt/placeholder mismatch: check the generated `model_input_summary.json` first.
