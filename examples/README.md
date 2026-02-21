# Examples

Per-modality end-to-end attribution examples using `meap`.

Each example script follows the same 3-stage flow:
1. Raw data -> processing -> dataloader input
   (`PreparedBatch` or `RawPairBatch + pair_batch_preparer`, selected by `--input-mode`)
2. Attribution (default `EAP`, `--method` also accepts other methods)
3. Graph visualization export (`graph_full.json`, `graph_topn.json`, `graph_topn.png`)

`--input-mode` defaults to `prepared`.

Outputs are saved under each modality folder and retained by run name/timestamp.

## Requirements

Recommended environment setup for running examples:

```bash
pip install -r requirements.txt
pip install -e .[multimodal,viz]
```

Alternative one-shot setup:

```bash
pip install -r requirements-dev.txt
```

Visualization dependency:
- `pygraphviz` is required for PNG graph rendering.
- Install with: `pip install pygraphviz`

## Text

Script: `examples/text/gpt2.py`

```bash
python examples/text/gpt2.py --device cpu --dtype float32 --method EAP --input-mode prepared
python examples/text/gpt2.py --device cpu --dtype float32 --method EAP --input-mode raw
```

Output folder:
- `examples/text/outputs/<run_name_or_timestamp>/`

Minimal prepared-input-first API example:
- Script: `examples/text/attribution_model_prepared.py`

```bash
python examples/text/attribution_model_prepared.py --device cpu --dtype float32 --method smoke
```

## Image

Script: `examples/image/Qwen2-VL-2B.py`

```bash
python examples/image/Qwen2-VL-2B.py --dtype float16 --method EAP --image-size 128 --input-mode prepared
python examples/image/Qwen2-VL-2B.py --dtype float16 --method EAP --image-size 128 --input-mode raw
```

Notes:
- The script uses a white clean image and a black corrupt image.
- Use `--image-size` to control dynamic image token count and memory use.

Output folder:
- `examples/image/outputs/<run_name_or_timestamp>/`

Non-empty top200 variant:
- Script: `examples/image/Qwen2-VL-2B_nonempty.py`
- Uses root-aware pruning (`input_layer0`) to avoid empty top200 pruned graph.

```bash
python examples/image/Qwen2-VL-2B_nonempty.py --dtype float16 --method EAP --image-size 128 --input-mode prepared
python examples/image/Qwen2-VL-2B_nonempty.py --dtype float16 --method EAP --image-size 128 --input-mode raw
```

## Audio

Script: `examples/audio/ultravox.py`

```bash
python examples/audio/ultravox.py --dtype float32 --method EAP --input-mode prepared
python examples/audio/ultravox.py --dtype float32 --method EAP --input-mode raw
```

Optional audio source override:
- `--audio-path /local/path/audio.wav`
- `--audio-url https://...`

Output folder:
- `examples/audio/outputs/<run_name_or_timestamp>/`

Non-empty top200 variant:
- Script: `examples/audio/ultravox_nonempty.py`
- Uses root-aware pruning (`input_layer0`) to avoid empty top200 pruned graph.

```bash
python examples/audio/ultravox_nonempty.py --dtype float32 --method EAP --input-mode prepared
python examples/audio/ultravox_nonempty.py --dtype float32 --method EAP --input-mode raw
```

## Clean/Corrupt Alignment Guide

See `examples/CLEAN_CORRUPT_ALIGNMENT.md` for how to construct aligned clean/corrupt samples correctly, with one real example for each modality (text/image/audio).
