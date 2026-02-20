# Examples

Per-modality end-to-end attribution examples using `mm-eap`.

Each example script follows the same 3-stage flow:
1. Raw data -> processing -> model input tensors (`PreparedBatch`)
2. Attribution (default `EAP`, `--method` also accepts other methods)
3. Graph visualization export (`graph_full.json`, `graph_topn.json`, `graph_topn.png`)

Outputs are saved under each modality folder and retained by run name/timestamp.

## Text

Script: `examples/text/gpt2.py`

```bash
python examples/text/gpt2.py --device cpu --dtype float32 --method EAP
```

Output folder:
- `examples/text/outputs/<run_name_or_timestamp>/`

## Image

Script: `examples/image/Qwen2-VL-2B.py`

```bash
python "examples/image/Qwen2-VL-2B.py" --device cpu --dtype float32 --method EAP
```

Optional data source override:
- `--image-path /local/path/image.jpg`
- `--image-url https://...`

Output folder:
- `examples/image/outputs/<run_name_or_timestamp>/`

## Audio

Script: ` examples/audio/ultravox.py`

```bash
python examples/audio/ultravox.py --device cpu --dtype float32 --method EAP
```

Optional data source override:
- `--audio-path /local/path/audio.wav`
- `--audio-url https://...`

Output folder:
- `examples/audio/outputs/<run_name_or_timestamp>/`

## Notes

- For PNG graph export, install `pygraphviz` (`pip install pygraphviz`).
- For audio URL/local loading, install `librosa` (`pip install librosa`).
- All examples use language-model trunk attribution via HF backend.
