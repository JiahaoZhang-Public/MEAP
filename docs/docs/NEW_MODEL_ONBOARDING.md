# New Model Onboarding (5-Minute Path)

This guide standardizes how to add a new Hugging Face model to `HFLLMBackend`.

Scope:
- Text models: require method-level parity (`vendor ~= TLens ~= HF`) on supported methods.
- VLM models: smoke-first on language trunk attribution.
- Assumption: model has an identifiable LM decoder backbone.

## 1) Decide Path

Use one of the two paths:

1. Existing adapter already matches model structure:
- No backend code changes.
- Run diagnostics + smoke/parity commands only.

2. New adapter is required:
- Copy adapter template and register it.
- Add tests and run full required checks.

## 2) Add a New Adapter

Template file:
- `meap/backend/adapters/_template.py`

Steps:

1. Create adapter file:
- `meap/backend/adapters/<arch>_like.py`

2. Implement required methods:
- `match(backbone)`
- `get_layers(backbone)`
- `get_embed_module(backbone)`
- `get_resid_module(backbone, layers)`
- `attn_result_module(attn_module)`
- `qkv_hook_modules(attn_module)`
- `projection_spec(attn_module, qkv)`

3. Register adapter:
- Export adapter in `meap/backend/adapters/__init__.py`
- Add class to `_default_adapter_classes()` in `meap/backend/registry.py`

4. Add tests:
- Adapter registry test: `tests/test_adapter_registry.py`
- Backend architecture smoke test: `tests/test_backend_<arch>.py`

## 3) Required Commands (Must Run)

### A. Structure diagnostics

```bash
python scripts/inspect_adapter_registry.py --model-id <model_id> --output reports/inspect_<name>.json
```

Success signal:
- `backend_init.status == "ok"`
- non-empty `adapter_name`, `backbone_path`, `arch_kind`

### B. Smoke run (single model)

Text:

```bash
python scripts/smoke_hf_matrix.py --text-models <model_id> --multimodal-models "" --device cpu --dtype float32 --output reports/smoke_<name>.json
```

VLM:

```bash
python scripts/smoke_hf_matrix.py --text-models "" --multimodal-models <model_id> --device cpu --dtype float32 --output reports/smoke_<name>.json
```

### C. Text parity (for text models)

```bash
python scripts/test_text_vendor_parity.py \
  --models <model_id> \
  --methods EAP,EAP-IG-inputs,clean-corrupted,EAP-IG-activations,exact \
  --strict \
  --max-exact-edges 30000 \
  --output reports/parity_<name>.json
```

### D. Repo quality gates

```bash
ruff check meap tests scripts
pytest -q
```

## 4) Acceptance Thresholds

For text-model strict parity:

- `max_abs <= 1e-3`
- `cosine >= 0.995`
- `top-k overlap >= 0.90` (default `k=200`)

Method policy:

- Required parity methods: `EAP`, `EAP-IG-inputs`, `clean-corrupted`, `EAP-IG-activations`
- `exact`: allowed `skip` when graph edges exceed `--max-exact-edges`; must include skip reason.

For VLM smoke:

- `status == "pass"` in smoke report
- graph can be built (`n_forward`, `n_backward`, `n_edges` > 0)
- failure must include structured diagnostics:
  - `error_type`
  - `error_message`
  - `resolution_error_hint`

## 5) Text vs VLM Notes

Text model onboarding:
- Requires parity validation against vendor/TLens path.
- Preferred baseline models for sanity: `gpt2`, `Qwen/Qwen2-0.5B`.

VLM onboarding:
- At this stage, attribute language trunk only.
- Keep clean/corrupt modality structure aligned to avoid placeholder mismatch.

## 6) Minimal PR Checklist

1. Adapter code + registration (if needed).
2. Tests added/updated.
3. Required commands executed.
4. Report artifacts generated locally (do not commit `reports/*.json`).
5. PR body includes:
- change scope
- test results
- known limits
- next-stage risks
