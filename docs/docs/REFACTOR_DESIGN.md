# REFACTOR DESIGN: From Text-Only EAP-IG to Multimodal EAP-IG

## Purpose
This document describes a high-level refactor plan for evolving the current text-only EAP-IG implementation into a multimodal-capable version while preserving the existing graph semantics and attribution core.

The key product decision is:
- Keep graph construction focused on the LLM trunk only.
- Do not model or attribute inside modality encoders (vision/audio) or projectors.
- Move multimodal preprocessing and clean/corrupt construction outside attribution functions.

---

## Scope and Non-Goals

### In Scope
- Reuse existing EAP/EAP-IG logic over LLM internal nodes and edges.
- Support multimodal inputs that are serialized into LLM-consumable sequence inputs.
- Standardize input/output contracts for attribution and evaluation APIs.

### Out of Scope
- Building graph nodes for modality encoders/projectors.
- Explaining encoder/projector internals.
- Replacing HookedTransformer graph semantics.

---

## 1. Original EAP-IG Design Logic (Text-Only)

### 1.1 Core Abstractions
- `Graph`: defines source nodes, destination indices, edge mask, and score tensors.
- `attribute.py`: computes attribution scores with methods:
  - `EAP`
  - `EAP-IG-inputs`
  - `EAP-IG-activations`
  - `clean-corrupted`
  - `exact`
- `evaluate.py`: evaluates pruned circuits under patching/ablation interventions.
- `utils.py`: tokenization, hook construction, activation difference computation.

### 1.2 Data Contract Today
Current data flow assumes each batch is:
- `clean` text
- `corrupted` text
- `label`

Inside attribution functions:
1. Convert `clean` and `corrupted` into tokens with `tokenize_plus`.
2. Build hook sets and activation-difference buffers.
3. Run corrupted forward pass, clean forward pass, then backward passes.
4. Accumulate edge scores in a `[n_forward, n_backward]` matrix.

### 1.3 Attribution Math Pattern
- EAP computes score updates using:
  - activation differences (`corrupted - clean`)
  - gradients at destination hooks
- EAP-IG approximates path integral by interpolating inputs or activations.
- `clean-corrupted` is a two-point approximation of IG.

### 1.4 Existing Assumptions
- Inputs are text strings handled by TransformerLens tokenizer.
- Hook names and graph indexing are tied to HookedTransformer internals.
- Sequence alignment is implicitly managed by tokenization logic inside attribution.

### 1.5 Limitation for Multimodal
The current API couples attribution to text tokenization and cannot directly accept multimodal-preprocessed model inputs from Hugging Face processors.

---

## 2. Multimodal LM Version of EAP(IG)

### 2.1 Design Principle
Treat multimodal as a preprocessing concern, not a graph-logic concern.

In other words:
- Multimodal encoder/projector transform raw modalities into LLM-facing sequence inputs externally.
- EAP-IG operates only on the LLM sequence representation and internal LLM graph.

### 2.2 Updated Input Contract
Attribution functions should consume preprocessed paired batches instead of raw text pairs:

```python
@dataclass
class PreparedBatch:
    clean_inputs: Dict[str, torch.Tensor]
    corrupt_inputs: Dict[str, torch.Tensor]
    labels: Any
    input_lengths: torch.Tensor
    meta: Optional[Dict[str, Any]] = None
```

Notes:
- `clean_inputs`/`corrupt_inputs` come from external preprocessing (for example HF `AutoProcessor`).
- Fields may include `input_ids`, `attention_mask`, and/or `inputs_embeds` depending on chosen integration.
- `meta` can contain modality spans and corruption metadata.

### 2.3 Two Supported Integration Modes

### Mode A: Token-Level Handoff
- External pipeline produces aligned tokenized model inputs.
- Attribution still runs LLM forward with tokenized inputs.
- Useful when LLM entrypoint and embedding lookup remain stable.

### Mode B: Embedding-Level Handoff
- External pipeline produces aligned `inputs_embeds` for clean/corrupt.
- Attribution injects embeddings at LLM input hook.
- Useful when multimodal formatting is complex or model-specific.

Both modes keep the graph and attribution logic focused on LLM internals only.

### 2.4 Clean/Corrupt Alignment Requirements
For every paired sample:
- Same batch size.
- Same sequence length `L`.
- Same `attention_mask` semantics.
- Same modality placeholder layout in sequence.

If any constraint fails, fail fast before attribution.

### 2.5 Updated Metric Contract
Metrics should accept the prepared batch object to access labels and metadata:

```python
def metric(logits, clean_logits, batch: PreparedBatch) -> torch.Tensor:
    ...
```

### 2.6 Output Contract
Primary attribution output remains unchanged:
- `scores_uv: Tensor[u, v]`
  - `u = # forward/source nodes`
  - `v = # backward edge slots`

Optional convenience outputs:
- real-edge index mapping
- flattened real-edge scores

### 2.7 Backward Compatibility
Keep current text-only API as a wrapper:
- Adapter converts `(clean, corrupted, label)` batches into `PreparedBatch`.
- Core attribution code uses only the new prepared-batch path.

---

## 3. High-Level Design

### 3.1 Target Architecture

1. `BatchPreparer` layer
- Owns all preprocessing, clean/corrupt generation, and validation.
- Uses HF processor (or any custom pipeline) outside attribution.

2. `PreparedBatch` stream
- Dataloader yields standardized prepared batches.

3. Attribution core (`attribute.py`)
- Reuses existing EAP/EAP-IG math and graph scoring.
- Removes internal tokenization dependence.

4. Evaluation core (`evaluate.py`)
- Uses same prepared batch contract.
- Preserves intervention semantics (patching/zero/mean).

5. Graph layer (`graph.py`)
- Stays LLM-centric and unchanged in semantic meaning.

### 3.2 API Direction

Current style:
- `get_scores_eap(model, graph, dataloader(clean,corrupt,label), metric, ...)`

Refactored style:
- `get_scores_eap(model, graph, prepared_batches, metric, ...)`

Where:
- `prepared_batches: Iterable[PreparedBatch]`
- `metric` consumes the batch object.

### 3.3 Refactor Steps

1. Introduce shared prepared-batch types.
2. Add validation utilities for clean/corrupt alignment.
3. Refactor `get_scores_eap`, `get_scores_eap_ig`, `get_scores_clean_corrupted`, and `evaluate_graph` to consume prepared batches.
4. Provide a text-only adapter for existing datasets.
5. Add multimodal adapter examples using HF processor.
6. Add regression tests for text-only parity and multimodal shape/alignment checks.

### 3.4 Risks and Mitigations

- Risk: Hook mismatch due to different runtime input path.
  - Mitigation: enforce a single supported LLM entry path per mode and test hook coverage.

- Risk: Incorrect clean/corrupt pairing.
  - Mitigation: strict pre-attribution validators with explicit failure messages.

- Risk: Behavior drift in text-only workflows.
  - Mitigation: keep legacy wrapper and add parity tests against current outputs.

### 3.5 Deliverables
- Refactored attribution/evaluation APIs using `PreparedBatch`.
- Text-only compatibility wrapper.
- Multimodal preprocessing examples (HF processor-based).
- Documentation updates and migration notes.

---

## Summary
The refactor should preserve what already works:
- same graph semantics,
- same attribution math,
- same score output shape.

It should only change where multimodal complexity lives:
- outside attribution, inside a standardized preprocessing layer that emits aligned clean/corrupt LLM-ready inputs.
