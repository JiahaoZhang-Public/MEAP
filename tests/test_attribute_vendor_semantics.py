import importlib
from types import SimpleNamespace

import torch

from meap.backend.base import BackendConfig, BackendRunInputs
from meap.batch import PreparedBatch

attribute_module = importlib.import_module("meap.attribute")


class _DummyNode:
    def __init__(self, name: str):
        self.out_hook = name


class _DummyGraph:
    def __init__(self):
        self.cfg = {"n_layers": 1}
        self.n_forward = 1
        self.n_backward = 1
        self.nodes = {
            "input": _DummyNode("input_hook"),
            "a0.h0": _DummyNode("attn_hook"),
            "m0": _DummyNode("mlp_hook"),
        }

    def forward_index(self, node) -> int:
        del node
        return 0


class _DummyBackend:
    kind = "hf"

    def __init__(self):
        self.config = BackendConfig(
            device=torch.device("cpu"),
            dtype=torch.float32,
            d_model=1,
            n_layers=1,
            n_heads=1,
            parallel_attn_mlp=False,
            n_key_value_heads=None,
            use_attn_result=True,
            use_split_qkv_input=True,
            use_hook_mlp_in=True,
            use_normalization_before_and_after=False,
        )
        self.tokenizer = None
        self.tokenization_model = None

    def prepare_inputs(self, inputs):
        del inputs
        return BackendRunInputs(
            input_ids=torch.ones((1, 3), dtype=torch.long),
            inputs_embeds=None,
            attention_mask=torch.ones((1, 3), dtype=torch.long),
            extra_fwd_hooks=[],
            model_kwargs={},
        )

    def forward(self, run_inputs, *, fwd_hooks=None, bwd_hooks=None):
        del run_inputs, fwd_hooks, bwd_hooks
        return torch.tensor(1.0, requires_grad=True)

    def zero_grad(self):
        return None

    def parameters(self):
        return []


def _prepared_batch() -> PreparedBatch:
    clean_ids = torch.tensor([[1, 2, 3]])
    corrupt_ids = torch.tensor([[1, 2, 4]])
    mask = torch.ones_like(clean_ids)
    return PreparedBatch(
        clean_inputs={"input_ids": clean_ids, "attention_mask": mask},
        corrupt_inputs={"input_ids": corrupt_ids, "attention_mask": mask.clone()},
        labels=None,
        input_lengths=mask.sum(dim=-1),
    )


def _metric(logits, clean_logits, batch):
    del clean_logits, batch
    return logits


def test_eap_ig_activations_uses_vendor_step_normalization(monkeypatch):
    backend = _DummyBackend()
    graph = _DummyGraph()
    batches = [_prepared_batch(), _prepared_batch()]
    state = {"call": 0, "scores": None}

    def fake_resolve_run_inputs(backend_obj, inputs):
        run_inputs = backend_obj.prepare_inputs(inputs)
        return SimpleNamespace(run_inputs=run_inputs)

    def fake_make_hooks_and_matrices(backend_obj, graph_obj, batch_size, n_pos, scores):
        del backend_obj, graph_obj, batch_size, n_pos
        state["call"] += 1
        state["scores"] = scores
        zeros = torch.zeros((1, 3, graph.n_forward, backend.config.d_model), dtype=backend.config.dtype)
        if state["call"] % 3 == 1:
            return (([], [], [("fake_bwd", lambda grad, hook: grad)]), zeros.clone())
        if state["call"] % 3 == 2:
            return (([("fake_fwd_corrupt", lambda x, h: x)], [], []), torch.ones_like(zeros))
        return (([("fake_fwd_clean", lambda x, h: x)], [], []), zeros.clone())

    def fake_forward_with_hooks(backend_obj, model_inputs, *, fwd_hooks=None, bwd_hooks=None):
        del backend_obj, model_inputs, fwd_hooks
        if bwd_hooks is not None:
            state["scores"] += 1.0
        return torch.tensor(1.0, requires_grad=True)

    monkeypatch.setattr(attribute_module, "resolve_run_inputs", fake_resolve_run_inputs)
    monkeypatch.setattr(attribute_module, "make_hooks_and_matrices", fake_make_hooks_and_matrices)
    monkeypatch.setattr(attribute_module, "forward_with_hooks", fake_forward_with_hooks)

    scores = attribute_module.get_scores_ig_activations(
        model=None,
        graph=graph,
        batches=batches,
        metric=_metric,
        backend=backend,
        steps=2,
        quiet=True,
    )

    # 2 batches * (3 nodes * 2 steps) updates = 12 raw increments.
    # Vendor semantics divide by total_items (2) and by per-batch total_steps (6): 12 / 2 / 6 = 1.
    assert torch.allclose(scores, torch.ones_like(scores), atol=1e-6, rtol=1e-6)
