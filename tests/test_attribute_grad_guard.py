import importlib

import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from multimodal_lm_eap_ig.backend import TLensBackend
from multimodal_lm_eap_ig.graph import Graph

attribute_module = importlib.import_module("multimodal_lm_eap_ig.attribute")


def _tiny_tlens_model(*, requires_grad: bool) -> HookedTransformer:
    cfg = HookedTransformerConfig(
        n_layers=1,
        n_heads=1,
        d_model=16,
        d_head=16,
        d_mlp=32,
        n_ctx=16,
        d_vocab=128,
        act_fn="relu",
    )
    model = HookedTransformer(cfg)
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    if not requires_grad:
        for p in model.parameters():
            p.requires_grad_(False)
    return model


def _dummy_metric(logits, clean_logits, batch):
    del logits, clean_logits, batch
    return torch.tensor(0.0)


def _dummy_graph() -> Graph:
    return Graph.from_model(
        {
            "n_layers": 1,
            "n_heads": 1,
            "parallel_attn_mlp": True,
            "d_model": 16,
        }
    )


def test_attribute_eap_raises_when_model_params_are_frozen():
    model = _tiny_tlens_model(requires_grad=False)
    backend = TLensBackend(model)
    graph = _dummy_graph()

    with pytest.raises(RuntimeError, match="requires_grad=True"):
        attribute_module.attribute(
            model=model,
            backend=backend,
            graph=graph,
            batches=[],
            metric=_dummy_metric,
            method="EAP",
        )


def test_attribute_eap_raises_under_no_grad_context():
    model = _tiny_tlens_model(requires_grad=True)
    backend = TLensBackend(model)
    graph = _dummy_graph()

    with torch.no_grad():
        with pytest.raises(RuntimeError, match="torch.no_grad\\(\\)|torch.inference_mode\\(\\)"):
            attribute_module.attribute(
                model=model,
                backend=backend,
                graph=graph,
                batches=[],
                metric=_dummy_metric,
                method="EAP",
            )


def test_attribute_exact_is_not_blocked_by_grad_guard(monkeypatch):
    model = _tiny_tlens_model(requires_grad=False)
    backend = TLensBackend(model)
    graph = _dummy_graph()
    called = {"value": False}

    def fake_get_scores_exact(*args, **kwargs):
        del args, kwargs
        called["value"] = True
        return torch.ones_like(graph.scores)

    monkeypatch.setattr(attribute_module, "get_scores_exact", fake_get_scores_exact)

    scores = attribute_module.attribute(
        model=model,
        backend=backend,
        graph=graph,
        batches=[],
        metric=_dummy_metric,
        method="exact",
    )

    assert called["value"]
    assert torch.all(scores == 1)


def test_attribute_smoke_is_not_blocked_by_grad_guard(monkeypatch):
    model = _tiny_tlens_model(requires_grad=False)
    backend = TLensBackend(model)
    graph = _dummy_graph()
    called = {"value": False}

    def fake_get_scores_smoke(*args, **kwargs):
        del args, kwargs
        called["value"] = True
        return torch.zeros_like(graph.scores)

    monkeypatch.setattr(attribute_module, "get_scores_smoke", fake_get_scores_smoke)

    with torch.no_grad():
        scores = attribute_module.attribute(
            model=model,
            backend=backend,
            graph=graph,
            batches=[],
            metric=_dummy_metric,
            method="smoke",
        )

    assert called["value"]
    assert torch.all(scores == 0)
