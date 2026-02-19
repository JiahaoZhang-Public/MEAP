import types

import importlib
import pytest
import torch

from multimodal_lm_eap_ig.graph import Graph

attribute_module = importlib.import_module("multimodal_lm_eap_ig.attribute")


class DummyModel:
    def __init__(self, *, requires_grad: bool):
        self.cfg = types.SimpleNamespace(
            use_attn_result=True,
            use_split_qkv_input=True,
            use_hook_mlp_in=True,
            n_key_value_heads=None,
            d_model=4,
        )
        self._param = torch.nn.Parameter(torch.tensor(1.0), requires_grad=requires_grad)

    def parameters(self):
        yield self._param


def _dummy_metric(logits, clean_logits, batch):
    del logits, clean_logits, batch
    return torch.tensor(0.0)


def _dummy_graph() -> Graph:
    return Graph.from_model(
        {
            "n_layers": 1,
            "n_heads": 1,
            "parallel_attn_mlp": True,
            "d_model": 4,
        }
    )


def test_attribute_eap_raises_when_model_params_are_frozen():
    model = DummyModel(requires_grad=False)
    graph = _dummy_graph()

    with pytest.raises(RuntimeError, match="requires_grad=True"):
        attribute_module.attribute(
            model=model,
            graph=graph,
            batches=[],
            metric=_dummy_metric,
            method="EAP",
        )


def test_attribute_eap_raises_under_no_grad_context():
    model = DummyModel(requires_grad=True)
    graph = _dummy_graph()

    with torch.no_grad():
        with pytest.raises(RuntimeError, match="torch.no_grad\\(\\)|torch.inference_mode\\(\\)"):
            attribute_module.attribute(
                model=model,
                graph=graph,
                batches=[],
                metric=_dummy_metric,
                method="EAP",
            )


def test_attribute_exact_is_not_blocked_by_grad_guard(monkeypatch):
    model = DummyModel(requires_grad=False)
    graph = _dummy_graph()
    called = {"value": False}

    def fake_get_scores_exact(*args, **kwargs):
        del args, kwargs
        called["value"] = True
        return torch.ones_like(graph.scores)

    monkeypatch.setattr(attribute_module, "get_scores_exact", fake_get_scores_exact)

    scores = attribute_module.attribute(
        model=model,
        graph=graph,
        batches=[],
        metric=_dummy_metric,
        method="exact",
    )

    assert called["value"]
    assert torch.all(scores == 1)


def test_attribute_smoke_is_not_blocked_by_grad_guard(monkeypatch):
    model = DummyModel(requires_grad=False)
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
            graph=graph,
            batches=[],
            metric=_dummy_metric,
            method="smoke",
        )

    assert called["value"]
    assert torch.all(scores == 0)
