import importlib

import pytest

import multimodal_lm_eap_ig as pkg

attribute_module = importlib.import_module("multimodal_lm_eap_ig.attribute")
evaluate_module = importlib.import_module("multimodal_lm_eap_ig.evaluate")
utils_module = importlib.import_module("multimodal_lm_eap_ig.utils")


def test_stable_public_api_symbols_are_exported():
    expected = {
        "AttributionRunResult",
        "attribute_from_dataloader",
        "evaluate_graph_from_dataloader",
        "evaluate_baseline_from_dataloader",
        "HFLLMBackend",
        "TLensBackend",
        "PreparedBatch",
        "RawPairBatch",
        "DictPairBatch",
        "HFProcessorAdapter",
        "Graph",
        "register_architecture_adapter",
        "inspect_model_architecture",
        "resolve_backend",
    }
    assert expected.issubset(set(pkg.__all__))


def test_deprecated_top_level_score_helpers_emit_warning(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(pkg, "_get_scores_smoke", lambda *args, **kwargs: sentinel)

    with pytest.warns(DeprecationWarning, match="removed in 1.2.0"):
        result = pkg.get_scores_smoke("dummy")

    assert result is sentinel


def test_internal_modules_are_marked_internal():
    assert "Internal" in (attribute_module.__doc__ or "")
    assert "Internal" in (evaluate_module.__doc__ or "")
    assert "Internal" in (utils_module.__doc__ or "")
