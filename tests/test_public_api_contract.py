import importlib

import pytest

import meap as pkg

attribute_module = importlib.import_module("meap.attribute")
evaluate_module = importlib.import_module("meap.evaluate")
utils_module = importlib.import_module("meap.utils")


def test_stable_public_api_symbols_are_exported():
    expected = {
        "AttributionModel",
        "AttributionResult",
        "AttributionRunResult",
        "CircuitEdgeSummary",
        "CircuitRunResult",
        "PreparedInputLike",
        "RouteInfo",
        "TaskSpec",
        "attribute_from_dataloader",
        "discover_circuit",
        "evaluate_graph_from_dataloader",
        "evaluate_baseline_from_dataloader",
        "HFLLMBackend",
        "TLensBackend",
        "PreparedBatch",
        "RawPairBatch",
        "HFProcessorAdapter",
        "Graph",
        "register_architecture_adapter",
        "inspect_model_architecture",
        "list_supported_architectures",
        "list_official_models",
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
