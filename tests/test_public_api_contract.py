import importlib

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


def test_removed_deprecated_top_level_helpers_are_not_exported():
    removed = {
        "get_real_edge_scores",
        "get_scores_clean_corrupted",
        "get_scores_eap",
        "get_scores_eap_ig",
        "get_scores_exact",
        "get_scores_ig_activations",
        "get_scores_smoke",
        "build_default_llava_processor",
        "prepare_llava_token_pair_batch",
    }
    assert removed.isdisjoint(set(pkg.__all__))
    for symbol in removed:
        assert not hasattr(pkg, symbol)


def test_internal_modules_are_marked_internal():
    assert "Internal" in (attribute_module.__doc__ or "")
    assert "Internal" in (evaluate_module.__doc__ or "")
    assert "Internal" in (utils_module.__doc__ or "")
