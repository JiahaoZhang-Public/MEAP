import json
from pathlib import Path

import scripts.route_graph_matrix as route_graph_matrix


def test_parse_model_list_handles_empty_entries():
    models = route_graph_matrix.parse_model_list("gpt2, ,Qwen/Qwen2-0.5B,,")
    assert models == ["gpt2", "Qwen/Qwen2-0.5B"]


def test_default_model_lists_use_core_tier_only():
    assert "google/gemma-2-2b" not in route_graph_matrix.DEFAULT_TEXT_MODELS
    assert "microsoft/Phi-3-mini-4k-instruct" not in route_graph_matrix.DEFAULT_TEXT_MODELS
    assert "Qwen/Qwen2-Audio-7B" not in route_graph_matrix.DEFAULT_MULTIMODAL_MODELS


def test_classify_error_categories():
    assert route_graph_matrix.classify_error(RuntimeError("CUDA out of memory")) == "oom"
    assert route_graph_matrix.classify_error(RuntimeError("401 Unauthorized")) == "auth"
    assert (
        route_graph_matrix.classify_error(ValueError("Unsupported HF architecture for HFLLMBackend"))
        == "unsupported_arch"
    )


def test_run_matrix_schema_with_stubbed_model_runners(monkeypatch):
    def fake_text(*args, **kwargs):
        del args, kwargs
        return route_graph_matrix.RouteGraphRow(
            model_id="gpt2",
            modality="text",
            entrypoint="AttributionModel.from_pretrained",
            model_loader="AutoModel",
            adapter_name="gpt2_like",
            language_trunk_path="model.transformer",
            arch_kind="gpt2_like",
            status="pass",
            seconds=0.1,
            error_type="",
            error_message="",
            resolution_error_hint="selected=gpt2_like@model.transformer",
            graph_stats={"n_layers": 1, "n_heads": 1, "d_model": 1, "n_forward": 1, "n_backward": 1, "n_edges_total": 1},
            hook_stats={"required_count": 1, "supported_count": 1, "missing_count": 0, "missing_hooks": []},
            attribute_smoke={"status": "pass", "score_shape_0": 1, "score_shape_1": 1, "n_examples": 1},
        )

    def fake_multimodal(*args, **kwargs):
        del args, kwargs
        return route_graph_matrix.RouteGraphRow(
            model_id="Qwen/Qwen2-VL-2B",
            modality="multimodal",
            entrypoint="AttributionModel.from_model",
            model_loader="AutoModelForImageTextToText",
            adapter_name="llama_like",
            language_trunk_path="model.language_model.model",
            arch_kind="llama_like",
            status="fail",
            seconds=0.2,
            error_type="runtime",
            error_message="boom",
            resolution_error_hint="attempt=llama_like@model.model: missing ...",
            graph_stats=None,
            hook_stats=None,
            attribute_smoke={"status": "fail", "error_type": "runtime", "error_message": "boom"},
        )

    monkeypatch.setattr(route_graph_matrix, "run_text_route_graph_model", fake_text)
    monkeypatch.setattr(route_graph_matrix, "run_multimodal_route_graph_model", fake_multimodal)

    report = route_graph_matrix.run_matrix(
        text_models=["gpt2"],
        multimodal_models=["Qwen/Qwen2-VL-2B"],
        device="cpu",
        dtype=route_graph_matrix.to_dtype("float32"),
        hf_token=None,
        trust_remote_code=True,
        run_attribute_smoke=True,
        quiet=True,
    )

    assert report["all_passed"] is False
    assert len(report["results"]) == 2
    row = report["results"][0]
    assert set(row.keys()) == {
        "model_id",
        "modality",
        "entrypoint",
        "model_loader",
        "adapter_name",
        "language_trunk_path",
        "arch_kind",
        "status",
        "seconds",
        "error_type",
        "error_message",
        "resolution_error_hint",
        "graph_stats",
        "hook_stats",
        "attribute_smoke",
    }


def test_main_writes_report(monkeypatch, tmp_path):
    fake_report = {
        "all_passed": True,
        "results": [
            {
                "model_id": "gpt2",
                "modality": "text",
                "entrypoint": "AttributionModel.from_pretrained",
                "model_loader": "AutoModel",
                "adapter_name": "gpt2_like",
                "language_trunk_path": "model.transformer",
                "arch_kind": "gpt2_like",
                "status": "pass",
                "seconds": 0.0,
                "error_type": "",
                "error_message": "",
                "resolution_error_hint": "",
                "graph_stats": {"n_layers": 1, "n_heads": 1, "d_model": 1, "n_forward": 1, "n_backward": 1, "n_edges_total": 1},
                "hook_stats": {"required_count": 1, "supported_count": 1, "missing_count": 0, "missing_hooks": []},
                "attribute_smoke": {"status": "pass", "score_shape_0": 1, "score_shape_1": 1, "n_examples": 1},
            }
        ],
    }

    monkeypatch.setattr(route_graph_matrix, "run_matrix", lambda **kwargs: fake_report)
    out = tmp_path / "report.json"
    monkeypatch.setattr(
        route_graph_matrix,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "text_models": "gpt2",
                "multimodal_models": "",
                "device": "cpu",
                "dtype": "float32",
                "hf_token": None,
                "trust_remote_code": True,
                "run_attribute_smoke": True,
                "output": str(out),
                "quiet": True,
            },
        )(),
    )

    route_graph_matrix.main()

    loaded = json.loads(Path(out).read_text(encoding="utf-8"))
    assert loaded["report_type"] == "route_graph_matrix"
    assert loaded["schema_version"] == "1.0.0"
    assert "generated_at_utc" in loaded
    assert loaded["all_passed"] is True
    assert loaded["results"] == fake_report["results"]
