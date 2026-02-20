import json
from pathlib import Path

import scripts.smoke_hf_matrix as smoke_matrix


def test_parse_model_list_handles_empty_entries():
    models = smoke_matrix.parse_model_list("gpt2, ,distilgpt2,,")
    assert models == ["gpt2", "distilgpt2"]


def test_classify_error_categories():
    assert smoke_matrix.classify_error(RuntimeError("CUDA out of memory")) == "oom"
    assert smoke_matrix.classify_error(RuntimeError("401 Unauthorized")) == "auth"
    assert smoke_matrix.classify_error(ValueError("Unsupported HF architecture")) == "unsupported_arch"
    assert (
        smoke_matrix.classify_error(ValueError("Image features and image tokens do not match"))
        == "placeholder_mismatch"
    )


def test_run_matrix_schema_with_stubbed_model_runners(monkeypatch):
    def fake_text(*args, **kwargs):
        del args, kwargs
        return smoke_matrix.SmokeRow(
            model_id="gpt2",
            modality="text",
            adapter_name="gpt2_like",
            backbone_path="model.transformer",
            arch_kind="gpt2_like",
            status="pass",
            seconds=0.1,
            error_type="",
            error_message="",
            resolution_error_hint="",
            graph_stats={"n_forward": 1, "n_backward": 1, "n_edges": 1, "score_shape_0": 1, "score_shape_1": 1},
        )

    def fake_mm(*args, **kwargs):
        del args, kwargs
        return smoke_matrix.SmokeRow(
            model_id="Qwen/Qwen2-VL-2B",
            modality="multimodal",
            adapter_name="llama_like",
            backbone_path="model.language_model.model",
            arch_kind="llama_like",
            status="fail",
            seconds=0.2,
            error_type="runtime",
            error_message="boom",
            resolution_error_hint="attempt=llama_like@model.model: missing ...",
            graph_stats=None,
        )

    monkeypatch.setattr(smoke_matrix, "run_text_smoke_model", fake_text)
    monkeypatch.setattr(smoke_matrix, "run_multimodal_smoke_model", fake_mm)

    report = smoke_matrix.run_matrix(
        text_models=["gpt2"],
        multimodal_models=["Qwen/Qwen2-VL-2B"],
        device="cpu",
        dtype=smoke_matrix.to_dtype("float32"),
        hf_token=None,
        audio_fallback_model="fixie-ai/ultravox-v0_5-llama-3_2-1b",
        quiet=True,
    )

    assert report["all_passed"] is False
    assert len(report["results"]) == 2
    row = report["results"][0]
    assert set(row.keys()) == {
        "model_id",
        "modality",
        "adapter_name",
        "backbone_path",
        "arch_kind",
        "status",
        "seconds",
        "error_type",
        "error_message",
        "resolution_error_hint",
        "graph_stats",
    }


def test_main_writes_report(monkeypatch, tmp_path):
    fake_report = {
        "all_passed": True,
        "results": [
            {
                "model_id": "gpt2",
                "modality": "text",
                "adapter_name": "gpt2_like",
                "backbone_path": "model.transformer",
                "arch_kind": "gpt2_like",
                "status": "pass",
                "seconds": 0.0,
                "error_type": "",
                "error_message": "",
                "resolution_error_hint": "",
                "graph_stats": {"n_forward": 1, "n_backward": 1, "n_edges": 1, "score_shape_0": 1, "score_shape_1": 1},
            }
        ],
    }

    monkeypatch.setattr(smoke_matrix, "run_matrix", lambda **kwargs: fake_report)
    out = tmp_path / "report.json"
    monkeypatch.setattr(
        smoke_matrix,
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
                "audio_fallback_model": "fixie-ai/ultravox-v0_5-llama-3_2-1b",
                "output": str(out),
                "quiet": True,
            },
        )(),
    )

    smoke_matrix.main()

    loaded = json.loads(Path(out).read_text(encoding="utf-8"))
    assert loaded == fake_report
