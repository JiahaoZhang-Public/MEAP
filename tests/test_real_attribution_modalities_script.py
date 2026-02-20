import json

import scripts.real_attribution_modalities as script


def test_parse_modalities_rejects_invalid_value():
    try:
        script.parse_modalities("text,video")
        raise AssertionError("Expected ValueError for unsupported modality")
    except ValueError as exc:
        assert "Unsupported modality values" in str(exc)


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["real_attribution_modalities.py"],
    )
    args = script.parse_args()
    assert args.method == "EAP"
    assert args.topn == 200


def test_classify_error_for_missing_pygraphviz():
    exc = ModuleNotFoundError("No module named 'pygraphviz'")
    assert script.classify_error(exc) == "dependency_missing"


def test_main_writes_report_with_stubbed_runners(monkeypatch, tmp_path):
    out_dir = tmp_path / "reports"

    monkeypatch.setattr(
        script,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "modalities": "text,image,audio",
                "method": "EAP",
                "device": "cpu",
                "dtype": "float32",
                "topn": 200,
                "output_dir": str(out_dir),
                "hf_token": None,
                "text_model": "distilgpt2",
                "image_model": "Qwen/Qwen2-VL-2B",
                "audio_model": "fixie-ai/ultravox-v0_5-llama-3_2-1b",
                "image_url": script.DEFAULT_IMAGE_URL,
                "audio_url": script.DEFAULT_AUDIO_URL,
                "image_path": "",
                "audio_path": "",
                "quiet": True,
            },
        )(),
    )

    def fake_run(modalities, ctx, quiet=False):
        del ctx, quiet
        assert list(modalities) == ["text", "image", "audio"]
        return [
            script.ModalityResult(
                modality="text",
                model_id="distilgpt2",
                method="EAP",
                status="pass",
                seconds=1.0,
                error_type="",
                error_message="",
                graph_stats={
                    "n_forward": 1,
                    "n_backward": 1,
                    "n_edges_total": 1,
                    "n_edges_in_topn_graph": 1,
                    "score_shape_0": 1,
                    "score_shape_1": 1,
                },
                artifacts={"topn_png": "a.png"},
            ),
            script.ModalityResult(
                modality="image",
                model_id="Qwen/Qwen2-VL-2B",
                method="EAP",
                status="fail",
                seconds=2.0,
                error_type="dependency_missing",
                error_message="pygraphviz missing",
                graph_stats=None,
                artifacts={},
            ),
        ]

    monkeypatch.setattr(script, "run_modalities", fake_run)

    try:
        script.main()
        raise AssertionError("Expected SystemExit because all_passed should be false")
    except SystemExit as exc:
        assert exc.code == 1

    payload = json.loads((out_dir / "real_attribution_modalities.json").read_text(encoding="utf-8"))
    assert payload["report_type"] == "real_attribution_modalities"
    assert payload["schema_version"] == "1.0.0"
    assert payload["all_passed"] is False
    assert set(payload["results"][0].keys()) == {
        "modality",
        "model_id",
        "method",
        "status",
        "seconds",
        "error_type",
        "error_message",
        "graph_stats",
        "artifacts",
    }
