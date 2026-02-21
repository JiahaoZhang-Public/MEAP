import json
from pathlib import Path

import scripts.test_stage_matrix as stage_matrix


def test_classify_parity_row_categories():
    assert stage_matrix._classify_parity_row({"status": "pass"}) == ""
    assert stage_matrix._classify_parity_row({"status": "skip"}) == "skip"
    assert stage_matrix._classify_parity_row({"status": "fail"}) == "parity_mismatch"
    assert (
        stage_matrix._classify_parity_row({"status": "error", "note": "CUDA out of memory"})
        == "oom"
    )


def test_stage_matrix_main_writes_unified_report(monkeypatch, tmp_path):
    out = tmp_path / "stage_matrix.json"

    def fake_run(name, command, *, output_path=""):
        del command
        if name.startswith("text_parity:"):
            payload = {
                "results": [
                    {
                        "method": "EAP",
                        "status": "pass",
                        "seconds": 1.0,
                        "note": "",
                        "skip_reason": "",
                        "vendor_vs_ours_hf": {"max_abs": 1e-6, "cosine": 1.0, "topk_overlap": 1.0},
                    }
                ]
            }
            Path(output_path).write_text(json.dumps(payload), encoding="utf-8")
            return stage_matrix.CommandResult(name=name, command=["cmd"], returncode=0, seconds=1.0, output_path=output_path)
        if name == "multimodal_smoke":
            payload = {
                "results": [
                    {
                        "model_id": "Qwen/Qwen2-VL-2B",
                        "status": "pass",
                        "seconds": 2.0,
                        "error_type": "",
                        "error_message": "",
                        "adapter_name": "llama_like",
                        "language_trunk_path": "model.language_model",
                        "arch_kind": "llama_like",
                        "resolution_error_hint": "",
                        "graph_stats": {"n_forward": 1, "n_backward": 1, "n_edges": 1},
                    }
                ]
            }
            Path(output_path).write_text(json.dumps(payload), encoding="utf-8")
            return stage_matrix.CommandResult(name=name, command=["cmd"], returncode=0, seconds=2.0, output_path=output_path)
        return stage_matrix.CommandResult(name=name, command=["cmd"], returncode=0, seconds=0.1, output_path=output_path)

    monkeypatch.setattr(stage_matrix, "_run", fake_run)
    monkeypatch.setattr(
        stage_matrix,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "output": str(out),
                "skip_pytest": True,
                "skip_ruff": True,
                "skip_text_parity": False,
                "skip_multimodal_smoke": False,
                "text_models": "gpt2",
                "multimodal_models": "Qwen/Qwen2-VL-2B",
                "parity_methods": "EAP",
                "device": "cpu",
                "dtype": "float32",
                "hf_token": None,
                "audio_fallback_model": "fixie-ai/ultravox-v0_5-llama-3_2-1b",
                "strict": True,
                "n_samples": 1,
                "batch_size": 1,
                "ig_steps": 1,
                "max_exact_edges": 30000,
                "quiet": True,
            },
        )(),
    )

    stage_matrix.main()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["report_type"] == "stage_matrix"
    assert payload["schema_version"] == "1.0.0"
    assert "generated_at_utc" in payload
    assert payload["all_passed"] is True
    assert len(payload["model_level"]) == 2
    assert len(payload["method_level"]) == 2
    assert payload["error_summary"] == {}
