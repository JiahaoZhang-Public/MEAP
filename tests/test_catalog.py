from meap.catalog import list_official_models, list_supported_architectures


def test_supported_architectures_contains_core_routes():
    rows = list_supported_architectures()
    assert len(rows) >= 3
    names = {row["adapter_name"] for row in rows}
    assert {"gpt2_like", "opt_like", "llama_like"}.issubset(names)


def test_official_model_catalog_contains_v2_core_set():
    rows = list_official_models()
    by_id = {row["model_id"]: row for row in rows}
    expected = {
        "gpt2": "gpt2_like",
        "distilgpt2": "gpt2_like",
        "facebook/opt-125m": "opt_like",
        "Qwen/Qwen2-0.5B": "llama_like",
        "Qwen/Qwen2-VL-2B": "llama_like",
        "llava-hf/llava-1.5-7b-hf": "llama_like",
    }
    assert expected.keys() <= by_id.keys()
    for model_id, adapter_name in expected.items():
        assert by_id[model_id]["adapter_name"] == adapter_name
