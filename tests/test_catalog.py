from meap.catalog import list_official_models, list_supported_architectures


def test_supported_architectures_contains_core_routes():
    rows = list_supported_architectures()
    assert len(rows) >= 6
    names = {row["adapter_name"] for row in rows}
    assert {
        "gpt2_like",
        "opt_like",
        "llama_like",
        "falcon_like",
        "gemma_like",
        "phi_like",
    }.issubset(names)


def test_official_model_catalog_contains_v150_expanded_set():
    rows = list_official_models()
    by_id = {row["model_id"]: row for row in rows}
    expected = {
        "gpt2": "gpt2_like",
        "distilgpt2": "gpt2_like",
        "facebook/opt-125m": "opt_like",
        "Qwen/Qwen2-0.5B": "llama_like",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "llama_like",
        "tiiuae/falcon-rw-1b": "falcon_like",
        "google/gemma-2-2b": "gemma_like",
        "microsoft/phi-2": "phi_like",
        "microsoft/Phi-3-mini-4k-instruct": "phi_like",
        "Qwen/Qwen2-VL-2B": "llama_like",
        "llava-hf/llava-1.5-7b-hf": "llama_like",
        "HuggingFaceTB/SmolVLM-Instruct": "llama_like",
        "fixie-ai/ultravox-v0_5-llama-3_2-1b": "llama_like",
        "Qwen/Qwen2-Audio-7B": "llama_like",
        "HuggingFaceM4/idefics2-8b": "llama_like",
    }
    assert expected.keys() <= by_id.keys()
    for model_id, adapter_name in expected.items():
        assert by_id[model_id]["adapter_name"] == adapter_name
