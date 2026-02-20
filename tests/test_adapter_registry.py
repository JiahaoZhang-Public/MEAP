import pytest
from transformers import LlamaConfig, LlamaForCausalLM

from meap.backend import (
    HFLLMBackend,
    get_registered_architecture_adapters,
    inspect_model_architecture,
    register_architecture_adapter,
    reset_architecture_adapter_registry,
)
from meap.backend.adapters.llama_like import LlamaLikeAdapter


class _PriorityLlamaAdapter(LlamaLikeAdapter):
    name = "priority_llama"


def _tiny_llama() -> LlamaForCausalLM:
    cfg = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        vocab_size=128,
        max_position_embeddings=32,
    )
    return LlamaForCausalLM(cfg)


@pytest.fixture(autouse=True)
def _reset_registry_fixture():
    reset_architecture_adapter_registry()
    yield
    reset_architecture_adapter_registry()


def test_register_architecture_adapter_prepend_priority():
    register_architecture_adapter(_PriorityLlamaAdapter, prepend=True)

    model = _tiny_llama()
    backend = HFLLMBackend(model)
    assert backend.adapter_name == "priority_llama"

    names = [cls.name for cls in get_registered_architecture_adapters()]
    assert names[0] == "priority_llama"

def test_register_architecture_adapter_replaces_same_name():
    register_architecture_adapter(_PriorityLlamaAdapter, prepend=True)
    register_architecture_adapter(_PriorityLlamaAdapter, prepend=False)

    names = [cls.name for cls in get_registered_architecture_adapters()]
    assert names.count("priority_llama") == 1

def test_inspect_model_architecture_contains_selection():
    model = _tiny_llama()
    report = inspect_model_architecture(model)

    assert "candidate_backbones" in report
    assert "adapters" in report
    assert isinstance(report["matches"], list)
    assert isinstance(report["adapter_attempts"], list)
    assert isinstance(report["errors"], list)
    assert report["selected"] is not None
    assert report["selected"]["adapter"] in {"llama_like", "priority_llama"}


def test_inspect_model_architecture_match_schema_is_stable():
    model = _tiny_llama()
    report = inspect_model_architecture(model)
    first = report["matches"][0]
    assert set(first.keys()) >= {
        "path",
        "adapter",
        "matched",
        "required_modules",
        "missing_modules",
        "error",
    }


def test_hf_resolution_is_deterministic_for_same_model():
    model = _tiny_llama()
    backend_a = HFLLMBackend(model)
    backend_b = HFLLMBackend(model)
    assert backend_a.adapter_name == backend_b.adapter_name
    assert backend_a.backbone_path == backend_b.backbone_path
    assert backend_a.arch_kind == backend_b.arch_kind
