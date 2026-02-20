import pytest
from transformers import LlamaConfig, LlamaForCausalLM

from multimodal_lm_eap_ig.backend import (
    HFLLMBackend,
    get_registered_architecture_adapters,
    inspect_model_architecture,
    register_architecture_adapter,
    reset_architecture_adapter_registry,
)
from multimodal_lm_eap_ig.backend.adapters.llama_like import LlamaLikeAdapter


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
    assert isinstance(report["matches"], list)
    assert report["selected"] is not None
    assert report["selected"]["adapter"] in {"llama_like", "priority_llama"}
