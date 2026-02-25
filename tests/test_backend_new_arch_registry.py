from transformers import (
    Gemma2Config,
    Gemma2ForCausalLM,
    Phi3Config,
    Phi3ForCausalLM,
    PhiConfig,
    PhiForCausalLM,
)

from meap.backend import HFLLMBackend


def _tiny_gemma():
    cfg = Gemma2Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=512,
        max_position_embeddings=64,
    )
    return Gemma2ForCausalLM(cfg).eval()


def _tiny_phi2():
    cfg = PhiConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        vocab_size=512,
    )
    return PhiForCausalLM(cfg).eval()


def _tiny_phi3():
    cfg = Phi3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        vocab_size=33000,
    )
    return Phi3ForCausalLM(cfg).eval()


def test_gemma_resolves_to_gemma_like_adapter():
    backend = HFLLMBackend(_tiny_gemma())
    assert backend.adapter_name == "gemma_like"
    assert backend.arch_kind == "gemma_like"


def test_phi2_resolves_to_phi_like_adapter():
    backend = HFLLMBackend(_tiny_phi2())
    assert backend.adapter_name == "phi_like"
    assert backend.arch_kind == "phi_like"


def test_phi3_resolves_to_phi_like_adapter():
    backend = HFLLMBackend(_tiny_phi3())
    assert backend.adapter_name == "phi_like"
    assert backend.arch_kind == "phi_like"
