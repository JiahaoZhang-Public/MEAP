import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

from multimodal_lm_eap_ig.attribute import attribute
from multimodal_lm_eap_ig.backend import HFLLMBackend
from multimodal_lm_eap_ig.batch import PreparedBatch
from multimodal_lm_eap_ig.graph import Graph


def _tiny_llama_lm() -> LlamaForCausalLM:
    cfg = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=32,
    )
    return LlamaForCausalLM(cfg)


def _prepared_batch() -> PreparedBatch:
    clean_ids = torch.tensor([[1, 2, 3, 4]])
    corrupt_ids = torch.tensor([[1, 2, 9, 4]])
    mask = torch.ones_like(clean_ids)
    return PreparedBatch(
        clean_inputs={"input_ids": clean_ids, "attention_mask": mask},
        corrupt_inputs={"input_ids": corrupt_ids, "attention_mask": mask.clone()},
        labels=None,
        input_lengths=mask.sum(dim=-1),
    )


def _metric(logits, clean_logits, batch):
    del clean_logits, batch
    return logits.sum()


def test_hf_backend_normalizes_llama_config_and_hook_scaffold():
    model = _tiny_llama_lm()
    backend = HFLLMBackend(model)

    assert backend.config.n_layers == 1
    assert backend.config.n_heads == 4
    assert backend.config.n_key_value_heads == 2
    assert backend.config.d_model == 16
    assert backend.config.parallel_attn_mlp is False

    names = set(backend.supported_hook_names)
    assert "hook_embed" in names
    assert "blocks.0.attn.hook_result" in names
    assert "blocks.0.hook_q_input" in names
    assert "blocks.0.hook_k_input" in names
    assert "blocks.0.hook_v_input" in names
    assert "blocks.0.hook_mlp_in" in names
    assert "blocks.0.hook_mlp_out" in names
    assert "blocks.0.hook_resid_post" in names


def test_hf_backend_supports_gpt2_and_builds_hook_scaffold():
    gpt2 = GPT2LMHeadModel(
        GPT2Config(n_layer=1, n_head=2, n_embd=16, n_positions=32, vocab_size=128)
    )
    backend = HFLLMBackend(gpt2)

    assert backend.config.n_layers == 1
    assert backend.config.n_heads == 2
    assert backend.config.d_model == 16
    assert backend.config.parallel_attn_mlp is False

    names = set(backend.supported_hook_names)
    assert "hook_embed" in names
    assert "blocks.0.attn.hook_result" in names
    assert "blocks.0.hook_q_input" in names
    assert "blocks.0.hook_k_input" in names
    assert "blocks.0.hook_v_input" in names
    assert "blocks.0.hook_mlp_in" in names
    assert "blocks.0.hook_mlp_out" in names
    assert "blocks.0.hook_resid_post" in names


def test_hf_backend_rejects_non_decoder_backbone():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)
            self.config = type("Cfg", (), {"hidden_size": 4, "num_hidden_layers": 1, "num_attention_heads": 1})()

    with pytest.raises(ValueError, match="Unsupported HF architecture"):
        HFLLMBackend(DummyModel())


def test_attribute_smoke_with_hf_backend_succeeds():
    model = _tiny_llama_lm()
    backend = HFLLMBackend(model)
    graph = Graph.from_model(backend.config)

    scores = attribute(
        model=model,
        backend=backend,
        graph=graph,
        batches=[_prepared_batch()],
        metric=_metric,
        method="smoke",
    )

    assert scores.shape == (graph.n_forward, graph.n_backward)
    assert torch.all(scores == 0)


def test_attribute_eap_ig_inputs_with_hf_backend_succeeds():
    model = _tiny_llama_lm()
    backend = HFLLMBackend(model)
    graph = Graph.from_model(backend.config)

    scores = attribute(
        model=model,
        backend=backend,
        graph=graph,
        batches=[_prepared_batch()],
        metric=_metric,
        method="EAP-IG-inputs",
        ig_steps=2,
    )

    assert scores.shape == (graph.n_forward, graph.n_backward)
    assert torch.isfinite(scores).all()


def test_attribute_non_smoke_with_hf_backend_raises_staged_error():
    model = _tiny_llama_lm()
    backend = HFLLMBackend(model)
    graph = Graph.from_model(backend.config)

    with pytest.raises(RuntimeError, match="supports `smoke` and `EAP-IG-inputs` only"):
        attribute(
            model=model,
            backend=backend,
            graph=graph,
            batches=[_prepared_batch()],
            metric=_metric,
            method="EAP",
        )


def test_attribute_requires_backend_for_non_tlens_model():
    graph = Graph.from_model(
        {
            "n_layers": 1,
            "n_heads": 1,
            "parallel_attn_mlp": True,
            "d_model": 16,
        }
    )
    with pytest.raises(ValueError, match="backend must be provided|A backend must be provided"):
        attribute(
            model=object(),
            graph=graph,
            batches=[_prepared_batch()],
            metric=_metric,
            method="smoke",
        )
