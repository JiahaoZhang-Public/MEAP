import pytest
import torch
from transformers import (
    Phi3Config,
    Phi3ForCausalLM,
    PhiConfig,
    PhiForCausalLM,
)

from meap.attribute import attribute
from meap.backend import HFLLMBackend
from meap.batch import PreparedBatch
from meap.graph import Graph


def _tiny_phi2_lm() -> PhiForCausalLM:
    cfg = PhiConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        vocab_size=512,
    )
    return PhiForCausalLM(cfg)


def _tiny_phi3_lm() -> Phi3ForCausalLM:
    cfg = Phi3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        vocab_size=33000,
    )
    return Phi3ForCausalLM(cfg)


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


@pytest.mark.parametrize(
    "factory",
    [_tiny_phi2_lm, _tiny_phi3_lm],
)
def test_hf_backend_phi_adapter_and_smoke(factory):
    model = factory().eval()
    backend = HFLLMBackend(model)

    assert backend.adapter_name == "phi_like"
    names = set(backend.supported_hook_names)
    assert "hook_embed" in names
    assert "blocks.0.attn.hook_result" in names
    assert "blocks.0.hook_q_input" in names
    assert "blocks.0.hook_k_input" in names
    assert "blocks.0.hook_v_input" in names

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
