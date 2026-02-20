import torch
from transformers import FalconConfig, FalconForCausalLM

from multimodal_lm_eap_ig.attribute import attribute
from multimodal_lm_eap_ig.backend import HFLLMBackend
from multimodal_lm_eap_ig.batch import PreparedBatch
from multimodal_lm_eap_ig.graph import Graph


def _tiny_falcon_lm() -> FalconForCausalLM:
    cfg = FalconConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        vocab_size=128,
        max_position_embeddings=32,
        multi_query=False,
        new_decoder_architecture=False,
    )
    return FalconForCausalLM(cfg)


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


def test_hf_backend_falcon_adapter_and_smoke():
    model = _tiny_falcon_lm().eval()
    backend = HFLLMBackend(model)

    assert backend.adapter_name == "falcon_like"
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
