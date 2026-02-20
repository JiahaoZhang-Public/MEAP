import torch
from transformers import MptConfig, MptForCausalLM

from meap.attribute import attribute
from meap.backend import HFLLMBackend
from meap.batch import PreparedBatch
from meap.graph import Graph


def _tiny_mpt_lm() -> MptForCausalLM:
    cfg = MptConfig(
        d_model=64,
        n_layers=1,
        n_heads=4,
        vocab_size=128,
        max_seq_len=32,
        attn_config={"attn_impl": "torch"},
    )
    return MptForCausalLM(cfg)


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


def test_hf_backend_mpt_adapter_and_smoke():
    model = _tiny_mpt_lm().eval()
    backend = HFLLMBackend(model)

    assert backend.adapter_name == "mpt_like"
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
