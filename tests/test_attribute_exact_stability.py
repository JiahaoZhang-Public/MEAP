import torch
from transformers import LlamaConfig, LlamaForCausalLM

from meap.attribute import attribute
from meap.backend import HFLLMBackend
from meap.batch import PreparedBatch
from meap.graph import Graph


def _metric(logits, clean_logits, batch):
    del clean_logits, batch
    return logits.sum()


def _tiny_model_and_backend():
    model = LlamaForCausalLM(
        LlamaConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=128,
            max_position_embeddings=32,
        )
    )
    backend = HFLLMBackend(model)
    return model, backend


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


def test_exact_supports_generator_batches_and_matches_list_result():
    model, backend = _tiny_model_and_backend()
    batch = _prepared_batch()

    graph_list = Graph.from_model(backend.config)
    scores_from_list = attribute(
        model=model,
        backend=backend,
        graph=graph_list,
        batches=[batch],
        metric=_metric,
        method="exact",
        quiet=True,
    ).clone()

    graph_gen = Graph.from_model(backend.config)
    scores_from_generator = attribute(
        model=model,
        backend=backend,
        graph=graph_gen,
        batches=(item for item in [batch]),
        metric=_metric,
        method="exact",
        quiet=True,
    ).clone()

    assert torch.allclose(scores_from_list, scores_from_generator, atol=1e-6, rtol=1e-6)


def test_exact_does_not_mutate_graph_structure_masks():
    model, backend = _tiny_model_and_backend()
    graph = Graph.from_model(backend.config)
    graph.reset(empty=True)
    in_graph_before = graph.in_graph.clone()

    _ = attribute(
        model=model,
        backend=backend,
        graph=graph,
        batches=[_prepared_batch()],
        metric=_metric,
        method="exact",
        quiet=True,
    )

    assert torch.equal(graph.in_graph, in_graph_before)
