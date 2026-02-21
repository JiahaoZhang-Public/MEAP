import pytest
import torch
from transformers import AutoModel, LlamaConfig, LlamaForCausalLM

from meap import AttributionModel, TaskSpec
from meap.batch import PreparedBatch, RawPairBatch


def _tiny_model() -> LlamaForCausalLM:
    return LlamaForCausalLM(
        LlamaConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=64,
            max_position_embeddings=32,
        )
    )


def _prepared_batch() -> PreparedBatch:
    clean_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    corrupt_ids = torch.tensor([[1, 2, 9, 4]], dtype=torch.long)
    mask = torch.ones_like(clean_ids)
    return PreparedBatch(
        clean_inputs={"input_ids": clean_ids, "attention_mask": mask},
        corrupt_inputs={"input_ids": corrupt_ids, "attention_mask": mask.clone()},
        labels=torch.tensor([4]),
        input_lengths=mask.sum(dim=-1),
    )


def _metric(logits, clean_logits, batch):
    del clean_logits, batch
    return logits.sum()


def test_from_model_exposes_route_info():
    model = AttributionModel.from_model(_tiny_model())
    route = model.route_info
    assert route.adapter_name == "llama_like"
    assert route.arch_kind == "llama_like"
    assert route.language_trunk_path
    assert isinstance(route.diagnostics, dict)


def test_from_pretrained_forwards_model_kwargs_and_route_selectors(monkeypatch):
    tiny = _tiny_model()
    captured: dict[str, object] = {}

    def _fake_from_pretrained(cls, model_id_or_path, **kwargs):
        del cls
        captured["model_id_or_path"] = model_id_or_path
        captured["kwargs"] = kwargs
        return tiny

    monkeypatch.setattr(
        AutoModel,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )

    model = AttributionModel.from_pretrained(
        "dummy/model",
        model_kwargs={"trust_remote_code": True},
        adapter_name="llama_like",
        language_trunk_path="model.model",
        cache=False,
    )
    route = model.route_info
    assert captured["model_id_or_path"] == "dummy/model"
    assert captured["kwargs"] == {"trust_remote_code": True}
    assert route.adapter_name == "llama_like"
    assert route.language_trunk_path == "model.model"


def test_validate_batches_supports_nested_prepared_mapping():
    model = AttributionModel.from_model(_tiny_model())
    batch = _prepared_batch()
    payload = {
        "clean_inputs": dict(batch.clean_inputs),
        "corrupt_inputs": dict(batch.corrupt_inputs),
        "labels": batch.labels,
        # input_lengths intentionally omitted to test derivation.
    }
    out = model.validate_batches([payload])
    assert len(out) == 1
    assert torch.equal(out[0].input_lengths, batch.input_lengths)


def test_validate_batches_rejects_bad_schema():
    model = AttributionModel.from_model(_tiny_model())
    with pytest.raises(ValueError, match="must include keys"):
        _ = model.validate_batches([{"foo": "bar"}])


def test_validate_batches_rejects_raw_pair_batch():
    model = AttributionModel.from_model(_tiny_model())
    raw = RawPairBatch(clean={"text": ["a"]}, corrupt={"text": ["b"]}, labels=[1])
    with pytest.raises(TypeError, match="does not accept RawPairBatch"):
        _ = model.validate_batches([raw])


def test_attribute_accepts_prepared_batch_and_returns_scores():
    model = AttributionModel.from_model(_tiny_model())
    result = model.attribute(
        batches=[_prepared_batch()],
        method="smoke",
        metric=_metric,
    )
    assert result.scores.shape == (result.graph.n_forward, result.graph.n_backward)
    assert torch.all(result.scores == 0)


def test_attribute_requires_exactly_one_of_metric_or_task():
    model = AttributionModel.from_model(_tiny_model())
    batch = _prepared_batch()

    with pytest.raises(ValueError, match="exactly one of metric or task"):
        _ = model.attribute(batches=[batch], method="smoke")

    with pytest.raises(ValueError, match="exactly one of metric or task"):
        _ = model.attribute(
            batches=[batch],
            method="smoke",
            metric=_metric,
            task=TaskSpec(task="next_token", labels=[1]),
        )


def test_evaluate_graph_smoke_path():
    model = AttributionModel.from_model(_tiny_model())
    graph = model.build_graph()
    graph.reset(empty=False)
    value = model.evaluate_graph(
        graph=graph,
        batches=[_prepared_batch()],
        metric=_metric,
    )
    assert torch.is_tensor(value)
