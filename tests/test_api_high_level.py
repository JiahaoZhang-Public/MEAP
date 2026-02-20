import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from meap.api import (
    attribute_from_dataloader,
    evaluate_baseline_from_dataloader,
    evaluate_graph_from_dataloader,
)
from meap.backend import HFLLMBackend
from meap.graph import Graph
from meap.preparer import HFProcessorAdapter


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<image>":
            return 42
        return 0


class DummyProcessor:
    def __init__(self):
        self.tokenizer = DummyTokenizer()

    def __call__(self, text=None, return_tensors="pt", padding=True, **kwargs):
        del kwargs
        assert return_tensors == "pt"
        assert padding is True
        seqs = []
        for item in text:
            token_ids = [1] + [2] * len(item.split())
            seqs.append(token_ids)

        max_len = max(len(s) for s in seqs)
        padded = [s + [0] * (max_len - len(s)) for s in seqs]
        masks = [[1] * len(s) + [0] * (max_len - len(s)) for s in seqs]
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


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
            vocab_size=64,
            max_position_embeddings=32,
        )
    )
    backend = HFLLMBackend(model)
    return model, backend


def test_attribute_from_dataloader_supports_raw_batches_with_processor():
    model, backend = _tiny_model_and_backend()

    dataloader = [
        {
            "clean": [{"text": "alpha beta"}, {"text": "gamma delta"}],
            "corrupt": [{"text": "omega beta"}, {"text": "theta delta"}],
            "labels": torch.tensor([0, 1]),
        }
    ]

    result = attribute_from_dataloader(
        model=model,
        backend=backend,
        dataloader=dataloader,
        metric=_metric,
        processor=DummyProcessor(),
        method="smoke",
    )

    assert result.scores.shape == (result.graph.n_forward, result.graph.n_backward)
    assert torch.all(result.scores == 0)


def test_attribute_from_dataloader_supports_raw_batches_with_custom_preparer():
    model, backend = _tiny_model_and_backend()

    dataloader = [
        {
            "clean": [{"text": "a b"}],
            "corrupt": [{"text": "c d"}],
            "labels": torch.tensor([0]),
        }
    ]
    result = attribute_from_dataloader(
        model=model,
        backend=backend,
        dataloader=dataloader,
        metric=_metric,
        pair_batch_preparer=HFProcessorAdapter(
            processor=DummyProcessor(),
            device=backend.config.device,
        ),
        method="smoke",
    )
    assert result.scores.shape == (result.graph.n_forward, result.graph.n_backward)


def test_attribute_from_dataloader_rejects_processor_and_preparer_together():
    model, backend = _tiny_model_and_backend()

    dataloader = [
        {
            "clean": [{"text": "a b"}],
            "corrupt": [{"text": "c d"}],
            "labels": torch.tensor([0]),
        }
    ]
    processor = DummyProcessor()

    with pytest.raises(ValueError, match="either processor or pair_batch_preparer"):
        _ = attribute_from_dataloader(
            model=model,
            backend=backend,
            dataloader=dataloader,
            metric=_metric,
            processor=processor,
            pair_batch_preparer=HFProcessorAdapter(
                processor=processor,
                device=backend.config.device,
            ),
            method="smoke",
        )


def test_attribute_from_dataloader_rejects_max_length_flag():
    model, backend = _tiny_model_and_backend()
    dataloader = [
        {
            "clean": [{"text": "a b"}],
            "corrupt": [{"text": "c d"}],
            "labels": torch.tensor([0]),
        }
    ]
    with pytest.raises(ValueError, match="max_length is not applied"):
        _ = attribute_from_dataloader(
            model=model,
            backend=backend,
            dataloader=dataloader,
            metric=_metric,
            processor=DummyProcessor(),
            max_length=16,
            method="smoke",
        )


def test_attribute_from_dataloader_requires_preparer_for_raw_batches():
    model, backend = _tiny_model_and_backend()
    dataloader = [
        {
            "clean": [{"text": "a b"}],
            "corrupt": [{"text": "c d"}],
            "labels": torch.tensor([0]),
        }
    ]
    with pytest.raises(TypeError, match="Raw clean/corrupt batches require a pair_batch_preparer"):
        _ = attribute_from_dataloader(
            model=model,
            backend=backend,
            dataloader=dataloader,
            metric=_metric,
            method="smoke",
        )


def test_attribute_from_dataloader_rejects_processor_kwargs_without_processor():
    model, backend = _tiny_model_and_backend()
    with pytest.raises(ValueError, match="processor_kwargs requires processor"):
        _ = attribute_from_dataloader(
            model=model,
            backend=backend,
            dataloader=[],
            metric=_metric,
            processor_kwargs={"truncation": True, "max_length": 16},
            method="smoke",
        )


def test_evaluate_graph_from_dataloader_rejects_max_length_flag():
    model, backend = _tiny_model_and_backend()
    graph = Graph.from_model(backend.config)
    dataloader = [
        {
            "clean": [{"text": "a b"}],
            "corrupt": [{"text": "c d"}],
            "labels": torch.tensor([0]),
        }
    ]
    with pytest.raises(ValueError, match="max_length is not applied"):
        _ = evaluate_graph_from_dataloader(
            model=model,
            graph=graph,
            backend=backend,
            dataloader=dataloader,
            metrics=_metric,
            processor=DummyProcessor(),
            max_length=8,
        )


def test_evaluate_baseline_from_dataloader_rejects_max_length_flag():
    model, backend = _tiny_model_and_backend()
    dataloader = [
        {
            "clean": [{"text": "a b"}],
            "corrupt": [{"text": "c d"}],
            "labels": torch.tensor([0]),
        }
    ]
    with pytest.raises(ValueError, match="max_length is not applied"):
        _ = evaluate_baseline_from_dataloader(
            model=model,
            backend=backend,
            dataloader=dataloader,
            metrics=_metric,
            processor=DummyProcessor(),
            max_length=8,
        )
