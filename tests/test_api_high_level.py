import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from multimodal_lm_eap_ig.api import attribute_from_dataloader
from multimodal_lm_eap_ig.backend import HFLLMBackend
from multimodal_lm_eap_ig.preparer import HFProcessorAdapter


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


def test_attribute_from_dataloader_supports_raw_batches_with_processor():
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
