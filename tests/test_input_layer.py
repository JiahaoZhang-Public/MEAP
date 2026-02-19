import pytest
import torch

from multimodal_lm_eap_ig.batch import iter_prepared_batches
from multimodal_lm_eap_ig.preparer import HFProcessorAdapter


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    unk_token_id = 999

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<image>":
            return 42
        return self.unk_token_id


class DummyProcessor:
    def __init__(self):
        self.tokenizer = DummyTokenizer()

    def __call__(self, text=None, images=None, return_tensors="pt", padding=True, **kwargs):
        del kwargs
        assert return_tensors == "pt"
        assert padding is True
        if text is None:
            raise ValueError("text is required")

        seqs = []
        for item in text:
            words = item.split()
            token_ids = [1]
            if images is not None:
                token_ids.append(42)
            token_ids.extend([2] * len(words))
            seqs.append(token_ids)

        max_len = max(len(s) for s in seqs)
        padded = [s + [0] * (max_len - len(s)) for s in seqs]
        masks = [[1] * len(s) + [0] * (max_len - len(s)) for s in seqs]

        out = {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }
        if images is not None:
            out["pixel_values"] = torch.stack([torch.zeros(3, 2, 2) for _ in text], dim=0)
        return out


def test_hf_processor_adapter_prepares_pair_batch_with_image_spans():
    processor = DummyProcessor()
    adapter = HFProcessorAdapter(processor=processor)

    clean = [
        {"text": "alpha beta", "images": torch.zeros(3, 2, 2)},
        {"text": "gamma delta", "images": torch.zeros(3, 2, 2)},
    ]
    corrupt = [
        {"text": "omega beta", "images": torch.zeros(3, 2, 2)},
        {"text": "theta delta", "images": torch.zeros(3, 2, 2)},
    ]

    prepared = adapter.prepare_batch(clean, corrupt, labels=torch.tensor([0, 1]))

    assert prepared.batch_size == 2
    assert "pixel_values" in prepared.clean_inputs
    assert "pixel_values" in prepared.corrupt_inputs
    assert torch.equal(prepared.clean_inputs["attention_mask"], prepared.corrupt_inputs["attention_mask"])

    assert prepared.meta is not None
    assert "clean_modality_spans" in prepared.meta
    assert "corrupt_modality_spans" in prepared.meta
    assert prepared.meta["clean_modality_spans"] == prepared.meta["corrupt_modality_spans"]


def test_iter_prepared_batches_supports_dict_batches_with_adapter():
    processor = DummyProcessor()
    adapter = HFProcessorAdapter(processor=processor)

    raw_batch = {
        "clean": [{"text": "a b"}, {"text": "c d"}],
        "corrupt": [{"text": "x b"}, {"text": "y d"}],
        "labels": torch.tensor([1, 0]),
    }

    prepared = list(
        iter_prepared_batches(
            tokenization_model=None,
            batches=[raw_batch],
            pair_batch_preparer=adapter,
        )
    )

    assert len(prepared) == 1
    assert prepared[0].batch_size == 2


def test_iter_prepared_batches_raw_pair_requires_adapter():
    raw_batch = {
        "clean": [{"text": "a b"}],
        "corrupt": [{"text": "c d"}],
        "labels": torch.tensor([0]),
    }

    with pytest.raises(TypeError, match="pair_batch_preparer"):
        _ = list(iter_prepared_batches(tokenization_model=None, batches=[raw_batch]))
