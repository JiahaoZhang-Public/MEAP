import pytest
import torch

from meap.batch import iter_prepared_batches
from meap.preparer import HFProcessorAdapter


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    unk_token_id = 999

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<image>":
            return 42
        if token == "<|video_pad|>":
            return 43
        if token == "<|audio_pad|>":
            return 44
        return self.unk_token_id


class DummyProcessor:
    def __init__(self):
        self.tokenizer = DummyTokenizer()

    def __call__(
        self,
        text=None,
        images=None,
        videos=None,
        audio=None,
        return_tensors="pt",
        padding=True,
        **kwargs,
    ):
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
            if videos is not None:
                token_ids.append(43)
            if audio is not None:
                token_ids.append(44)
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
        if videos is not None:
            out["pixel_values_videos"] = torch.stack([torch.zeros(3, 2, 2) for _ in text], dim=0)
        if audio is not None:
            out["input_features"] = torch.stack([torch.zeros(2, 8) for _ in text], dim=0)
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


def test_iter_prepared_batches_text_tuple_requires_explicit_preparer():
    tuple_batch = (["a b"], ["c d"], torch.tensor([0]))
    with pytest.raises(TypeError, match="pair_batch_preparer"):
        _ = list(iter_prepared_batches(tokenization_model=object(), batches=[tuple_batch]))


def test_iter_prepared_batches_rejects_max_length_legacy_flag():
    raw_batch = {
        "clean": [{"text": "a b"}],
        "corrupt": [{"text": "c d"}],
        "labels": torch.tensor([0]),
    }
    with pytest.raises(ValueError, match="max_length is no longer applied"):
        _ = list(iter_prepared_batches(tokenization_model=None, batches=[raw_batch], max_length=16))


def test_hf_processor_adapter_supports_video_and_audio_alias_fields():
    processor = DummyProcessor()
    adapter = HFProcessorAdapter(processor=processor)

    clean = [{"text": "look", "image": torch.zeros(3, 2, 2), "video": torch.zeros(3, 2, 2), "audio": torch.zeros(2, 8)}]
    corrupt = [{"text": "look", "image": torch.zeros(3, 2, 2), "video": torch.zeros(3, 2, 2), "audio": torch.zeros(2, 8)}]
    prepared = adapter.prepare_batch(clean, corrupt, labels=torch.tensor([1]))

    assert "pixel_values" in prepared.clean_inputs
    assert "pixel_values_videos" in prepared.clean_inputs
    assert "input_features" in prepared.clean_inputs
    assert prepared.meta is not None
    assert "modality_token_counts" in prepared.meta


def test_hf_processor_adapter_precheck_rejects_mismatched_modality_tokens():
    processor = DummyProcessor()
    adapter = HFProcessorAdapter(processor=processor)

    clean = [{"text": "a b", "images": torch.zeros(3, 2, 2)}]
    corrupt = [{"text": "a b"}]
    with pytest.raises(ValueError, match="placeholder token counts must match"):
        _ = adapter.prepare_batch(clean, corrupt, labels=torch.tensor([0]))
