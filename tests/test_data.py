import pytest
import torch

from multimodal_lm_eap_ig.batch import PreparedBatch, validate_prepared_batch


def _build_batch():
    clean_ids = torch.tensor([[1, 2, 3, 0], [1, 4, 0, 0]])
    corrupt_ids = torch.tensor([[1, 9, 3, 0], [1, 8, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    return PreparedBatch(
        clean_inputs={"input_ids": clean_ids, "attention_mask": mask},
        corrupt_inputs={"input_ids": corrupt_ids, "attention_mask": mask.clone()},
        labels=torch.tensor([0, 1]),
        input_lengths=mask.sum(dim=-1),
    )


def test_validate_prepared_batch_accepts_aligned_pair():
    batch = _build_batch()
    validate_prepared_batch(batch)


def test_validate_prepared_batch_rejects_mask_mismatch():
    batch = _build_batch()
    batch.corrupt_inputs["attention_mask"][0, 2] = 0

    with pytest.raises(ValueError, match="attention_mask semantics must match"):
        validate_prepared_batch(batch)


def test_validate_prepared_batch_rejects_modality_layout_mismatch():
    batch = _build_batch()
    batch.meta = {
        "clean_modality_spans": [[1], [1]],
        "corrupt_modality_spans": [[2], [1]],
    }

    with pytest.raises(ValueError, match="modality placeholder layout must match"):
        validate_prepared_batch(batch)
