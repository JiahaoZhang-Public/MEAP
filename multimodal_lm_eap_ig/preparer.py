from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
from torch import Tensor

from .batch import PreparedBatch, validate_prepared_batch
from .config import DEFAULT_BACKBONE_MODEL_ID, DEFAULT_MULTIMODAL_MODEL_ID


def _batchfeature_to_tensors(batch_feature) -> Dict[str, Tensor]:
    return {k: v for k, v in batch_feature.items() if isinstance(v, Tensor)}


def _pad_2d(tensor: Tensor, target_length: int, pad_value: int) -> Tensor:
    pad = target_length - int(tensor.shape[1])
    if pad <= 0:
        return tensor
    return torch.nn.functional.pad(tensor, (0, pad), value=pad_value)


def _pad_pair_inputs(
    clean_inputs: Dict[str, Tensor],
    corrupt_inputs: Dict[str, Tensor],
    *,
    pad_token_id: int,
) -> None:
    if "input_ids" not in clean_inputs or "input_ids" not in corrupt_inputs:
        return

    target_len = max(int(clean_inputs["input_ids"].shape[1]), int(corrupt_inputs["input_ids"].shape[1]))
    clean_inputs["input_ids"] = _pad_2d(clean_inputs["input_ids"], target_len, pad_token_id)
    corrupt_inputs["input_ids"] = _pad_2d(corrupt_inputs["input_ids"], target_len, pad_token_id)

    if "attention_mask" in clean_inputs and "attention_mask" in corrupt_inputs:
        clean_inputs["attention_mask"] = _pad_2d(clean_inputs["attention_mask"], target_len, 0)
        corrupt_inputs["attention_mask"] = _pad_2d(corrupt_inputs["attention_mask"], target_len, 0)


def _find_image_token_id(processor) -> Optional[int]:
    if not hasattr(processor, "tokenizer"):
        return None

    tokenizer = processor.tokenizer
    for token in ("<image>", "<im_patch>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            continue
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if unk_id is not None and token_id == unk_id:
            continue
        if isinstance(token_id, int) and token_id >= 0:
            return token_id
    return None


def _image_token_spans(input_ids: Tensor, image_token_id: int):
    spans = []
    for row in input_ids:
        positions = (row == image_token_id).nonzero(as_tuple=False).view(-1).tolist()
        spans.append(positions)
    return spans


def build_default_llava_processor(model_id: str = DEFAULT_MULTIMODAL_MODEL_ID):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_id)


def prepare_llava_token_pair_batch(
    processor,
    clean_prompts: Sequence[str],
    corrupt_prompts: Sequence[str],
    images: Sequence[Any],
    labels: Any,
    *,
    model_id: str = DEFAULT_MULTIMODAL_MODEL_ID,
    backbone_model_id: str = DEFAULT_BACKBONE_MODEL_ID,
    device: Optional[torch.device] = None,
) -> PreparedBatch:
    """Prepare a clean/corrupt LLaVA pair batch using HF AutoProcessor.

    This helper emits token-level prepared batches for the LLM-trunk attribution path.
    """

    clean_feature = processor(
        text=list(clean_prompts),
        images=list(images),
        return_tensors="pt",
        padding=True,
    )
    corrupt_feature = processor(
        text=list(corrupt_prompts),
        images=list(images),
        return_tensors="pt",
        padding=True,
    )

    clean_inputs = _batchfeature_to_tensors(clean_feature)
    corrupt_inputs = _batchfeature_to_tensors(corrupt_feature)

    pad_token_id = 0
    if hasattr(processor, "tokenizer"):
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = processor.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0

    _pad_pair_inputs(clean_inputs, corrupt_inputs, pad_token_id=int(pad_token_id))

    if "attention_mask" not in clean_inputs and "input_ids" in clean_inputs:
        clean_inputs["attention_mask"] = (clean_inputs["input_ids"] != int(pad_token_id)).long()
    if "attention_mask" not in corrupt_inputs and "input_ids" in corrupt_inputs:
        corrupt_inputs["attention_mask"] = (corrupt_inputs["input_ids"] != int(pad_token_id)).long()

    if device is not None:
        clean_inputs = {k: v.to(device) for k, v in clean_inputs.items()}
        corrupt_inputs = {k: v.to(device) for k, v in corrupt_inputs.items()}

    if "attention_mask" not in clean_inputs:
        raise ValueError("LLaVA prepared inputs must include attention_mask")

    meta: Dict[str, Any] = {
        "model_id": model_id,
        "backbone_model_id": backbone_model_id,
    }

    image_token_id = _find_image_token_id(processor)
    if image_token_id is not None and "input_ids" in clean_inputs and "input_ids" in corrupt_inputs:
        meta["clean_modality_spans"] = _image_token_spans(clean_inputs["input_ids"], image_token_id)
        meta["corrupt_modality_spans"] = _image_token_spans(corrupt_inputs["input_ids"], image_token_id)

    input_lengths = clean_inputs["attention_mask"].sum(dim=-1)

    prepared = PreparedBatch(
        clean_inputs=clean_inputs,
        corrupt_inputs=corrupt_inputs,
        labels=labels,
        input_lengths=input_lengths,
        meta=meta,
    )
    validate_prepared_batch(prepared)
    return prepared
