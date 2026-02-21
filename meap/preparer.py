from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

import torch
from torch import Tensor

from .batch import PreparedBatch, validate_prepared_batch
from .config import DEFAULT_BACKBONE_MODEL_ID, DEFAULT_MULTIMODAL_MODEL_ID


def _batchfeature_to_inputs(batch_feature) -> Dict[str, Any]:
    return {k: v for k, v in batch_feature.items() if v is not None}


def _pad_2d(tensor: Tensor, target_length: int, pad_value: int) -> Tensor:
    pad = target_length - int(tensor.shape[1])
    if pad <= 0:
        return tensor
    return torch.nn.functional.pad(tensor, (0, pad), value=pad_value)


def _pad_pair_inputs(
    clean_inputs: Dict[str, Any],
    corrupt_inputs: Dict[str, Any],
    *,
    pad_token_id: int,
) -> None:
    if "input_ids" not in clean_inputs or "input_ids" not in corrupt_inputs:
        return
    if not torch.is_tensor(clean_inputs["input_ids"]) or not torch.is_tensor(corrupt_inputs["input_ids"]):
        return

    target_len = max(int(clean_inputs["input_ids"].shape[1]), int(corrupt_inputs["input_ids"].shape[1]))
    clean_inputs["input_ids"] = _pad_2d(clean_inputs["input_ids"], target_len, pad_token_id)
    corrupt_inputs["input_ids"] = _pad_2d(corrupt_inputs["input_ids"], target_len, pad_token_id)

    if "attention_mask" in clean_inputs and "attention_mask" in corrupt_inputs:
        clean_inputs["attention_mask"] = _pad_2d(clean_inputs["attention_mask"], target_len, 0)
        corrupt_inputs["attention_mask"] = _pad_2d(corrupt_inputs["attention_mask"], target_len, 0)


def _find_image_token_id(processor) -> Optional[int]:
    token_ids = _find_modality_token_ids(processor).get("image", [])
    if len(token_ids) == 0:
        return None
    return int(token_ids[0])


def _image_token_spans(input_ids: Tensor, image_token_id: int):
    spans = []
    for row in input_ids:
        positions = (row == image_token_id).nonzero(as_tuple=False).view(-1).tolist()
        spans.append(positions)
    return spans


def _resolve_pad_token_id(processor) -> int:
    pad_token_id = 0
    if hasattr(processor, "tokenizer"):
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = processor.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0
    return int(pad_token_id)


def _resolve_token_id_from_tokenizer(tokenizer, token: str) -> Optional[int]:
    try:
        token_id = tokenizer.convert_tokens_to_ids(token)
    except Exception:
        return None
    if token_id is None:
        return None
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None and token_id == unk_id:
        return None
    if isinstance(token_id, int) and token_id >= 0:
        return int(token_id)
    return None


def _find_modality_token_ids(processor) -> Dict[str, list[int]]:
    token_ids: Dict[str, set[int]] = {"image": set(), "video": set(), "audio": set()}

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        candidate_tokens = {
            "image": ("<image>", "<im_patch>", "<|image_pad|>", "<image_pad>"),
            "video": ("<video>", "<|video_pad|>", "<video_pad>"),
            "audio": ("<audio>", "<|audio_pad|>", "<audio_pad>"),
        }
        for modality, tokens in candidate_tokens.items():
            for token in tokens:
                token_id = _resolve_token_id_from_tokenizer(tokenizer, token)
                if token_id is not None:
                    token_ids[modality].add(token_id)

    for modality in ("image", "video", "audio"):
        attr_name = f"{modality}_token_id"
        for obj in (processor, tokenizer):
            if obj is None:
                continue
            value = getattr(obj, attr_name, None)
            if isinstance(value, int) and value >= 0:
                token_ids[modality].add(int(value))

    return {k: sorted(v) for k, v in token_ids.items()}


def _modality_token_count_per_row(input_ids: Tensor, token_ids: Sequence[int]) -> list[int]:
    if len(token_ids) == 0:
        return [0 for _ in range(int(input_ids.shape[0]))]
    token_tensor = torch.tensor(token_ids, device=input_ids.device, dtype=input_ids.dtype)
    # [batch, seq, n_tokens] -> [batch, seq] -> [batch]
    is_modality = (input_ids.unsqueeze(-1) == token_tensor.view(1, 1, -1)).any(dim=-1)
    return [int(x) for x in is_modality.sum(dim=-1).tolist()]


def _feature_present(inputs: Dict[str, Any], keys: Sequence[str]) -> bool:
    return any(key in inputs for key in keys)


def _precheck_modality_placeholders(
    clean_inputs: Dict[str, Any],
    corrupt_inputs: Dict[str, Any],
    *,
    processor,
) -> Dict[str, Dict[str, list[int]]]:
    if "input_ids" not in clean_inputs or "input_ids" not in corrupt_inputs:
        return {}
    if not torch.is_tensor(clean_inputs["input_ids"]) or not torch.is_tensor(corrupt_inputs["input_ids"]):
        return {}

    token_ids = _find_modality_token_ids(processor)
    modality_feature_keys = {
        "image": ("pixel_values", "image_grid_thw"),
        "video": ("pixel_values_videos", "video_grid_thw"),
        "audio": ("input_features", "input_values"),
    }

    meta_counts: Dict[str, Dict[str, list[int]]] = {}
    for modality, feature_keys in modality_feature_keys.items():
        if not (
            _feature_present(clean_inputs, feature_keys)
            or _feature_present(corrupt_inputs, feature_keys)
        ):
            continue

        ids = token_ids.get(modality, [])
        if len(ids) == 0:
            # Unknown token ids for this processor/model version; skip strict precheck.
            continue

        clean_counts = _modality_token_count_per_row(clean_inputs["input_ids"], ids)
        corrupt_counts = _modality_token_count_per_row(corrupt_inputs["input_ids"], ids)
        if clean_counts != corrupt_counts:
            raise ValueError(
                f"{modality} placeholder token counts must match between clean and corrupt: "
                f"clean={clean_counts}, corrupt={corrupt_counts}"
            )
        if any(count == 0 for count in clean_counts):
            raise ValueError(
                f"{modality} modality features present but placeholder tokens missing in input_ids: "
                f"counts={clean_counts}"
            )

        meta_counts[modality] = {
            "clean": clean_counts,
            "corrupt": corrupt_counts,
        }

    return meta_counts


def _canonicalize_processor_inputs(samples: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(samples)
    if "image" in normalized and "images" not in normalized:
        normalized["images"] = normalized.pop("image")
    if "video" in normalized and "videos" not in normalized:
        normalized["videos"] = normalized.pop("video")
    return normalized


def _normalize_modal_samples(samples: Any) -> Dict[str, Any]:
    if isinstance(samples, Mapping):
        return dict(samples)

    if isinstance(samples, Sequence) and not isinstance(samples, (str, bytes)):
        if len(samples) == 0:
            raise ValueError("samples sequence cannot be empty")

        first = samples[0]
        if isinstance(first, Mapping):
            keys = set(first.keys())
            for idx, sample in enumerate(samples):
                if not isinstance(sample, Mapping):
                    raise TypeError(f"samples[{idx}] must be a mapping, got {type(sample)}")
                if set(sample.keys()) != keys:
                    raise ValueError("all mapping samples must share the same keys")
            return {key: [sample[key] for sample in samples] for key in sorted(keys)}

        if all(isinstance(item, str) for item in samples):
            return {"text": list(samples)}

    raise TypeError(
        "samples must be one of: "
        "(1) batched mapping, "
        "(2) sequence of per-example mappings, or "
        "(3) sequence of text strings"
    )


def _prepare_processor_inputs(
    samples: Any,
    *,
    processor_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    processor_inputs = _canonicalize_processor_inputs(_normalize_modal_samples(samples))
    if processor_kwargs is None:
        return processor_inputs
    # Keep caller kwargs overriding normalized sample keys when explicitly set.
    merged = dict(processor_inputs)
    merged.update(processor_kwargs)
    return merged


@dataclass
class HFProcessorAdapter:
    """Convert clean/corrupt raw modality batches into a validated PreparedBatch."""

    processor: Any
    processor_kwargs: Optional[Dict[str, Any]] = None
    device: Optional[torch.device] = None

    def prepare_batch(
        self,
        clean_samples: Any,
        corrupt_samples: Any,
        labels: Any,
        *,
        meta: Optional[MutableMapping[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> PreparedBatch:
        call_kwargs = {"return_tensors": "pt", "padding": True}
        if self.processor_kwargs:
            call_kwargs.update(self.processor_kwargs)

        clean_inputs_for_processor = _prepare_processor_inputs(clean_samples, processor_kwargs=call_kwargs)
        corrupt_inputs_for_processor = _prepare_processor_inputs(
            corrupt_samples,
            processor_kwargs=call_kwargs,
        )

        clean_feature = self.processor(**clean_inputs_for_processor)
        corrupt_feature = self.processor(**corrupt_inputs_for_processor)

        clean_inputs = _batchfeature_to_inputs(clean_feature)
        corrupt_inputs = _batchfeature_to_inputs(corrupt_feature)

        pad_token_id = _resolve_pad_token_id(self.processor)
        _pad_pair_inputs(clean_inputs, corrupt_inputs, pad_token_id=pad_token_id)

        if (
            "attention_mask" not in clean_inputs
            and "input_ids" in clean_inputs
            and torch.is_tensor(clean_inputs["input_ids"])
        ):
            clean_inputs["attention_mask"] = (clean_inputs["input_ids"] != pad_token_id).long()
        if (
            "attention_mask" not in corrupt_inputs
            and "input_ids" in corrupt_inputs
            and torch.is_tensor(corrupt_inputs["input_ids"])
        ):
            corrupt_inputs["attention_mask"] = (corrupt_inputs["input_ids"] != pad_token_id).long()

        target_device = device if device is not None else self.device
        if target_device is not None:
            clean_inputs = {
                k: (v.to(target_device) if torch.is_tensor(v) else v) for k, v in clean_inputs.items()
            }
            corrupt_inputs = {
                k: (v.to(target_device) if torch.is_tensor(v) else v) for k, v in corrupt_inputs.items()
            }

        if "attention_mask" not in clean_inputs:
            raise ValueError("Prepared inputs must include attention_mask")
        if not torch.is_tensor(clean_inputs["attention_mask"]):
            raise ValueError("attention_mask in prepared inputs must be a tensor")

        meta_dict: Dict[str, Any] = dict(meta or {})
        placeholder_counts = _precheck_modality_placeholders(
            clean_inputs,
            corrupt_inputs,
            processor=self.processor,
        )
        if len(placeholder_counts) > 0:
            meta_dict.setdefault("modality_token_counts", placeholder_counts)
        image_token_id = _find_image_token_id(self.processor)
        if (
            image_token_id is not None
            and "input_ids" in clean_inputs
            and "input_ids" in corrupt_inputs
            and torch.is_tensor(clean_inputs["input_ids"])
            and torch.is_tensor(corrupt_inputs["input_ids"])
        ):
            meta_dict.setdefault("clean_modality_spans", _image_token_spans(clean_inputs["input_ids"], image_token_id))
            meta_dict.setdefault(
                "corrupt_modality_spans",
                _image_token_spans(corrupt_inputs["input_ids"], image_token_id),
            )

        input_lengths = clean_inputs["attention_mask"].sum(dim=-1)
        prepared = PreparedBatch(
            clean_inputs=clean_inputs,
            corrupt_inputs=corrupt_inputs,
            labels=labels,
            input_lengths=input_lengths,
            meta=(meta_dict or None),
        )
        validate_prepared_batch(prepared)
        return prepared


def prepare_pair_batch_with_processor(
    processor: Any,
    clean_samples: Any,
    corrupt_samples: Any,
    labels: Any,
    *,
    processor_kwargs: Optional[Dict[str, Any]] = None,
    meta: Optional[MutableMapping[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> PreparedBatch:
    adapter = HFProcessorAdapter(
        processor=processor,
        processor_kwargs=processor_kwargs,
        device=device,
    )
    return adapter.prepare_batch(
        clean_samples=clean_samples,
        corrupt_samples=corrupt_samples,
        labels=labels,
        meta=meta,
        device=device,
    )


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
    """Backward-compatible helper for LLaVA token-level clean/corrupt preparation."""

    meta: Dict[str, Any] = {
        "model_id": model_id,
        "backbone_model_id": backbone_model_id,
    }
    adapter = HFProcessorAdapter(processor=processor, device=device)
    return adapter.prepare_batch(
        clean_samples={"text": list(clean_prompts), "images": list(images)},
        corrupt_samples={"text": list(corrupt_prompts), "images": list(images)},
        labels=labels,
        meta=meta,
        device=device,
    )
