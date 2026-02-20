from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

import torch
from torch import Tensor


@dataclass
class PreparedBatch:
    """Prepared clean/corrupt pair consumed by attribution/evaluation."""

    clean_inputs: Dict[str, Tensor]
    corrupt_inputs: Dict[str, Tensor]
    labels: Any
    input_lengths: Tensor
    meta: Optional[Dict[str, Any]] = None

    @property
    def batch_size(self) -> int:
        return int(self.input_lengths.shape[0])


LegacyTextBatch = Tuple[Sequence[str], Sequence[str], Any]


@dataclass
class RawPairBatch:
    """Raw clean/corrupt batch for processor-based preparation."""

    clean: Any
    corrupt: Any
    labels: Any
    meta: Optional[Dict[str, Any]] = None


@dataclass
class DictPairBatch:
    """Dict-style alias for dataloader output compatibility."""

    clean: Any
    corrupt: Any
    labels: Any
    meta: Optional[Dict[str, Any]] = None


class PairBatchPreparer(Protocol):
    def prepare_batch(
        self,
        clean_samples: Any,
        corrupt_samples: Any,
        labels: Any,
        *,
        meta: Optional[Dict[str, Any]] = None,
    ) -> PreparedBatch:
        ...


BatchLike = Union[PreparedBatch, RawPairBatch, DictPairBatch, Mapping[str, Any], Tuple[Any, Any, Any], Tuple[Any, Any, Any, Any]]


def _sequence_shape(inputs: Dict[str, Tensor]) -> Tuple[int, int]:
    if "input_ids" in inputs:
        input_ids = inputs["input_ids"]
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be rank-2 [batch, seq]")
        return int(input_ids.shape[0]), int(input_ids.shape[1])

    if "inputs_embeds" in inputs:
        inputs_embeds = inputs["inputs_embeds"]
        if inputs_embeds.ndim != 3:
            raise ValueError("inputs_embeds must be rank-3 [batch, seq, d_model]")
        return int(inputs_embeds.shape[0]), int(inputs_embeds.shape[1])

    raise ValueError("inputs must contain input_ids or inputs_embeds")


def validate_prepared_batch(batch: PreparedBatch) -> None:
    """Validate clean/corrupt alignment constraints and fail fast."""

    clean_b, clean_l = _sequence_shape(batch.clean_inputs)
    corrupt_b, corrupt_l = _sequence_shape(batch.corrupt_inputs)

    if clean_b != corrupt_b:
        raise ValueError(f"Batch size mismatch: clean={clean_b}, corrupt={corrupt_b}")

    if clean_l != corrupt_l:
        raise ValueError(f"Sequence length mismatch: clean={clean_l}, corrupt={corrupt_l}")

    clean_mask = batch.clean_inputs.get("attention_mask")
    corrupt_mask = batch.corrupt_inputs.get("attention_mask")

    if (clean_mask is None) != (corrupt_mask is None):
        raise ValueError("attention_mask must be present on both clean and corrupt inputs")

    if clean_mask is not None:
        if clean_mask.shape != corrupt_mask.shape:
            raise ValueError(
                "attention_mask shape mismatch: "
                f"clean={tuple(clean_mask.shape)}, corrupt={tuple(corrupt_mask.shape)}"
            )
        if not torch.equal(clean_mask, corrupt_mask):
            raise ValueError("attention_mask semantics must match between clean and corrupt")

        expected_lengths = clean_mask.sum(dim=-1).to(batch.input_lengths.device)
        if batch.input_lengths.ndim != 1:
            raise ValueError("input_lengths must be rank-1 [batch]")
        if batch.input_lengths.shape[0] != clean_b:
            raise ValueError(
                "input_lengths batch mismatch: "
                f"input_lengths={batch.input_lengths.shape[0]}, clean={clean_b}"
            )
        if not torch.equal(expected_lengths, batch.input_lengths):
            raise ValueError("input_lengths must match attention_mask.sum(-1)")

    if batch.meta is None:
        return

    clean_spans = batch.meta.get("clean_modality_spans")
    corrupt_spans = batch.meta.get("corrupt_modality_spans")
    if clean_spans is not None and corrupt_spans is not None and clean_spans != corrupt_spans:
        raise ValueError("modality placeholder layout must match between clean and corrupt")


def _require_text_tokenizer_model(model: Any) -> None:
    if not hasattr(model, "to_tokens"):
        raise TypeError("Legacy text tuple batches require a model/backend exposing to_tokens(...)")
    if not hasattr(model, "tokenizer"):
        raise TypeError("Legacy text tuple batches require a model/backend exposing tokenizer")


def _build_attention_mask_from_tokens(tokens: Tensor, tokenizer) -> Tensor:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        return torch.ones_like(tokens, dtype=torch.long)
    return (tokens != int(pad_token_id)).to(dtype=torch.long)


def _tokenize_text_batch(
    model: Any,
    inputs: Sequence[str],
    *,
    max_length: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    _require_text_tokenizer_model(model)

    old_n_ctx = None
    if max_length is not None and hasattr(model, "cfg") and hasattr(model.cfg, "n_ctx"):
        old_n_ctx = model.cfg.n_ctx
        model.cfg.n_ctx = max_length

    tokens = model.to_tokens(
        list(inputs),
        prepend_bos=True,
        padding_side="right",
        truncate=(max_length is not None),
    )

    if old_n_ctx is not None:
        model.cfg.n_ctx = old_n_ctx

    attention_mask = None
    try:
        # Match vendor/eap-ig semantics exactly for BOS/padding handling.
        from transformer_lens.utils import get_attention_mask

        attention_mask = get_attention_mask(model.tokenizer, tokens, prepend_bos=True)
    except Exception:
        attention_mask = _build_attention_mask_from_tokens(tokens, model.tokenizer)
    return tokens, attention_mask


def _pad_tokens_and_mask(
    tokens: Tensor,
    attention_mask: Tensor,
    *,
    target_length: int,
    pad_token_id: int,
) -> Tuple[Tensor, Tensor]:
    pad = target_length - int(tokens.shape[1])
    if pad <= 0:
        return tokens, attention_mask

    tokens = torch.nn.functional.pad(tokens, (0, pad), value=pad_token_id)
    attention_mask = torch.nn.functional.pad(attention_mask, (0, pad), value=0)
    return tokens, attention_mask


def text_batch_to_prepared_batch(
    model: Any,
    clean_text: Sequence[str],
    corrupt_text: Sequence[str],
    labels: Any,
    *,
    max_length: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> PreparedBatch:
    clean_tokens, clean_mask = _tokenize_text_batch(model, clean_text, max_length=max_length)
    corrupt_tokens, corrupt_mask = _tokenize_text_batch(model, corrupt_text, max_length=max_length)

    target_length = max(int(clean_tokens.shape[1]), int(corrupt_tokens.shape[1]))
    pad_token_id = getattr(model.tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(model.tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0

    clean_tokens, clean_mask = _pad_tokens_and_mask(
        clean_tokens,
        clean_mask,
        target_length=target_length,
        pad_token_id=int(pad_token_id),
    )
    corrupt_tokens, corrupt_mask = _pad_tokens_and_mask(
        corrupt_tokens,
        corrupt_mask,
        target_length=target_length,
        pad_token_id=int(pad_token_id),
    )

    input_lengths = clean_mask.sum(dim=-1)
    batch = PreparedBatch(
        clean_inputs={"input_ids": clean_tokens, "attention_mask": clean_mask},
        corrupt_inputs={"input_ids": corrupt_tokens, "attention_mask": corrupt_mask},
        labels=labels,
        input_lengths=input_lengths,
        meta=meta,
    )
    validate_prepared_batch(batch)
    return batch


def iter_prepared_batches(
    tokenization_model: Optional[Any],
    batches: Iterable[BatchLike],
    *,
    max_length: Optional[int] = None,
    pair_batch_preparer: Optional[PairBatchPreparer] = None,
) -> Iterator[PreparedBatch]:
    del tokenization_model
    if max_length is not None:
        raise ValueError(
            "max_length is no longer applied in iter_prepared_batches. "
            "Apply truncation in your processor/preparer before building PreparedBatch."
        )

    def _prepare_raw_pair(clean: Any, corrupt: Any, labels: Any, meta: Optional[Dict[str, Any]] = None):
        if pair_batch_preparer is None:
            raise TypeError(
                "Raw clean/corrupt batches require a pair_batch_preparer. "
                "Pass processor=... via API or provide your custom preparer via "
                "iter_prepared_batches(..., pair_batch_preparer=...)."
            )
        return pair_batch_preparer.prepare_batch(clean, corrupt, labels, meta=meta)

    for batch in batches:
        if isinstance(batch, PreparedBatch):
            validate_prepared_batch(batch)
            yield batch
            continue

        if isinstance(batch, RawPairBatch):
            yield _prepare_raw_pair(batch.clean, batch.corrupt, batch.labels, meta=batch.meta)
            continue

        if isinstance(batch, DictPairBatch):
            yield _prepare_raw_pair(batch.clean, batch.corrupt, batch.labels, meta=batch.meta)
            continue

        if isinstance(batch, Mapping):
            required = {"clean", "corrupt", "labels"}
            if required.issubset(batch.keys()):
                yield _prepare_raw_pair(
                    batch["clean"],
                    batch["corrupt"],
                    batch["labels"],
                    meta=batch.get("meta"),
                )
                continue
            raise TypeError(
                "Mapping batches must include keys {'clean', 'corrupt', 'labels'} "
                "(optional 'meta')."
            )

        if not isinstance(batch, tuple) or len(batch) not in (3, 4):
            raise TypeError(
                "Expected one of: PreparedBatch, RawPairBatch, dict(clean/corrupt/labels), "
                "or tuple(clean, corrupt, labels[, meta])."
            )

        clean_data, corrupt_data, labels = batch[:3]
        meta = batch[3] if len(batch) == 4 else None

        yield _prepare_raw_pair(clean_data, corrupt_data, labels, meta=meta)
