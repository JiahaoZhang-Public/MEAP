#!/usr/bin/env python3
"""Minimal prepared-input example using AttributionModel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from meap import AttributionModel, PreparedBatch  # noqa: E402
from meap.batch import validate_prepared_batch  # noqa: E402


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _metric(logits, clean_logits, batch):
    del clean_logits, batch
    return logits.sum()


def _prepare_batch(tokenizer, clean_text: str, corrupt_text: str, max_length: int = 128) -> PreparedBatch:
    clean = tokenizer(
        [clean_text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    corrupt = tokenizer(
        [corrupt_text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    clean_ids = clean["input_ids"]
    corrupt_ids = corrupt["input_ids"]
    max_len = max(int(clean_ids.shape[1]), int(corrupt_ids.shape[1]))

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    if clean_ids.shape[1] < max_len:
        clean_ids = torch.nn.functional.pad(clean_ids, (0, max_len - int(clean_ids.shape[1])), value=int(pad_id))
        clean_mask = torch.nn.functional.pad(clean["attention_mask"], (0, max_len - int(clean["attention_mask"].shape[1])), value=0)
    else:
        clean_mask = clean["attention_mask"]

    if corrupt_ids.shape[1] < max_len:
        corrupt_ids = torch.nn.functional.pad(corrupt_ids, (0, max_len - int(corrupt_ids.shape[1])), value=int(pad_id))
        corrupt_mask = torch.nn.functional.pad(corrupt["attention_mask"], (0, max_len - int(corrupt["attention_mask"].shape[1])), value=0)
    else:
        corrupt_mask = corrupt["attention_mask"]

    shared_mask = torch.minimum(clean_mask, corrupt_mask)
    batch = PreparedBatch(
        clean_inputs={"input_ids": clean_ids, "attention_mask": shared_mask},
        corrupt_inputs={"input_ids": corrupt_ids, "attention_mask": shared_mask.clone()},
        labels=torch.tensor([0], dtype=torch.long),
        input_lengths=shared_mask.sum(dim=-1),
    )
    validate_prepared_batch(batch)
    return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AttributionModel prepared-input text example.")
    parser.add_argument("--model-id", default="openai-community/gpt2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--method", default="smoke", choices=["smoke", "EAP", "EAP-IG-inputs", "clean-corrupted", "EAP-IG-activations", "exact"])
    parser.add_argument("--hf-token", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=args.hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=args.hf_token,
        dtype=_dtype(args.dtype),
    ).to(args.device)
    model.eval()

    attribution_model = AttributionModel.from_model(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        dtype=_dtype(args.dtype),
    )
    batch = _prepare_batch(
        tokenizer,
        clean_text="The capital of France is",
        corrupt_text="The capital of Germany is",
    )

    result = attribution_model.attribute(
        batches=[batch],
        metric=_metric,
        method=args.method,
    )
    summary = {
        "method": args.method,
        "route_info": {
            "adapter_name": result.route_info.adapter_name,
            "arch_kind": result.route_info.arch_kind,
            "language_trunk_path": result.route_info.language_trunk_path,
        },
        "graph": {
            "n_forward": result.graph.n_forward,
            "n_backward": result.graph.n_backward,
            "n_edges": len(result.graph.edges),
        },
        "scores": {
            "shape": [int(result.scores.shape[0]), int(result.scores.shape[1])],
            "mean": float(result.scores.mean().item()),
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
