#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from typing import Any, Dict, List

import numpy as np
import torch
from transformer_lens import HookedTransformer
from transformers import AutoProcessor

from multimodal_lm_eap_ig import (
    DEFAULT_BACKBONE_MODEL_ID,
    DEFAULT_MULTIMODAL_MODEL_ID,
    Graph,
    attribute,
    get_real_edge_scores,
    prepare_llava_token_pair_batch,
)


@dataclass
class RunConfig:
    model_id: str
    backbone_model_id: str
    method: str
    ig_steps: int
    topk: int
    seed: int
    device: str
    dtype: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end attribution smoke test for LLaVA-1.5 + Llama-2-7B backbone.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MULTIMODAL_MODEL_ID)
    parser.add_argument("--backbone-model-id", default=DEFAULT_BACKBONE_MODEL_ID)
    parser.add_argument(
        "--method",
        default="EAP",
        choices=["EAP", "EAP-IG-inputs", "clean-corrupted", "EAP-IG-activations", "exact"],
        help="Attribution method.",
    )
    parser.add_argument("--ig-steps", type=int, default=4)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN"),
        help="HF token; defaults to HUGGINGFACE_HUB_TOKEN/HF_TOKEN if set.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def dtype_from_name(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def build_demo_images() -> List[np.ndarray]:
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[48:176, 48:176, :] = 255
    return [img]


def build_demo_prompts() -> Dict[str, List[str]]:
    return {
        "clean": ["USER: <image>\nQuestion: What color is the square?\nASSISTANT:"],
        "corrupt": ["USER: <image>\nQuestion: What animal is shown?\nASSISTANT:"],
    }


def load_tlens_backbone(model_id: str, device: str, dtype: torch.dtype) -> HookedTransformer:
    try:
        model = HookedTransformer.from_pretrained(
            model_id,
            device=device,
            dtype=dtype,
            default_padding_side="right",
        )
    except Exception:
        model = HookedTransformer.from_pretrained_no_processing(
            model_id,
            device=device,
            dtype=dtype,
            default_padding_side="right",
        )

    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    if model.cfg.n_key_value_heads is not None:
        model.cfg.ungroup_grouped_query_attention = True

    return model


def make_metric_fn(tokenizer):
    target_token = " white"
    target_token_id = tokenizer.encode(target_token, add_special_tokens=False)
    if len(target_token_id) != 1:
        raise ValueError(f"Expected a single token id for target token '{target_token}'")
    target_id = int(target_token_id[0])

    def metric(logits: torch.Tensor, clean_logits: torch.Tensor, batch) -> torch.Tensor:
        del clean_logits
        last_positions = (batch.input_lengths - 1).to(device=logits.device)
        final_logits = logits[torch.arange(logits.size(0), device=logits.device), last_positions]
        return final_logits[:, target_id].sum()

    return metric


def hash_inputs(prompts: Dict[str, List[str]], images: List[np.ndarray]) -> str:
    m = hashlib.sha256()
    m.update(json.dumps(prompts, sort_keys=True).encode("utf-8"))
    for img in images:
        m.update(img.tobytes())
    return m.hexdigest()


def main() -> None:
    args = parse_args()
    if args.ig_steps <= 0:
        raise ValueError("--ig-steps must be > 0")

    set_seed(args.seed)

    run_config = RunConfig(
        model_id=args.model_id,
        backbone_model_id=args.backbone_model_id,
        method=args.method,
        ig_steps=args.ig_steps,
        topk=args.topk,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
    )

    processor_kwargs: Dict[str, Any] = {}
    if args.hf_token:
        processor_kwargs["token"] = args.hf_token

    print(f"Loading processor: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id, **processor_kwargs)

    images = build_demo_images()
    prompts = build_demo_prompts()

    prepared_batch = prepare_llava_token_pair_batch(
        processor=processor,
        clean_prompts=prompts["clean"],
        corrupt_prompts=prompts["corrupt"],
        images=images,
        labels=None,
        model_id=args.model_id,
        backbone_model_id=args.backbone_model_id,
    )

    print(f"Loading TransformerLens backbone: {args.backbone_model_id}")
    dtype = dtype_from_name(args.dtype)
    model = load_tlens_backbone(args.backbone_model_id, args.device, dtype)

    graph = Graph.from_model(model)
    metric = make_metric_fn(model.tokenizer)

    print(f"Running attribution method={args.method}")
    scores = attribute(
        model=model,
        graph=graph,
        batches=[prepared_batch],
        metric=metric,
        method=args.method,
        ig_steps=args.ig_steps,
        quiet=args.quiet,
    )

    edge_indices, edge_scores = get_real_edge_scores(graph, scores=scores)
    abs_scores = edge_scores.abs()
    k = min(args.topk, abs_scores.numel())
    top_vals, top_pos = torch.topk(abs_scores, k=k)

    print("Top edges by |score|:")
    for rank, pos in enumerate(top_pos.tolist(), start=1):
        src_idx = int(edge_indices[pos, 0].item())
        dst_idx = int(edge_indices[pos, 1].item())
        signed_score = float(edge_scores[pos].item())
        print(f"{rank:02d}. src={src_idx} dst={dst_idx} score={signed_score:.6f}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = Path(args.output_dir) / run_id
    report_dir.mkdir(parents=True, exist_ok=True)

    torch.save(scores.detach().cpu(), report_dir / "scores.pt")

    metadata = {
        "git_commit": get_git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "config": asdict(run_config),
        "config_hash": hashlib.sha256(json.dumps(asdict(run_config), sort_keys=True).encode("utf-8")).hexdigest(),
        "seed": args.seed,
        "input_data_fingerprint": hash_inputs(prompts, images),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "transformer_lens": __import__("transformer_lens").__version__,
        },
        "artifacts": {
            "scores": str(report_dir / "scores.pt"),
        },
    }

    with (report_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved scores and metadata under: {report_dir}")


if __name__ == "__main__":
    main()
