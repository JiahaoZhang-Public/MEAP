#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from typing import Any, Dict, List, Optional

from huggingface_hub import hf_hub_download
import numpy as np
from safetensors.torch import load_file
import torch
from transformer_lens import HookedTransformer
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

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
    language_weights_source: str
    method: str
    ig_steps: int
    topk: int
    seed: int
    device: str
    dtype: str
    metric_token: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end attribution smoke test for LLaVA-1.5 + Llama-2-7B backbone.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MULTIMODAL_MODEL_ID)
    parser.add_argument("--backbone-model-id", default=DEFAULT_BACKBONE_MODEL_ID)
    parser.add_argument(
        "--language-weights-source",
        default="llava",
        choices=["llava", "backbone"],
        help="`llava`: use llava language_model weights; `backbone`: use backbone weights directly.",
    )
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
    parser.add_argument(
        "--metric-token",
        default=" white",
        help="Preferred token string for the scalar metric logit. Falls back automatically if not single-token.",
    )
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
    # Keep prompt templates token-length aligned (single-token swap: "white" <-> "black")
    # so clean/corrupt attention masks match for attribution.
    return {
        "clean": ["USER: <image>\nQuestion: Is the square white?\nASSISTANT:"],
        "corrupt": ["USER: <image>\nQuestion: Is the square black?\nASSISTANT:"],
    }


def load_llava_language_model_for_tlens(
    model_id: str,
    *,
    dtype: torch.dtype,
    hf_token: Optional[str] = None,
):
    """Load llava language_model.* weights into a standalone AutoModelForCausalLM."""
    kwargs: Dict[str, Any] = {}
    if hf_token:
        kwargs["token"] = hf_token

    cfg = AutoConfig.from_pretrained(model_id, **kwargs)
    if not hasattr(cfg, "text_config"):
        raise ValueError(f"{model_id} does not expose text_config; cannot extract language model")

    hf_lm = AutoModelForCausalLM.from_config(cfg.text_config, torch_dtype=dtype)
    expected_keys = set(hf_lm.state_dict().keys())

    index_path = hf_hub_download(model_id, "model.safetensors.index.json", **kwargs)
    with open(index_path, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    prefix = "language_model."
    shard_to_keys = defaultdict(list)
    for key, shard in weight_map.items():
        if key.startswith(prefix):
            shard_to_keys[shard].append(key)

    loaded_keys = set()
    unexpected_keys = set()

    for shard_name, shard_keys in shard_to_keys.items():
        shard_path = hf_hub_download(model_id, shard_name, **kwargs)
        shard_state = load_file(shard_path, device="cpu")
        remapped = {}
        for full_key in shard_keys:
            mapped_key = full_key[len(prefix) :]
            if mapped_key in expected_keys:
                remapped[mapped_key] = shard_state[full_key]
                loaded_keys.add(mapped_key)
            else:
                unexpected_keys.add(mapped_key)
        hf_lm.load_state_dict(remapped, strict=False)

    missing_keys = sorted(expected_keys - loaded_keys)
    if missing_keys:
        raise ValueError(
            "Missing language-model keys when extracting llava language weights. "
            f"Examples: {missing_keys[:8]}"
        )
    if unexpected_keys:
        print(f"Warning: ignored {len(unexpected_keys)} unexpected mapped keys from llava language weights")

    hf_lm.eval()
    return hf_lm


def load_tlens_backbone(
    model_id: str,
    *,
    tokenizer,
    device: str,
    dtype: torch.dtype,
    llava_model_id: str,
    language_weights_source: str,
    hf_token: Optional[str] = None,
) -> HookedTransformer:
    kwargs: Dict[str, Any] = {}
    if hf_token:
        kwargs["token"] = hf_token

    if language_weights_source == "llava":
        hf_lm = load_llava_language_model_for_tlens(
            llava_model_id,
            dtype=dtype,
            hf_token=hf_token,
        )
        model = HookedTransformer.from_pretrained(
            model_id,
            hf_model=hf_lm,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            fold_value_biases=False,
            default_padding_side="right",
            **kwargs,
        )
    else:
        try:
            model = HookedTransformer.from_pretrained(
                model_id,
                tokenizer=tokenizer,
                device=device,
                dtype=dtype,
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
                fold_value_biases=False,
                default_padding_side="right",
                **kwargs,
            )
        except Exception:
            model = HookedTransformer.from_pretrained_no_processing(
                model_id,
                device=device,
                dtype=dtype,
                default_padding_side="right",
                **kwargs,
            )

    model.eval()
    model.requires_grad_(False)

    try:
        model.cfg.use_attn_result = True
        model.cfg.use_split_qkv_input = True
        model.cfg.use_hook_mlp_in = True
        if model.cfg.n_key_value_heads is not None:
            model.cfg.ungroup_grouped_query_attention = True
    except Exception as exc:
        raise RuntimeError(
            "Failed to set TLens hook flags; this model may be incompatible with EAP-IG"
        ) from exc

    return model


def make_metric_fn(tokenizer):
    target_id, chosen_token = resolve_metric_token_id(tokenizer, preferred_token=" white")
    print(f"Metric token: {chosen_token!r} (id={target_id})")

    def metric(logits: torch.Tensor, clean_logits: torch.Tensor, batch) -> torch.Tensor:
        del clean_logits
        last_positions = (batch.input_lengths - 1).to(device=logits.device)
        final_logits = logits[torch.arange(logits.size(0), device=logits.device), last_positions]
        return final_logits[:, target_id].sum()

    return metric


def resolve_metric_token_id(tokenizer, preferred_token: str) -> tuple[int, str]:
    def encode_one(s: str) -> Optional[int]:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
        return None

    candidates = [
        preferred_token,
        " white",
        " black",
        " yes",
        " no",
        " true",
        " false",
        ".",
        "?",
    ]

    seen = set()
    for token in candidates:
        if token in seen:
            continue
        seen.add(token)
        token_id = encode_one(token)
        if token_id is not None:
            return token_id, token

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        print("Warning: no single-token metric candidate found; falling back to eos_token_id")
        return int(eos_id), "<eos>"

    raise ValueError("Unable to resolve any token id for metric.")


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
    if args.device == "cpu" and args.dtype == "float16":
        print("Warning: float16 on CPU is unsupported/unstable. Switching dtype to bfloat16.")
        args.dtype = "bfloat16"

    set_seed(args.seed)

    run_config = RunConfig(
        model_id=args.model_id,
        backbone_model_id=args.backbone_model_id,
        language_weights_source=args.language_weights_source,
        method=args.method,
        ig_steps=args.ig_steps,
        topk=args.topk,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        metric_token=args.metric_token,
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

    print(
        f"Loading TransformerLens backbone: {args.backbone_model_id} "
        f"(weights={args.language_weights_source})"
    )
    dtype = dtype_from_name(args.dtype)
    model = load_tlens_backbone(
        args.backbone_model_id,
        tokenizer=processor.tokenizer,
        device=args.device,
        dtype=dtype,
        llava_model_id=args.model_id,
        language_weights_source=args.language_weights_source,
        hf_token=args.hf_token,
    )

    graph = Graph.from_model(model)
    metric_token_id, metric_token_text = resolve_metric_token_id(
        model.tokenizer,
        preferred_token=args.metric_token,
    )
    print(f"Metric token: {metric_token_text!r} (id={metric_token_id})")

    def metric(logits: torch.Tensor, clean_logits: torch.Tensor, batch) -> torch.Tensor:
        del clean_logits
        last_positions = (batch.input_lengths - 1).to(device=logits.device)
        final_logits = logits[torch.arange(logits.size(0), device=logits.device), last_positions]
        return final_logits[:, metric_token_id].sum()

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
            "transformer_lens": importlib.metadata.version("transformer_lens"),
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
