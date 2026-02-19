#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from multimodal_lm_eap_ig import HFLLMBackend, attribute_from_dataloader


@dataclass
class RunConfig:
    model_id: str
    method: str
    device: str
    dtype: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen2-VL HF backend smoke attribution script.")
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-2B")
    parser.add_argument(
        "--method",
        default="smoke",
        choices=["smoke", "EAP-IG-inputs", "EAP", "clean-corrupted", "EAP-IG-activations", "exact"],
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output", default="reports/qwen2_vl_smoke.json")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _to_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _load_multimodal_model(model_id: str, *, dtype: torch.dtype, token: Optional[str]):
    kwargs: dict[str, Any] = {"torch_dtype": dtype, "trust_remote_code": True}
    if token:
        kwargs["token"] = token

    # Qwen2-VL may be registered under different auto classes depending on transformers version.
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return model


def _build_demo_image() -> Image.Image:
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[48:176, 48:176, :] = 255
    return Image.fromarray(img)


def _build_prompt(processor, question: str) -> str:
    if hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return f"<image>\n{question}"


def _metric(logits, clean_logits, batch):
    del clean_logits, batch
    return logits.sum()


def main() -> None:
    args = parse_args()
    dtype = _to_dtype(args.dtype)

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True, token=args.hf_token)
    model = _load_multimodal_model(args.model_id, dtype=dtype, token=args.hf_token)
    model = model.to(device=args.device)
    model.eval()

    backend = HFLLMBackend(model, tokenizer=getattr(processor, "tokenizer", None))

    image = _build_demo_image()
    clean_prompt = _build_prompt(processor, "Is the square white?")
    corrupt_prompt = _build_prompt(processor, "Is the square black?")

    dataloader = [
        {
            "clean": [{"text": clean_prompt, "images": image}],
            "corrupt": [{"text": corrupt_prompt, "images": image}],
            "labels": None,
        }
    ]

    result = attribute_from_dataloader(
        model=model,
        backend=backend,
        dataloader=dataloader,
        metric=_metric,
        processor=processor,
        method=args.method,
        quiet=args.quiet,
    )

    output = {
        "config": asdict(
            RunConfig(
                model_id=args.model_id,
                method=args.method,
                device=args.device,
                dtype=args.dtype,
            )
        ),
        "graph": {
            "n_forward": result.graph.n_forward,
            "n_backward": result.graph.n_backward,
            "n_edges": len(result.graph.edges),
        },
        "scores": {
            "shape": list(result.scores.shape),
            "min": float(result.scores.min().item()),
            "max": float(result.scores.max().item()),
            "mean": float(result.scores.mean().item()),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))
    print(f"Saved report to {out_path}")


if __name__ == "__main__":
    main()
