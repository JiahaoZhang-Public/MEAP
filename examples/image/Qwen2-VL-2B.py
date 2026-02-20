#!/usr/bin/env python3
"""Image-text attribution example for Qwen2-VL-2B.

Workflow:
1. Raw image + question pairs -> processor -> model inputs
2. Attribution (default: EAP)
3. Graph visualization export (JSON + PNG)

Run:
  python "examples/image/Qwen2-VL-2B.py" --device cpu --dtype float32 --method EAP
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.common import (  # noqa: E402
    DEFAULT_IMAGE_URL,
    classify_error,
    dataclass_dict,
    export_graph_artifacts,
    load_image_from_path_or_url,
    metric_logit_diff,
    resolve_target_pair,
    to_dtype,
    write_model_input_summary,
    write_run_summary,
)

from multimodal_lm_eap_ig import (  # noqa: E402
    HFLLMBackend,
    attribute_from_dataloader,
)
from multimodal_lm_eap_ig.preparer import prepare_pair_batch_with_processor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen2-VL attribution example with graph visualization.")
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-2B")
    parser.add_argument(
        "--method",
        default="EAP",
        choices=["EAP", "EAP-IG-inputs", "clean-corrupted", "EAP-IG-activations", "exact", "smoke"],
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--topn", type=int, default=200)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--image-path", default="")
    parser.add_argument("--image-url", default=DEFAULT_IMAGE_URL)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def build_output_dir(run_name: str) -> Path:
    base = Path(__file__).resolve().parent / "outputs"
    if run_name:
        out = base / run_name
    else:
        out = base / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _build_prompt(processor: Any, question: str) -> str:
    if hasattr(processor, "apply_chat_template"):
        message = [{"type": "image"}, {"type": "text", "text": question}]
        prompt = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        if isinstance(prompt, str) and prompt.strip():
            return prompt
    return f"<image>\n{question}"


def main() -> None:
    args = parse_args()
    output_dir = build_output_dir(args.run_name)
    start = time.time()

    try:
        processor = AutoProcessor.from_pretrained(args.model_id, token=args.hf_token, trust_remote_code=True)
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                args.model_id,
                dtype=to_dtype(args.dtype),
                token=args.hf_token,
                trust_remote_code=True,
            )
        except Exception:
            from transformers import AutoModelForVision2Seq

            model = AutoModelForVision2Seq.from_pretrained(
                args.model_id,
                dtype=to_dtype(args.dtype),
                token=args.hf_token,
                trust_remote_code=True,
            )
        model = model.to(args.device)
        model.eval()

        backend = HFLLMBackend(model, tokenizer=getattr(processor, "tokenizer", None))

        # Step 1: raw data (image + prompt pairs)
        image = load_image_from_path_or_url(image_path=args.image_path, image_url=args.image_url)
        clean_prompt = _build_prompt(processor, "What animal is on the candy?")
        corrupt_prompt = _build_prompt(processor, "What object is shown in this scene?")

        labels = resolve_target_pair(getattr(processor, "tokenizer", None), model)

        # Step 1 -> model input tensors via processor
        prepared_batch = prepare_pair_batch_with_processor(
            processor=processor,
            clean_samples=[{"text": clean_prompt, "images": image}],
            corrupt_samples=[{"text": corrupt_prompt, "images": image}],
            labels=labels,
            processor_kwargs={"padding": True},
            device=backend.config.device,
        )
        write_model_input_summary(
            prepared_batch,
            output_dir / "model_input_summary.json",
            extra={
                "raw_data": {
                    "image_path": args.image_path,
                    "image_url": args.image_url,
                    "clean_prompt": clean_prompt,
                    "corrupt_prompt": corrupt_prompt,
                }
            },
        )

        # Step 2: attribution
        run = attribute_from_dataloader(
            model=model,
            backend=backend,
            dataloader=[prepared_batch],
            metric=metric_logit_diff,
            method=args.method,
            quiet=args.quiet,
        )

        # Step 3: visualization
        artifacts, graph_stats = export_graph_artifacts(run.graph, output_dir, topn=args.topn)

        result: Dict[str, Any] = {
            "modality": "image",
            "model_id": args.model_id,
            "method": args.method,
            "status": "pass",
            "seconds": time.time() - start,
            "graph_stats": dataclass_dict(graph_stats),
            "artifacts": {
                **dataclass_dict(artifacts),
                "model_input_summary": str(output_dir / "model_input_summary.json"),
            },
        }
    except Exception as exc:  # noqa: BLE001
        result = {
            "modality": "image",
            "model_id": args.model_id,
            "method": args.method,
            "status": "fail",
            "seconds": time.time() - start,
            "error_type": classify_error(exc),
            "error_message": str(exc),
        }

    write_run_summary(output_dir / "run_summary.json", result)
    print(json.dumps(result, indent=2))
    print(f"Saved outputs to {output_dir}")

    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
