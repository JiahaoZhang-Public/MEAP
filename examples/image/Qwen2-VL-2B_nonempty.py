#!/usr/bin/env python3
"""Image attribution example targeting non-empty top200 circuits for Qwen2-VL-2B.

Workflow:
1. White/black image pair + shared question -> dataloader input
   (`PreparedBatch` or `RawPairBatch + pair_batch_preparer`)
2. Attribution (default: EAP)
3. Root-aware topn export to avoid empty pruned circuit at small topn

Run:
  python examples/image/Qwen2-VL-2B_nonempty.py --dtype float16 --method EAP --image-size 128 --input-mode prepared
  python examples/image/Qwen2-VL-2B_nonempty.py --dtype float16 --method EAP --image-size 128 --input-mode raw
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict

from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.common import (  # noqa: E402
    classify_error,
    dataclass_dict,
    export_graph_artifacts_nonempty_topn,
    to_dtype,
    write_model_input_summary,
    write_run_summary,
)

from meap import (  # noqa: E402
    HFLLMBackend,
    HFProcessorAdapter,
    PreparedBatch,
    RawPairBatch,
    attribute_from_dataloader,
)
from meap.batch import validate_prepared_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen2-VL non-empty top200 attribution example with root-aware pruning."
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-2B")
    parser.add_argument(
        "--method",
        default="EAP",
        choices=["EAP", "EAP-IG-inputs", "clean-corrupted", "EAP-IG-activations", "exact", "smoke"],
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--topn", type=int, default=200)
    parser.add_argument(
        "--input-mode",
        default="prepared",
        choices=["prepared", "raw"],
        help="Dataloader entry type: PreparedBatch or RawPairBatch + pair_batch_preparer.",
    )
    parser.add_argument("--hf-token", default=None)
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Square image size to control dynamic image token count.",
    )
    parser.add_argument(
        "--root-policy",
        default="input_layer0",
        choices=["input_layer0"],
        help="Root selection policy for non-empty topn pruning.",
    )
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
        message_variants = [
            [{"type": "image"}, {"type": "text", "text": question}],
            [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": question}],
                }
            ],
        ]
        for messages in message_variants:
            try:
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                continue
            if isinstance(prompt, str) and prompt.strip():
                if any(tok in prompt for tok in ("<|image_pad|>", "<image>", "<im_patch>", "<image_pad>")):
                    return prompt
    return f"<|vision_start|><|image_pad|><|vision_end|>\n{question}"


def _batchfeature_to_tensors(batch_feature) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in batch_feature.items() if torch.is_tensor(v)}


def _pad_to_length(tensor: torch.Tensor, target_length: int, pad_value: int) -> torch.Tensor:
    pad = target_length - int(tensor.shape[1])
    if pad <= 0:
        return tensor
    return torch.nn.functional.pad(tensor, (0, pad), value=pad_value)


def _resolve_pad_token_id(processor: Any) -> int:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return 0
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0
    return int(pad_token_id)


def _prepare_image_pair_batch(
    *,
    processor: Any,
    clean_prompt: str,
    corrupt_prompt: str,
    clean_image: Any,
    corrupt_image: Any,
    labels: Any,
    device: torch.device,
) -> PreparedBatch:
    clean_feature = processor(
        text=[clean_prompt],
        images=[clean_image],
        return_tensors="pt",
        padding=True,
    )
    corrupt_feature = processor(
        text=[corrupt_prompt],
        images=[corrupt_image],
        return_tensors="pt",
        padding=True,
    )

    clean_inputs = _batchfeature_to_tensors(clean_feature)
    corrupt_inputs = _batchfeature_to_tensors(corrupt_feature)

    pad_token_id = _resolve_pad_token_id(processor)
    target_len = max(int(clean_inputs["input_ids"].shape[1]), int(corrupt_inputs["input_ids"].shape[1]))
    clean_inputs["input_ids"] = _pad_to_length(clean_inputs["input_ids"], target_len, pad_token_id)
    corrupt_inputs["input_ids"] = _pad_to_length(corrupt_inputs["input_ids"], target_len, pad_token_id)

    if "attention_mask" in clean_inputs:
        clean_inputs["attention_mask"] = _pad_to_length(clean_inputs["attention_mask"], target_len, 0)
    else:
        clean_inputs["attention_mask"] = (clean_inputs["input_ids"] != pad_token_id).long()
    if "attention_mask" in corrupt_inputs:
        corrupt_inputs["attention_mask"] = _pad_to_length(corrupt_inputs["attention_mask"], target_len, 0)
    else:
        corrupt_inputs["attention_mask"] = (corrupt_inputs["input_ids"] != pad_token_id).long()

    clean_inputs = {k: v.to(device) for k, v in clean_inputs.items()}
    corrupt_inputs = {k: v.to(device) for k, v in corrupt_inputs.items()}

    shared_mask = torch.minimum(clean_inputs["attention_mask"], corrupt_inputs["attention_mask"])
    clean_inputs["attention_mask"] = shared_mask
    corrupt_inputs["attention_mask"] = shared_mask.clone()

    prepared_batch = PreparedBatch(
        clean_inputs=clean_inputs,
        corrupt_inputs=corrupt_inputs,
        labels=labels,
        input_lengths=shared_mask.sum(dim=-1),
    )
    validate_prepared_batch(prepared_batch)
    return prepared_batch


def _build_clean_corrupt_images(image_size: int) -> tuple[Image.Image, Image.Image]:
    if image_size <= 0:
        raise ValueError(f"--image-size must be positive, got {image_size}")
    clean = Image.new("RGB", (image_size, image_size), color=(255, 255, 255))
    corrupt = Image.new("RGB", (image_size, image_size), color=(0, 0, 0))
    return clean, corrupt


def _single_token_id(tokenizer: Any, token: str) -> int | None:
    try:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
    except Exception:
        return None
    if len(token_ids) == 1:
        return int(token_ids[0])
    return None


def _resolve_white_black_pair(tokenizer: Any) -> torch.Tensor:
    candidates = [(" white", " black"), ("white", "black"), (" White", " Black")]
    for white_text, black_text in candidates:
        white_id = _single_token_id(tokenizer, white_text)
        black_id = _single_token_id(tokenizer, black_text)
        if white_id is not None and black_id is not None:
            return torch.tensor([[white_id, black_id]], dtype=torch.long)
    raise ValueError("Could not resolve single-token ids for 'white' and 'black'.")


def _metric_white_black_logit_diff(logits: torch.Tensor, clean_logits: torch.Tensor | None, batch) -> torch.Tensor:
    del clean_logits
    labels = batch.labels
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels, dtype=torch.long, device=logits.device)
    else:
        labels = labels.to(device=logits.device, dtype=torch.long)
    if labels.ndim == 1:
        labels = labels.unsqueeze(0)

    input_lengths = batch.input_lengths.to(device=logits.device, dtype=torch.long)
    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    final_logits = logits[batch_idx, input_lengths - 1]
    selected = torch.gather(final_logits, dim=-1, index=labels)
    return (selected[:, 0] - selected[:, 1]).mean()


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
        clean_image, corrupt_image = _build_clean_corrupt_images(args.image_size)
        question = "what is the color of the image?"
        clean_prompt = _build_prompt(processor, question)
        corrupt_prompt = _build_prompt(processor, question)

        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("Processor tokenizer is required to resolve white/black token ids.")
        labels = _resolve_white_black_pair(tokenizer)

        if args.input_mode == "prepared":
            prepared_batch = _prepare_image_pair_batch(
                processor=processor,
                clean_prompt=clean_prompt,
                corrupt_prompt=corrupt_prompt,
                clean_image=clean_image,
                corrupt_image=corrupt_image,
                labels=labels,
                device=backend.config.device,
            )
            dataloader = [prepared_batch]
            pair_batch_preparer = None
            summary_batch = prepared_batch
        else:
            raw_batch = RawPairBatch(
                clean={"text": [clean_prompt], "images": [clean_image]},
                corrupt={"text": [corrupt_prompt], "images": [corrupt_image]},
                labels=labels,
            )
            pair_batch_preparer = HFProcessorAdapter(
                processor=processor,
                device=backend.config.device,
            )
            dataloader = [raw_batch]
            summary_batch = pair_batch_preparer.prepare_batch(
                clean_samples=raw_batch.clean,
                corrupt_samples=raw_batch.corrupt,
                labels=raw_batch.labels,
                meta=raw_batch.meta,
            )

        write_model_input_summary(
            summary_batch,
            output_dir / "model_input_summary.json",
            extra={
                "raw_data": {
                    "clean_image": "solid_white",
                    "corrupt_image": "solid_black",
                    "image_size": args.image_size,
                    "clean_prompt": clean_prompt,
                    "corrupt_prompt": corrupt_prompt,
                    "labels": labels.tolist(),
                },
                "input_mode": args.input_mode,
            },
        )

        run = attribute_from_dataloader(
            model=model,
            backend=backend,
            dataloader=dataloader,
            pair_batch_preparer=pair_batch_preparer,
            metric=_metric_white_black_logit_diff,
            method=args.method,
            quiet=args.quiet,
        )
        artifacts, graph_stats, selection = export_graph_artifacts_nonempty_topn(
            run.graph,
            output_dir,
            topn=args.topn,
            root_policy=args.root_policy,
        )

        result: Dict[str, Any] = {
            "modality": "image",
            "model_id": args.model_id,
            "method": args.method,
            "input_mode": args.input_mode,
            "status": "pass",
            "seconds": time.time() - start,
            "selection": selection,
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
            "input_mode": args.input_mode,
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
