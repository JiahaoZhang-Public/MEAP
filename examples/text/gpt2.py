#!/usr/bin/env python3
"""Text attribution example using AttributionModel + PreparedBatch.

Workflow:
1. Raw clean/corrupt text pair -> tokenizer -> PreparedBatch
2. AttributionModel resolves language trunk and runs attribution
3. Export graph JSON/PNG artifacts

Run:
  python examples/text/gpt2.py --device cpu --dtype float32 --method EAP
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
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.common import (  # noqa: E402
    classify_error,
    dataclass_dict,
    export_graph_artifacts,
    metric_logit_diff,
    resolve_target_pair,
    to_dtype,
    write_model_input_summary,
    write_run_summary,
)

from meap import AttributionModel, PreparedBatch  # noqa: E402
from meap.batch import validate_prepared_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPT-2 text attribution example.")
    parser.add_argument("--model-id", default="gpt2")
    parser.add_argument(
        "--method",
        default="EAP",
        choices=["EAP", "EAP-IG-inputs", "clean-corrupted", "EAP-IG-activations", "exact", "smoke"],
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--topn", type=int, default=200)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _pad_2d(tensor: torch.Tensor, target_len: int, pad_value: int) -> torch.Tensor:
    pad = target_len - int(tensor.shape[1])
    if pad <= 0:
        return tensor
    return torch.nn.functional.pad(tensor, (0, pad), value=pad_value)


def prepare_text_batch(
    tokenizer: Any,
    *,
    clean_text: str,
    corrupt_text: str,
    labels: torch.Tensor,
    device: torch.device,
    max_length: int,
) -> PreparedBatch:
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

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    target_len = max(int(clean["input_ids"].shape[1]), int(corrupt["input_ids"].shape[1]))
    clean_ids = _pad_2d(clean["input_ids"], target_len, int(pad_token_id)).to(device)
    corrupt_ids = _pad_2d(corrupt["input_ids"], target_len, int(pad_token_id)).to(device)
    clean_mask = _pad_2d(clean["attention_mask"], target_len, 0).to(device)
    corrupt_mask = _pad_2d(corrupt["attention_mask"], target_len, 0).to(device)

    shared_mask = torch.minimum(clean_mask, corrupt_mask)
    batch = PreparedBatch(
        clean_inputs={"input_ids": clean_ids, "attention_mask": shared_mask},
        corrupt_inputs={"input_ids": corrupt_ids, "attention_mask": shared_mask.clone()},
        labels=labels.to(device=device, dtype=torch.long),
        input_lengths=shared_mask.sum(dim=-1),
    )
    validate_prepared_batch(batch)
    return batch


def build_output_dir(run_name: str) -> Path:
    base = Path(__file__).resolve().parent / "outputs"
    if run_name:
        out = base / run_name
    else:
        out = base / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _model_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "dtype": to_dtype(args.dtype),
        "trust_remote_code": args.trust_remote_code,
    }
    if args.hf_token:
        kwargs["token"] = args.hf_token
    return kwargs


def main() -> None:
    args = parse_args()
    output_dir = build_output_dir(args.run_name)
    start = time.time()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            token=args.hf_token,
            trust_remote_code=args.trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        attribution_model = AttributionModel.from_pretrained(
            args.model_id,
            device=args.device,
            dtype=args.dtype,
            model_kwargs=_model_kwargs(args),
            cache=False,
        )

        clean_text = "The capital of France is"
        corrupt_text = "The capital of Germany is"
        labels = resolve_target_pair(tokenizer, attribution_model.model)

        prepared_batch = prepare_text_batch(
            tokenizer,
            clean_text=clean_text,
            corrupt_text=corrupt_text,
            labels=labels,
            device=attribution_model.backend.config.device,
            max_length=args.max_length,
        )
        write_model_input_summary(
            prepared_batch,
            output_dir / "model_input_summary.json",
            extra={
                "raw_data": {
                    "clean_text": clean_text,
                    "corrupt_text": corrupt_text,
                }
            },
        )

        run = attribution_model.attribute(
            batches=[prepared_batch],
            metric=metric_logit_diff,
            method=args.method,
            quiet=args.quiet,
        )
        artifacts, graph_stats = export_graph_artifacts(run.graph, output_dir, topn=args.topn)

        route = run.route_info
        result: Dict[str, Any] = {
            "modality": "text",
            "model_id": args.model_id,
            "method": args.method,
            "status": "pass",
            "seconds": time.time() - start,
            "route_info": {
                "adapter_name": route.adapter_name,
                "arch_kind": route.arch_kind,
                "language_trunk_path": route.language_trunk_path,
            },
            "graph_stats": dataclass_dict(graph_stats),
            "artifacts": {
                **dataclass_dict(artifacts),
                "model_input_summary": str(output_dir / "model_input_summary.json"),
            },
        }
    except Exception as exc:  # noqa: BLE001
        result = {
            "modality": "text",
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
