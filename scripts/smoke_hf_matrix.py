#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multimodal_lm_eap_ig import (  # noqa: E402
    HFLLMBackend,
    PreparedBatch,
    attribute,
    attribute_from_dataloader,
)
from multimodal_lm_eap_ig.graph import Graph  # noqa: E402

DEFAULT_TEXT_MODELS = [
    "gpt2",
    "distilgpt2",
    "facebook/opt-125m",
    "Qwen/Qwen2-0.5B",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]
DEFAULT_MULTIMODAL_MODELS = [
    "Qwen/Qwen2-VL-2B",
    "llava-hf/llava-1.5-7b-hf",
    "HuggingFaceTB/SmolVLM-Instruct",
]


@dataclass
class SmokeRow:
    model_id: str
    modality: str
    status: str
    seconds: float
    error_type: str
    error_message: str
    graph_stats: Optional[Dict[str, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HF backend smoke matrix for text and multimodal models.")
    parser.add_argument("--text-models", default=",".join(DEFAULT_TEXT_MODELS))
    parser.add_argument("--multimodal-models", default=",".join(DEFAULT_MULTIMODAL_MODELS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--output", default="reports/smoke_hf_matrix.json")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def to_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def parse_model_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def classify_error(exc: Exception) -> str:
    text = str(exc).lower()
    if "out of memory" in text or "cuda oom" in text:
        return "oom"
    if "401" in text or "403" in text or "unauthorized" in text or "forbidden" in text:
        return "auth"
    if "unsupported hf architecture" in text:
        return "unsupported_arch"
    if "placeholder" in text or "image features and image tokens do not match" in text:
        return "placeholder_mismatch"
    if "from_pretrained" in text or "configuration class" in text:
        return "model_load"
    return "runtime"


def smoke_metric(logits, clean_logits, batch):
    del clean_logits, batch
    return logits.sum()


def _pad_to_target(tensor: torch.Tensor, target_len: int, value: int) -> torch.Tensor:
    pad = target_len - int(tensor.shape[1])
    if pad <= 0:
        return tensor
    return torch.nn.functional.pad(tensor, (0, pad), value=value)


def build_text_prepared_batch(tokenizer) -> PreparedBatch:
    clean = tokenizer(["The Eiffel Tower is in"], return_tensors="pt", padding=True)
    corrupt = tokenizer(["The Brandenburg Gate is in"], return_tensors="pt", padding=True)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    target_len = max(int(clean["input_ids"].shape[1]), int(corrupt["input_ids"].shape[1]))
    clean_ids = _pad_to_target(clean["input_ids"], target_len, int(pad_token_id))
    corrupt_ids = _pad_to_target(corrupt["input_ids"], target_len, int(pad_token_id))
    clean_mask = _pad_to_target(clean["attention_mask"], target_len, 0)
    corrupt_mask = _pad_to_target(corrupt["attention_mask"], target_len, 0)
    shared_mask = torch.minimum(clean_mask, corrupt_mask)

    return PreparedBatch(
        clean_inputs={"input_ids": clean_ids, "attention_mask": shared_mask},
        corrupt_inputs={"input_ids": corrupt_ids, "attention_mask": shared_mask.clone()},
        labels=None,
        input_lengths=shared_mask.sum(dim=-1),
    )


def _build_multimodal_prompt(processor, question: str) -> str:
    if hasattr(processor, "apply_chat_template"):
        name = processor.__class__.__name__.lower()
        if "qwen2vl" in name or "qwen2_vl" in name:
            message = [{"type": "image"}, {"type": "text", "text": question}]
        else:
            message = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": question}],
                }
            ]
        prompt = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        if isinstance(prompt, str) and prompt.strip():
            return prompt
    return f"<image>\n{question}"


def _build_demo_image() -> torch.Tensor:
    image = torch.zeros((224, 224, 3), dtype=torch.uint8)
    image[48:176, 48:176, :] = 255
    return image.numpy()


def _load_multimodal_model(
    model_id: str,
    *,
    dtype: torch.dtype,
    token: Optional[str],
):
    kwargs: Dict[str, Any] = {"dtype": dtype, "trust_remote_code": True}
    if token:
        kwargs["token"] = token

    try:
        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    except Exception:
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq.from_pretrained(model_id, **kwargs)


def run_text_smoke_model(
    model_id: str,
    *,
    device: str,
    dtype: torch.dtype,
    token: Optional[str],
) -> SmokeRow:
    start = time.time()
    kwargs: Dict[str, Any] = {}
    if token:
        kwargs["token"] = token

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, **kwargs)
        model = model.to(device=device)
        model.eval()

        backend = HFLLMBackend(model, tokenizer=tokenizer)
        graph = Graph.from_model(backend.config)
        batch = build_text_prepared_batch(tokenizer)
        scores = attribute(
            model=model,
            backend=backend,
            graph=graph,
            batches=[batch],
            metric=smoke_metric,
            method="smoke",
            quiet=True,
        )

        return SmokeRow(
            model_id=model_id,
            modality="text",
            status="pass",
            seconds=time.time() - start,
            error_type="",
            error_message="",
            graph_stats={
                "n_forward": graph.n_forward,
                "n_backward": graph.n_backward,
                "n_edges": len(graph.edges),
                "score_shape_0": int(scores.shape[0]),
                "score_shape_1": int(scores.shape[1]),
            },
        )
    except Exception as exc:
        return SmokeRow(
            model_id=model_id,
            modality="text",
            status="fail",
            seconds=time.time() - start,
            error_type=classify_error(exc),
            error_message=str(exc),
            graph_stats=None,
        )


def run_multimodal_smoke_model(
    model_id: str,
    *,
    device: str,
    dtype: torch.dtype,
    token: Optional[str],
) -> SmokeRow:
    start = time.time()
    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if token:
        kwargs["token"] = token

    try:
        processor = AutoProcessor.from_pretrained(model_id, **kwargs)
        model = _load_multimodal_model(model_id, dtype=dtype, token=token)
        model = model.to(device=device)
        model.eval()

        backend = HFLLMBackend(model, tokenizer=getattr(processor, "tokenizer", None))
        graph = Graph.from_model(backend.config)

        image = _build_demo_image()
        clean_prompt = _build_multimodal_prompt(processor, "Is the square white?")
        corrupt_prompt = _build_multimodal_prompt(processor, "Is the square black?")
        dataloader = [
            {
                "clean": [{"text": clean_prompt, "images": image}],
                "corrupt": [{"text": corrupt_prompt, "images": image}],
                "labels": None,
            }
        ]

        scores = attribute_from_dataloader(
            model=model,
            backend=backend,
            dataloader=dataloader,
            metric=smoke_metric,
            processor=processor,
            method="smoke",
            quiet=True,
        ).scores

        return SmokeRow(
            model_id=model_id,
            modality="multimodal",
            status="pass",
            seconds=time.time() - start,
            error_type="",
            error_message="",
            graph_stats={
                "n_forward": graph.n_forward,
                "n_backward": graph.n_backward,
                "n_edges": len(graph.edges),
                "score_shape_0": int(scores.shape[0]),
                "score_shape_1": int(scores.shape[1]),
            },
        )
    except Exception as exc:
        return SmokeRow(
            model_id=model_id,
            modality="multimodal",
            status="fail",
            seconds=time.time() - start,
            error_type=classify_error(exc),
            error_message=str(exc),
            graph_stats=None,
        )


def run_matrix(
    *,
    text_models: Iterable[str],
    multimodal_models: Iterable[str],
    device: str,
    dtype: torch.dtype,
    hf_token: Optional[str],
    quiet: bool,
) -> Dict[str, Any]:
    rows: List[SmokeRow] = []

    for model_id in text_models:
        if not quiet:
            print(f"[text] {model_id}")
        rows.append(run_text_smoke_model(model_id, device=device, dtype=dtype, token=hf_token))

    for model_id in multimodal_models:
        if not quiet:
            print(f"[multimodal] {model_id}")
        rows.append(run_multimodal_smoke_model(model_id, device=device, dtype=dtype, token=hf_token))

    return {
        "all_passed": all(row.status == "pass" for row in rows),
        "results": [asdict(row) for row in rows],
    }


def main() -> None:
    args = parse_args()
    text_models = parse_model_list(args.text_models)
    multimodal_models = parse_model_list(args.multimodal_models)
    report = run_matrix(
        text_models=text_models,
        multimodal_models=multimodal_models,
        device=args.device,
        dtype=to_dtype(args.dtype),
        hf_token=args.hf_token,
        quiet=args.quiet,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Saved smoke matrix report to {output_path}")

    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
