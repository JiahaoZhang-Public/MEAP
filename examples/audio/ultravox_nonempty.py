#!/usr/bin/env python3
"""Audio attribution example targeting non-empty top200 circuits for Ultravox.

Workflow:
1. Raw audio + clean/corrupt turns -> dataloader input
   (`PreparedBatch` or `RawPairBatch + pair_batch_preparer`)
2. Attribution (default: EAP)
3. Root-aware topn export to avoid empty pruned circuit at small topn

Run:
  python examples/audio/ultravox_nonempty.py --method EAP --dtype float32 --input-mode prepared
  python examples/audio/ultravox_nonempty.py --method EAP --dtype float32 --input-mode raw
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Mapping, Optional

import torch
from transformers import AutoModel, pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.common import (  # noqa: E402
    DEFAULT_AUDIO_URL,
    classify_error,
    dataclass_dict,
    export_graph_artifacts_nonempty_topn,
    load_audio_from_path_or_url,
    metric_logit_diff,
    resolve_target_pair,
    to_dtype,
    write_model_input_summary,
    write_run_summary,
)

from meap import (  # noqa: E402
    HFLLMBackend,
    PreparedBatch,
    RawPairBatch,
    attribute_from_dataloader,
)
from meap.batch import PairBatchPreparer, validate_prepared_batch  # noqa: E402

ULTRAVOX_AUDIO_PLACEHOLDER = "<|audio|>"


def _pad_to_length(tensor: torch.Tensor, target_length: int, pad_value: int) -> torch.Tensor:
    pad = target_length - int(tensor.shape[1])
    if pad <= 0:
        return tensor
    return torch.nn.functional.pad(tensor, (0, pad), value=pad_value)


def _resolve_pad_token_id(infer_pipe: Any) -> int:
    tokenizer = getattr(infer_pipe, "tokenizer", None)
    if tokenizer is None:
        return 0
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0
    return int(pad_token_id)


def normalize_ultravox_user_prompt(prompt: str, *, audio_placeholder: str) -> str:
    placeholder = str(audio_placeholder).strip() or ULTRAVOX_AUDIO_PLACEHOLDER
    text = str(prompt).strip()
    for token in ("<|audio_bos|>", "<|audio_eos|>", "<|AUDIO|>", ULTRAVOX_AUDIO_PLACEHOLDER):
        if token == placeholder:
            continue
        text = text.replace(token, " ")

    segments = [seg.strip() for seg in text.split(placeholder)]
    merged = " ".join(seg for seg in segments if seg)
    if merged:
        return f"{placeholder}\n{merged}"
    return placeholder


@dataclass
class UltravoxPairPreparer(PairBatchPreparer):
    infer_pipe: Any
    device: torch.device

    def _encode_one(self, sample: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        features = self.infer_pipe.preprocess(
            {
                "audio": sample["audio"],
                "turns": sample["turns"],
                "sampling_rate": sample["sampling_rate"],
            }
        )
        if not isinstance(features, Mapping):
            raise TypeError("Ultravox pipeline preprocess must return a mapping of model tensors.")

        out: Dict[str, torch.Tensor] = {}
        for key, value in features.items():
            if torch.is_tensor(value):
                out[key] = value.to(self.device)

        if "attention_mask" not in out and "input_ids" in out:
            out["attention_mask"] = torch.ones_like(
                out["input_ids"],
                dtype=torch.long,
                device=self.device,
            )
        return out

    def prepare_batch(
        self,
        clean_samples,
        corrupt_samples,
        labels,
        *,
        meta: Optional[Dict[str, Any]] = None,
    ) -> PreparedBatch:
        clean_inputs = self._encode_one(clean_samples)
        corrupt_inputs = self._encode_one(corrupt_samples)
        if "attention_mask" not in clean_inputs or "attention_mask" not in corrupt_inputs:
            raise ValueError("Ultravox preparer requires attention_mask in both clean and corrupt inputs.")

        pad_token_id = _resolve_pad_token_id(self.infer_pipe)
        target_len = max(int(clean_inputs["input_ids"].shape[1]), int(corrupt_inputs["input_ids"].shape[1]))
        clean_inputs["input_ids"] = _pad_to_length(clean_inputs["input_ids"], target_len, pad_token_id)
        corrupt_inputs["input_ids"] = _pad_to_length(corrupt_inputs["input_ids"], target_len, pad_token_id)
        clean_inputs["attention_mask"] = _pad_to_length(clean_inputs["attention_mask"], target_len, 0)
        corrupt_inputs["attention_mask"] = _pad_to_length(corrupt_inputs["attention_mask"], target_len, 0)

        shared_mask = torch.minimum(clean_inputs["attention_mask"], corrupt_inputs["attention_mask"])
        clean_inputs["attention_mask"] = shared_mask
        corrupt_inputs["attention_mask"] = shared_mask.clone()

        prepared = PreparedBatch(
            clean_inputs=clean_inputs,
            corrupt_inputs=corrupt_inputs,
            labels=labels,
            input_lengths=shared_mask.sum(dim=-1),
            meta=meta,
        )
        validate_prepared_batch(prepared)
        return prepared


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ultravox non-empty top200 attribution example with root-aware pruning."
    )
    parser.add_argument("--model-id", default="fixie-ai/ultravox-v0_5-llama-3_2-1b")
    parser.add_argument(
        "--method",
        default="EAP",
        choices=["EAP", "EAP-IG-inputs", "clean-corrupted", "EAP-IG-activations", "exact", "smoke"],
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--topn", type=int, default=200)
    parser.add_argument(
        "--input-mode",
        default="prepared",
        choices=["prepared", "raw"],
        help="Dataloader entry type: PreparedBatch or RawPairBatch + pair_batch_preparer.",
    )
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--audio-path", default="")
    parser.add_argument("--audio-url", default=DEFAULT_AUDIO_URL)
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


def _pipeline_device_arg(device: torch.device) -> int:
    if device.type == "cuda":
        return 0 if device.index is None else int(device.index)
    return -1


def main() -> None:
    args = parse_args()
    output_dir = build_output_dir(args.run_name)
    start = time.time()

    try:
        model = AutoModel.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            dtype=to_dtype(args.dtype),
            token=args.hf_token,
        ).to(args.device)
        model.eval()

        backend = HFLLMBackend(model)
        infer_pipe = pipeline(
            model=args.model_id,
            trust_remote_code=True,
            token=args.hf_token,
            device=_pipeline_device_arg(backend.config.device),
        )
        pair_preparer = UltravoxPairPreparer(infer_pipe=infer_pipe, device=backend.config.device)
        audio_placeholder = getattr(
            getattr(infer_pipe, "processor", None),
            "audio_placeholder",
            ULTRAVOX_AUDIO_PLACEHOLDER,
        )

        audio, sr = load_audio_from_path_or_url(
            audio_path=args.audio_path,
            audio_url=args.audio_url,
            sampling_rate=16000,
        )
        turns_clean = [
            {"role": "system", "content": "You are concise."},
            {
                "role": "user",
                "content": normalize_ultravox_user_prompt(
                    "Summarize the spoken content in one sentence.",
                    audio_placeholder=audio_placeholder,
                ),
            },
        ]
        turns_corrupt = [
            {"role": "system", "content": "You are concise."},
            {
                "role": "user",
                "content": normalize_ultravox_user_prompt(
                    "Transcribe the spoken content.",
                    audio_placeholder=audio_placeholder,
                ),
            },
        ]
        labels = resolve_target_pair(getattr(infer_pipe, "tokenizer", None), model)
        clean_sample = {"audio": audio, "sampling_rate": sr, "turns": turns_clean}
        corrupt_sample = {"audio": audio, "sampling_rate": sr, "turns": turns_corrupt}

        if args.input_mode == "prepared":
            prepared_batch = pair_preparer.prepare_batch(
                clean_samples=clean_sample,
                corrupt_samples=corrupt_sample,
                labels=labels,
            )
            dataloader = [prepared_batch]
            pair_batch_preparer = None
            summary_batch = prepared_batch
        else:
            raw_batch = RawPairBatch(clean=clean_sample, corrupt=corrupt_sample, labels=labels)
            dataloader = [raw_batch]
            pair_batch_preparer = pair_preparer
            summary_batch = pair_preparer.prepare_batch(
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
                    "audio_path": args.audio_path,
                    "audio_url": args.audio_url,
                    "sampling_rate": sr,
                    "clean_turns": turns_clean,
                    "corrupt_turns": turns_corrupt,
                },
                "input_mode": args.input_mode,
            },
        )

        run = attribute_from_dataloader(
            model=model,
            backend=backend,
            dataloader=dataloader,
            pair_batch_preparer=pair_batch_preparer,
            metric=metric_logit_diff,
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
            "modality": "audio",
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
            "modality": "audio",
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
