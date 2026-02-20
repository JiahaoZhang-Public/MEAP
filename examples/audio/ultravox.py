#!/usr/bin/env python3
"""Audio attribution example for Ultravox.

Workflow:
1. Raw audio + dialogue turns -> pipeline preprocess -> model inputs
2. Attribution (default: EAP)
3. Graph visualization export (JSON + PNG)

Run:
  python examples/audio/ultravox.py --device cpu --dtype float32 --method EAP
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
    export_graph_artifacts,
    load_audio_from_path_or_url,
    metric_logit_diff,
    resolve_target_pair,
    to_dtype,
    write_model_input_summary,
    write_run_summary,
)

from multimodal_lm_eap_ig import (  # noqa: E402
    HFLLMBackend,
    PreparedBatch,
    attribute_from_dataloader,
)
from multimodal_lm_eap_ig.batch import PairBatchPreparer, validate_prepared_batch  # noqa: E402


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

        prepared = PreparedBatch(
            clean_inputs=clean_inputs,
            corrupt_inputs=corrupt_inputs,
            labels=labels,
            input_lengths=clean_inputs["attention_mask"].sum(dim=-1),
            meta=meta,
        )
        validate_prepared_batch(prepared)
        return prepared


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultravox audio attribution example with graph visualization.")
    parser.add_argument("--model-id", default="fixie-ai/ultravox-v0_5-llama-3_2-1b")
    parser.add_argument(
        "--method",
        default="EAP",
        choices=["EAP", "EAP-IG-inputs", "clean-corrupted", "EAP-IG-activations", "exact", "smoke"],
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--topn", type=int, default=200)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--audio-path", default="")
    parser.add_argument("--audio-url", default=DEFAULT_AUDIO_URL)
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
        infer_pipe = pipeline(model=args.model_id, trust_remote_code=True, token=args.hf_token)
        pair_preparer = UltravoxPairPreparer(infer_pipe=infer_pipe, device=backend.config.device)

        # Step 1: raw audio and prompt turns
        audio, sr = load_audio_from_path_or_url(
            audio_path=args.audio_path,
            audio_url=args.audio_url,
            sampling_rate=16000,
        )
        turns_clean = [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Summarize the spoken content in one sentence."},
        ]
        turns_corrupt = [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Transcribe the spoken content."},
        ]
        labels = resolve_target_pair(getattr(infer_pipe, "tokenizer", None), model)

        # Step 1 -> model inputs
        prepared_batch = pair_preparer.prepare_batch(
            clean_samples={"audio": audio, "sampling_rate": sr, "turns": turns_clean},
            corrupt_samples={"audio": audio, "sampling_rate": sr, "turns": turns_corrupt},
            labels=labels,
        )
        write_model_input_summary(
            prepared_batch,
            output_dir / "model_input_summary.json",
            extra={
                "raw_data": {
                    "audio_path": args.audio_path,
                    "audio_url": args.audio_url,
                    "sampling_rate": sr,
                    "clean_turns": turns_clean,
                    "corrupt_turns": turns_corrupt,
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
            "modality": "audio",
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
            "modality": "audio",
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
