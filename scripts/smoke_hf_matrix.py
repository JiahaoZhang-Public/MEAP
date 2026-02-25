#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    pipeline,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from meap import (  # noqa: E402
    AttributionModel,
    HFLLMBackend,
    HFProcessorAdapter,
    PreparedBatch,
    attribute,
    inspect_model_architecture,
    list_official_models,
)
from meap.batch import PairBatchPreparer, validate_prepared_batch  # noqa: E402
from meap.graph import Graph  # noqa: E402

_OFFICIAL_MODELS = list_official_models()
DEFAULT_TEXT_MODELS = [
    row["model_id"]
    for row in _OFFICIAL_MODELS
    if row["modality"] == "text" and row.get("tier", "core") == "core"
]
DEFAULT_MULTIMODAL_MODELS = [
    row["model_id"]
    for row in _OFFICIAL_MODELS
    if row["modality"] == "multimodal" and row.get("tier", "core") == "core"
]
REPORT_TYPE = "smoke_hf_matrix"
SCHEMA_VERSION = "1.1.0"


@dataclass
class SmokeRow:
    model_id: str
    modality: str
    adapter_name: str
    language_trunk_path: str
    arch_kind: str
    status: str
    seconds: float
    error_type: str
    error_message: str
    resolution_error_hint: str
    graph_stats: Optional[Dict[str, int]]


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
        out = {}
        for key, value in features.items():
            if torch.is_tensor(value):
                out[key] = value.to(self.device)
        if "attention_mask" not in out and "input_ids" in out:
            out["attention_mask"] = torch.ones_like(out["input_ids"], dtype=torch.long, device=self.device)
        return out

    def prepare_batch(self, clean_samples, corrupt_samples, labels, *, meta: Optional[Dict[str, Any]] = None):
        clean_inputs = self._encode_one(clean_samples)
        corrupt_inputs = self._encode_one(corrupt_samples)
        if "attention_mask" not in clean_inputs or "attention_mask" not in corrupt_inputs:
            raise ValueError("Ultravox preparer requires attention_mask in both clean and corrupt inputs.")
        input_lengths = clean_inputs["attention_mask"].sum(dim=-1)
        prepared = PreparedBatch(
            clean_inputs=clean_inputs,
            corrupt_inputs=corrupt_inputs,
            labels=labels,
            input_lengths=input_lengths,
            meta=meta,
        )
        validate_prepared_batch(prepared)
        return prepared


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HF backend smoke matrix for text and multimodal models.")
    parser.add_argument("--text-models", default=",".join(DEFAULT_TEXT_MODELS))
    parser.add_argument("--multimodal-models", default=",".join(DEFAULT_MULTIMODAL_MODELS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--hf-token", default=None)
    parser.add_argument(
        "--audio-fallback-model",
        default="fixie-ai/ultravox-v0_5-llama-3_2-1b",
        help="Fallback model for audio smoke when primary audio model fails to load.",
    )
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
    if "no module named" in text:
        return "dependency_missing"
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


def _build_demo_audio() -> np.ndarray:
    return np.zeros((16000,), dtype=np.float32)


def _is_audio_model_id(model_id: str) -> bool:
    lower = model_id.lower()
    return "audio" in lower or "ultravox" in lower


def _load_multimodal_model(
    model_id: str,
    *,
    dtype: torch.dtype,
    token: Optional[str],
):
    kwargs: Dict[str, Any] = {"dtype": dtype, "trust_remote_code": True}
    if token:
        kwargs["token"] = token

    if "qwen2-audio" in model_id.lower():
        from transformers import Qwen2AudioForConditionalGeneration

        return Qwen2AudioForConditionalGeneration.from_pretrained(model_id, **kwargs)

    try:
        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    except Exception:
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq.from_pretrained(model_id, **kwargs)


def _load_audio_model(
    model_id: str,
    *,
    dtype: torch.dtype,
    token: Optional[str],
):
    kwargs: Dict[str, Any] = {"dtype": dtype, "trust_remote_code": True}
    if token:
        kwargs["token"] = token
    if "qwen2-audio" in model_id.lower():
        from transformers import Qwen2AudioForConditionalGeneration

        return Qwen2AudioForConditionalGeneration.from_pretrained(model_id, **kwargs)
    try:
        return AutoModel.from_pretrained(model_id, **kwargs)
    except Exception as auto_exc:
        try:
            return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        except Exception as causal_exc:
            raise RuntimeError(
                "Failed to load audio model with both AutoModel and AutoModelForCausalLM. "
                f"AutoModel error: {auto_exc}; AutoModelForCausalLM error: {causal_exc}"
            ) from causal_exc


def run_text_smoke_model(
    model_id: str,
    *,
    device: str,
    dtype: torch.dtype,
    token: Optional[str],
) -> SmokeRow:
    start = time.time()
    model = None
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
            adapter_name=backend.adapter_name,
            language_trunk_path=backend.language_trunk_path,
            arch_kind=backend.arch_kind,
            status="pass",
            seconds=time.time() - start,
            error_type="",
            error_message="",
            resolution_error_hint="",
            graph_stats={
                "n_forward": graph.n_forward,
                "n_backward": graph.n_backward,
                "n_edges": len(graph.edges),
                "score_shape_0": int(scores.shape[0]),
                "score_shape_1": int(scores.shape[1]),
            },
        )
    except Exception as exc:
        hint = ""
        if model is not None and "unsupported hf architecture" in str(exc).lower():
            try:
                diag = inspect_model_architecture(model)
                selected = diag.get("selected")
                if selected:
                    hint = f"selected={selected.get('adapter')}@{selected.get('path')}"
                else:
                    attempts = diag.get("adapter_attempts", [])
                    if attempts:
                        top = attempts[0]
                        hint = (
                            f"attempt={top.get('adapter')}@{top.get('path')}: "
                            f"{top.get('detail', '')}"
                        )
            except Exception:
                hint = ""
        return SmokeRow(
            model_id=model_id,
            modality="text",
            adapter_name="",
            language_trunk_path="",
            arch_kind="",
            status="fail",
            seconds=time.time() - start,
            error_type=classify_error(exc),
            error_message=str(exc),
            resolution_error_hint=hint,
            graph_stats=None,
        )


def run_multimodal_smoke_model(
    model_id: str,
    *,
    device: str,
    dtype: torch.dtype,
    token: Optional[str],
    audio_fallback_model: Optional[str],
) -> SmokeRow:
    if _is_audio_model_id(model_id):
        return run_audio_smoke_model(
            model_id=model_id,
            device=device,
            dtype=dtype,
            token=token,
            audio_fallback_model=audio_fallback_model,
        )

    start = time.time()
    model = None
    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if token:
        kwargs["token"] = token

    try:
        processor = AutoProcessor.from_pretrained(model_id, **kwargs)
        model = _load_multimodal_model(model_id, dtype=dtype, token=token)
        model = model.to(device=device)
        model.eval()

        attribution_model = AttributionModel.from_model(
            model,
            tokenizer=getattr(processor, "tokenizer", None),
            device=device,
            dtype=dtype,
        )
        route_info = attribution_model.route_info
        graph = attribution_model.build_graph()

        image = _build_demo_image()
        clean_prompt = _build_multimodal_prompt(processor, "Is the square white?")
        corrupt_prompt = _build_multimodal_prompt(processor, "Is the square black?")
        preparer = HFProcessorAdapter(
            processor=processor,
            device=attribution_model.backend.config.device,
        )
        prepared = preparer.prepare_batch(
            clean_samples=[{"text": clean_prompt, "images": image}],
            corrupt_samples=[{"text": corrupt_prompt, "images": image}],
            labels=None,
        )
        scores = attribution_model.attribute(
            batches=[prepared],
            metric=smoke_metric,
            method="smoke",
            quiet=True,
        ).scores

        return SmokeRow(
            model_id=model_id,
            modality="multimodal",
            adapter_name=route_info.adapter_name,
            language_trunk_path=route_info.language_trunk_path,
            arch_kind=route_info.arch_kind,
            status="pass",
            seconds=time.time() - start,
            error_type="",
            error_message="",
            resolution_error_hint="",
            graph_stats={
                "n_forward": graph.n_forward,
                "n_backward": graph.n_backward,
                "n_edges": len(graph.edges),
                "score_shape_0": int(scores.shape[0]),
                "score_shape_1": int(scores.shape[1]),
            },
        )
    except Exception as exc:
        hint = ""
        if model is not None and "unsupported hf architecture" in str(exc).lower():
            try:
                diag = inspect_model_architecture(model)
                selected = diag.get("selected")
                if selected:
                    hint = f"selected={selected.get('adapter')}@{selected.get('path')}"
                else:
                    attempts = diag.get("adapter_attempts", [])
                    if attempts:
                        top = attempts[0]
                        hint = (
                            f"attempt={top.get('adapter')}@{top.get('path')}: "
                            f"{top.get('detail', '')}"
                        )
            except Exception:
                hint = ""
        return SmokeRow(
            model_id=model_id,
            modality="multimodal",
            adapter_name="",
            language_trunk_path="",
            arch_kind="",
            status="fail",
            seconds=time.time() - start,
            error_type=classify_error(exc),
            error_message=str(exc),
            resolution_error_hint=hint,
            graph_stats=None,
        )


def run_audio_smoke_model(
    model_id: str,
    *,
    device: str,
    dtype: torch.dtype,
    token: Optional[str],
    audio_fallback_model: Optional[str],
) -> SmokeRow:
    start = time.time()
    model = None
    requested_model_id = model_id
    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if token:
        kwargs["token"] = token

    def _attempt_audio(current_model_id: str) -> SmokeRow:
        nonlocal model
        audio = _build_demo_audio()
        if "ultravox" in current_model_id.lower():
            model = _load_audio_model(current_model_id, dtype=dtype, token=token).to(device=device)
            model.eval()
            attribution_model = AttributionModel.from_model(
                model,
                device=device,
                dtype=dtype,
            )
            route_info = attribution_model.route_info
            infer_pipe = pipeline(model=current_model_id, trust_remote_code=True, token=token)
            pair_preparer = UltravoxPairPreparer(
                infer_pipe=infer_pipe,
                device=attribution_model.backend.config.device,
            )
            turns_clean = [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "Summarize the spoken content in one sentence."},
            ]
            turns_corrupt = [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "Transcribe the spoken content."},
            ]
            prepared = pair_preparer.prepare_batch(
                clean_samples={"audio": audio, "sampling_rate": 16000, "turns": turns_clean},
                corrupt_samples={"audio": audio, "sampling_rate": 16000, "turns": turns_corrupt},
                labels=None,
            )
            scores = attribution_model.attribute(
                batches=[prepared],
                metric=smoke_metric,
                method="smoke",
                quiet=True,
            ).scores
        else:
            processor = AutoProcessor.from_pretrained(current_model_id, **kwargs)
            model = _load_audio_model(current_model_id, dtype=dtype, token=token).to(device=device)
            model.eval()
            attribution_model = AttributionModel.from_model(
                model,
                tokenizer=getattr(processor, "tokenizer", None),
                device=device,
                dtype=dtype,
            )
            route_info = attribution_model.route_info
            clean_prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Summarize the audio:"
            corrupt_prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Transcribe the audio:"
            preparer = HFProcessorAdapter(
                processor=processor,
                device=attribution_model.backend.config.device,
            )
            prepared = preparer.prepare_batch(
                clean_samples={"text": clean_prompt, "audio": audio, "sampling_rate": 16000},
                corrupt_samples={"text": corrupt_prompt, "audio": audio, "sampling_rate": 16000},
                labels=None,
            )
            scores = attribution_model.attribute(
                batches=[prepared],
                metric=smoke_metric,
                method="smoke",
                quiet=True,
            ).scores

        graph = attribution_model.build_graph()
        return SmokeRow(
            model_id=requested_model_id,
            modality="multimodal",
            adapter_name=route_info.adapter_name,
            language_trunk_path=route_info.language_trunk_path,
            arch_kind=route_info.arch_kind,
            status="pass",
            seconds=time.time() - start,
            error_type="",
            error_message="",
            resolution_error_hint=(
                ""
                if requested_model_id == current_model_id
                else f"audio fallback used: {current_model_id}"
            ),
            graph_stats={
                "n_forward": graph.n_forward,
                "n_backward": graph.n_backward,
                "n_edges": len(graph.edges),
                "score_shape_0": int(scores.shape[0]),
                "score_shape_1": int(scores.shape[1]),
            },
        )

    try:
        return _attempt_audio(model_id)
    except Exception as primary_exc:
        if (
            audio_fallback_model
            and audio_fallback_model != model_id
            and "ultravox" in audio_fallback_model.lower()
        ):
            try:
                return _attempt_audio(audio_fallback_model)
            except Exception as fallback_exc:
                primary_message = str(primary_exc)
                fallback_message = str(fallback_exc)
                return SmokeRow(
                    model_id=requested_model_id,
                    modality="multimodal",
                    adapter_name="",
                    language_trunk_path="",
                    arch_kind="",
                    status="fail",
                    seconds=time.time() - start,
                    error_type=classify_error(fallback_exc),
                    error_message=(
                        f"primary={requested_model_id}: {primary_message}; "
                        f"fallback={audio_fallback_model}: {fallback_message}"
                    ),
                    resolution_error_hint=f"audio fallback attempted: {audio_fallback_model}",
                    graph_stats=None,
                )

        return SmokeRow(
            model_id=requested_model_id,
            modality="multimodal",
            adapter_name="",
            language_trunk_path="",
            arch_kind="",
            status="fail",
            seconds=time.time() - start,
            error_type=classify_error(primary_exc),
            error_message=str(primary_exc),
            resolution_error_hint="",
            graph_stats=None,
        )


def run_matrix(
    *,
    text_models: Iterable[str],
    multimodal_models: Iterable[str],
    device: str,
    dtype: torch.dtype,
    hf_token: Optional[str],
    audio_fallback_model: Optional[str],
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
        rows.append(
            run_multimodal_smoke_model(
                model_id,
                device=device,
                dtype=dtype,
                token=hf_token,
                audio_fallback_model=audio_fallback_model,
            )
        )

    return {
        "all_passed": all(row.status == "pass" for row in rows),
        "results": [asdict(row) for row in rows],
    }


def main() -> None:
    args = parse_args()
    text_models = parse_model_list(args.text_models)
    multimodal_models = parse_model_list(args.multimodal_models)
    core_report = run_matrix(
        text_models=text_models,
        multimodal_models=multimodal_models,
        device=args.device,
        dtype=to_dtype(args.dtype),
        hf_token=args.hf_token,
        audio_fallback_model=args.audio_fallback_model,
        quiet=args.quiet,
    )
    report = {
        "report_type": REPORT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "text_models": text_models,
            "multimodal_models": multimodal_models,
            "device": args.device,
            "dtype": args.dtype,
            "audio_fallback_model": args.audio_fallback_model,
        },
        **core_report,
    }

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
