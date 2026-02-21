#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlparse
from urllib.request import urlretrieve

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
    HFProcessorAdapter,
    PreparedBatch,
)
from meap.batch import PairBatchPreparer, validate_prepared_batch  # noqa: E402

DEFAULT_AUDIO_PROMPT = "Generate the caption in English:"
DEFAULT_AUDIO_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/glass-breaking-151256.mp3"
ULTRAVOX_AUDIO_PLACEHOLDER = "<|audio|>"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal API examples for text/image/audio attribution.")
    parser.add_argument(
        "--example",
        choices=["text-gpt2", "text-qwen2", "image-qwen2vl", "audio-ultravox"],
        required=True,
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument(
        "--method",
        default="smoke",
        choices=["smoke", "EAP", "EAP-IG-inputs", "clean-corrupted", "EAP-IG-activations", "exact"],
    )
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--audio-path", default="", help="Optional local audio path for audio-ultravox example.")
    parser.add_argument("--audio-url", default=DEFAULT_AUDIO_URL, help="Audio URL used when --audio-path is empty.")
    parser.add_argument(
        "--audio-prompt",
        default=DEFAULT_AUDIO_PROMPT,
        help="User prompt text used for default audio-ultravox test.",
    )
    parser.add_argument("--output", default="", help="Optional path to write JSON summary.")
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _metric(logits, clean_logits, batch):
    del clean_logits, batch
    return logits.sum()


def _pad_to_length(tensor: torch.Tensor, target_len: int, pad_value: int) -> torch.Tensor:
    pad = target_len - int(tensor.shape[1])
    if pad <= 0:
        return tensor
    return torch.nn.functional.pad(tensor, (0, pad), value=pad_value)


def _text_prepared_batch(tokenizer, clean_text: str, corrupt_text: str, max_length: int = 128) -> PreparedBatch:
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

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    target_len = max(int(clean["input_ids"].shape[1]), int(corrupt["input_ids"].shape[1]))
    clean_ids = _pad_to_length(clean["input_ids"], target_len, int(pad_id))
    corrupt_ids = _pad_to_length(corrupt["input_ids"], target_len, int(pad_id))
    clean_mask = _pad_to_length(clean["attention_mask"], target_len, 0)
    corrupt_mask = _pad_to_length(corrupt["attention_mask"], target_len, 0)

    # PreparedBatch requires aligned semantics between clean/corrupt masks.
    shared_mask = torch.minimum(clean_mask, corrupt_mask)
    batch = PreparedBatch(
        clean_inputs={"input_ids": clean_ids, "attention_mask": shared_mask},
        corrupt_inputs={"input_ids": corrupt_ids, "attention_mask": shared_mask.clone()},
        labels=torch.tensor([0]),
        input_lengths=shared_mask.sum(dim=-1),
    )
    validate_prepared_batch(batch)
    return batch


def _summarize(example: str, method: str, result) -> Dict[str, Any]:
    return {
        "example": example,
        "method": method,
        "graph": {
            "n_forward": result.graph.n_forward,
            "n_backward": result.graph.n_backward,
            "n_edges": len(result.graph.edges),
        },
        "scores": {
            "shape": [int(result.scores.shape[0]), int(result.scores.shape[1])],
            "min": float(result.scores.min().item()),
            "max": float(result.scores.max().item()),
            "mean": float(result.scores.mean().item()),
        },
    }


def _print_summary(summary: Dict[str, Any]) -> None:
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _resolve_audio_path(args: argparse.Namespace) -> str:
    if args.audio_path:
        return args.audio_path

    parsed = urlparse(args.audio_url)
    filename = Path(parsed.path).name or "sample_audio.mp3"
    download_path = REPO_ROOT / ".cache" / "demo_audio" / filename
    download_path.parent.mkdir(parents=True, exist_ok=True)
    if not download_path.exists():
        print(f"Downloading default audio from {args.audio_url} -> {download_path}")
        urlretrieve(args.audio_url, download_path)
    return str(download_path)


def _normalize_ultravox_user_prompt(prompt: str, *, audio_placeholder: str) -> str:
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


def _run_text_gpt2(args: argparse.Namespace) -> Dict[str, Any]:
    model_id = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=args.hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=_dtype(args.dtype), token=args.hf_token).to(
        args.device
    )
    model.eval()
    attr_model = AttributionModel.from_model(model, tokenizer=tokenizer, device=args.device, dtype=_dtype(args.dtype))

    batch = _text_prepared_batch(
        tokenizer,
        clean_text="The capital of France is",
        corrupt_text="The capital of Germany is",
    )
    result = attr_model.attribute(
        batches=[batch],
        metric=_metric,
        method=args.method,
    )
    return _summarize("text-gpt2", args.method, result)


def _run_text_qwen2(args: argparse.Namespace) -> Dict[str, Any]:
    model_id = "Qwen/Qwen2-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=args.hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=_dtype(args.dtype), token=args.hf_token).to(
        args.device
    )
    model.eval()
    attr_model = AttributionModel.from_model(model, tokenizer=tokenizer, device=args.device, dtype=_dtype(args.dtype))

    clean_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Who are you?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    corrupt_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is your name?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    batch = _text_prepared_batch(tokenizer, clean_prompt, corrupt_prompt)
    result = attr_model.attribute(
        batches=[batch],
        metric=_metric,
        method=args.method,
    )
    return _summarize("text-qwen2", args.method, result)


def _qwen2_vl_prompt(processor, question: str) -> str:
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
                prompt = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                continue
            if isinstance(prompt, str) and prompt.strip():
                if any(tok in prompt for tok in ("<|image_pad|>", "<image>", "<im_patch>", "<image_pad>")):
                    return prompt
    return f"<|vision_start|><|image_pad|><|vision_end|>\n{question}"


def _run_image_qwen2_vl(args: argparse.Namespace) -> Dict[str, Any]:
    model_id = "Qwen/Qwen2-VL-2B"
    processor = AutoProcessor.from_pretrained(model_id, token=args.hf_token, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=_dtype(args.dtype),
        token=args.hf_token,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()
    attr_model = AttributionModel.from_model(
        model,
        tokenizer=getattr(processor, "tokenizer", None),
        device=args.device,
        dtype=_dtype(args.dtype),
    )

    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[48:176, 48:176, :] = 255
    clean_prompt = _qwen2_vl_prompt(processor, "Is the square white?")
    corrupt_prompt = _qwen2_vl_prompt(processor, "Is the square black?")

    preparer = HFProcessorAdapter(
        processor=processor,
        processor_kwargs={"padding": True},
        device=attr_model.backend.config.device,
    )
    batch = preparer.prepare_batch(
        clean_samples=[{"text": clean_prompt, "images": image}],
        corrupt_samples=[{"text": corrupt_prompt, "images": image}],
        labels=torch.tensor([0]),
    )
    result = attr_model.attribute(
        batches=[batch],
        metric=_metric,
        method=args.method,
    )
    return _summarize("image-qwen2vl", args.method, result)


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
        if len(clean_samples) != 1 or len(corrupt_samples) != 1:
            raise ValueError("Ultravox minimal example currently supports batch size 1.")
        clean_inputs = self._encode_one(clean_samples[0])
        corrupt_inputs = self._encode_one(corrupt_samples[0])
        if "attention_mask" not in clean_inputs or "attention_mask" not in corrupt_inputs:
            raise ValueError("Ultravox preparer requires attention_mask in both clean and corrupt.")
        input_lengths = clean_inputs["attention_mask"].sum(dim=-1)
        batch = PreparedBatch(
            clean_inputs=clean_inputs,
            corrupt_inputs=corrupt_inputs,
            labels=labels,
            input_lengths=input_lengths,
            meta=meta,
        )
        validate_prepared_batch(batch)
        return batch


def _run_audio_ultravox(args: argparse.Namespace) -> Dict[str, Any]:
    import librosa

    audio_path = _resolve_audio_path(args)
    audio, sr = librosa.load(audio_path, sr=16000)
    prompt = str(args.audio_prompt).strip() or DEFAULT_AUDIO_PROMPT

    model_id = "fixie-ai/ultravox-v0_5-llama-3_2-1b"
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=_dtype(args.dtype),
        token=args.hf_token,
    ).to(args.device)
    model.eval()

    # Ultravox has custom preprocessing; build PreparedBatch explicitly.
    infer_pipe = pipeline(model=model_id, trust_remote_code=True, token=args.hf_token)
    attr_model = AttributionModel.from_model(model, device=args.device, dtype=_dtype(args.dtype))
    pair_preparer = UltravoxPairPreparer(infer_pipe=infer_pipe, device=attr_model.backend.config.device)
    audio_placeholder = getattr(getattr(infer_pipe, "processor", None), "audio_placeholder", ULTRAVOX_AUDIO_PLACEHOLDER)
    clean_user_prompt = _normalize_ultravox_user_prompt(prompt, audio_placeholder=audio_placeholder)
    corrupt_user_prompt = _normalize_ultravox_user_prompt(
        "Transcribe the spoken content.",
        audio_placeholder=audio_placeholder,
    )
    turns_clean = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": clean_user_prompt},
    ]
    turns_corrupt = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": corrupt_user_prompt},
    ]
    batch = pair_preparer.prepare_batch(
        clean_samples=[{"audio": audio, "sampling_rate": sr, "turns": turns_clean}],
        corrupt_samples=[{"audio": audio, "sampling_rate": sr, "turns": turns_corrupt}],
        labels=torch.tensor([0]),
    )
    result = attr_model.attribute(
        batches=[batch],
        metric=_metric,
        method=args.method,
    )
    return _summarize("audio-ultravox", args.method, result)


def main() -> Dict[str, Any]:
    args = _parse_args()
    if args.example == "text-gpt2":
        summary = _run_text_gpt2(args)
    elif args.example == "text-qwen2":
        summary = _run_text_qwen2(args)
    elif args.example == "image-qwen2vl":
        summary = _run_image_qwen2_vl(args)
    elif args.example == "audio-ultravox":
        summary = _run_audio_ultravox(args)
    else:
        raise ValueError(f"Unsupported example: {args.example}")

    _print_summary(summary)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved summary to {out}")
    return summary


if __name__ == "__main__":
    main()
