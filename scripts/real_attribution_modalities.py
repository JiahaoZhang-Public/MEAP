#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from urllib.request import urlopen

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

from multimodal_lm_eap_ig import (  # noqa: E402
    HFLLMBackend,
    PreparedBatch,
    attribute_from_dataloader,
)
from multimodal_lm_eap_ig.batch import PairBatchPreparer, validate_prepared_batch  # noqa: E402

REPORT_TYPE = "real_attribution_modalities"
SCHEMA_VERSION = "1.0.0"
DEFAULT_MODALITIES = ["text", "image", "audio"]
DEFAULT_TEXT_MODEL = "distilgpt2"
DEFAULT_IMAGE_MODEL = "Qwen/Qwen2-VL-2B"
DEFAULT_AUDIO_MODEL = "fixie-ai/ultravox-v0_5-llama-3_2-1b"
DEFAULT_IMAGE_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"
)
DEFAULT_AUDIO_URL = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"
)


@dataclass
class ModalityResult:
    modality: str
    model_id: str
    method: str
    status: str
    seconds: float
    error_type: str
    error_message: str
    graph_stats: Optional[Dict[str, Any]]
    artifacts: Dict[str, str]


class VisualizationDependencyError(RuntimeError):
    """Raised when graph image rendering dependencies are missing."""


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


@dataclass
class GraphArtifacts:
    full_json: str
    topn_json: str
    topn_png: str


@dataclass
class GraphStats:
    n_forward: int
    n_backward: int
    n_edges_total: int
    n_edges_in_topn_graph: int
    score_shape_0: int
    score_shape_1: int


@dataclass
class RunContext:
    device: str
    dtype: torch.dtype
    method: str
    topn: int
    hf_token: Optional[str]
    image_model: str
    text_model: str
    audio_model: str
    image_url: str
    audio_url: str
    image_path: str
    audio_path: str
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real attribution experiments on text/image/audio.")
    parser.add_argument("--modalities", default=",".join(DEFAULT_MODALITIES))
    parser.add_argument(
        "--method",
        default="EAP",
        choices=["EAP", "EAP-IG-inputs", "clean-corrupted", "EAP-IG-activations", "exact", "smoke"],
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--topn", type=int, default=200)
    parser.add_argument("--output-dir", default="reports/real_attribution")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--image-model", default=DEFAULT_IMAGE_MODEL)
    parser.add_argument("--audio-model", default=DEFAULT_AUDIO_MODEL)
    parser.add_argument("--image-url", default=DEFAULT_IMAGE_URL)
    parser.add_argument("--audio-url", default=DEFAULT_AUDIO_URL)
    parser.add_argument("--image-path", default="")
    parser.add_argument("--audio-path", default="")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def to_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def parse_modalities(raw: str) -> List[str]:
    modalities = [item.strip() for item in raw.split(",") if item.strip()]
    unsupported = [m for m in modalities if m not in {"text", "image", "audio"}]
    if unsupported:
        raise ValueError(f"Unsupported modality values: {unsupported}")
    if len(modalities) == 0:
        raise ValueError("--modalities must contain at least one modality")
    return modalities


def classify_error(exc: Exception) -> str:
    text = str(exc).lower()
    if "pygraphviz" in text:
        return "dependency_missing"
    if isinstance(exc, VisualizationDependencyError):
        return "dependency_missing"
    if "urlopen" in text or "http error" in text or "timed out" in text:
        return "sample_fetch"
    if "from_pretrained" in text or "configuration class" in text:
        return "model_load"
    if "unsupported hf architecture" in text:
        return "unsupported_arch"
    if "out of memory" in text or "cuda oom" in text:
        return "oom"
    return "runtime"


def _single_token_id(tokenizer: Any, token: str) -> Optional[int]:
    try:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
    except Exception:
        return None
    if len(token_ids) == 1:
        return int(token_ids[0])
    return None


def resolve_target_pair(tokenizer: Optional[Any], model: Any) -> torch.Tensor:
    if tokenizer is not None:
        candidates = [(" Paris", " Berlin"), (" yes", " no"), (".", ",")]
        for pos, neg in candidates:
            pos_id = _single_token_id(tokenizer, pos)
            neg_id = _single_token_id(tokenizer, neg)
            if pos_id is not None and neg_id is not None:
                return torch.tensor([[pos_id, neg_id]], dtype=torch.long)

    eos_id = getattr(getattr(model, "config", None), "eos_token_id", None)
    vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", 0) or 0)
    if eos_id is not None and vocab_size > 1:
        alt = int((int(eos_id) + 1) % vocab_size)
        return torch.tensor([[int(eos_id), alt]], dtype=torch.long)
    raise ValueError("Could not resolve a valid single-token target pair for metric computation.")


def metric_logit_diff(logits: torch.Tensor, clean_logits: Optional[torch.Tensor], batch: PreparedBatch) -> torch.Tensor:
    del clean_logits
    labels = batch.labels
    if labels is None:
        raise ValueError("labels are required for metric_logit_diff")

    if not torch.is_tensor(labels):
        labels = torch.tensor(labels, dtype=torch.long, device=logits.device)
    else:
        labels = labels.to(device=logits.device, dtype=torch.long)

    if labels.ndim == 1:
        labels = labels.unsqueeze(0)
    if labels.shape[-1] != 2:
        raise ValueError(f"labels must have shape [batch, 2], got {tuple(labels.shape)}")

    input_lengths = batch.input_lengths.to(device=logits.device, dtype=torch.long)
    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    final_logits = logits[batch_idx, input_lengths - 1]
    selected = torch.gather(final_logits, dim=-1, index=labels)
    return -(selected[:, 0] - selected[:, 1]).mean()


def _pad_2d(tensor: torch.Tensor, target_len: int, pad_value: int) -> torch.Tensor:
    pad = target_len - int(tensor.shape[1])
    if pad <= 0:
        return tensor
    return torch.nn.functional.pad(tensor, (0, pad), value=pad_value)


def prepare_text_batch(
    tokenizer: Any,
    clean_text: str,
    corrupt_text: str,
    labels: torch.Tensor,
) -> PreparedBatch:
    clean = tokenizer([clean_text], return_tensors="pt", padding=True)
    corrupt = tokenizer([corrupt_text], return_tensors="pt", padding=True)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    target_len = max(int(clean["input_ids"].shape[1]), int(corrupt["input_ids"].shape[1]))
    clean_ids = _pad_2d(clean["input_ids"], target_len, int(pad_token_id))
    corrupt_ids = _pad_2d(corrupt["input_ids"], target_len, int(pad_token_id))
    clean_mask = _pad_2d(clean["attention_mask"], target_len, 0)
    corrupt_mask = _pad_2d(corrupt["attention_mask"], target_len, 0)
    shared_mask = torch.minimum(clean_mask, corrupt_mask)

    prepared = PreparedBatch(
        clean_inputs={"input_ids": clean_ids, "attention_mask": shared_mask},
        corrupt_inputs={"input_ids": corrupt_ids, "attention_mask": shared_mask.clone()},
        labels=labels,
        input_lengths=shared_mask.sum(dim=-1),
    )
    validate_prepared_batch(prepared)
    return prepared


def load_image(image_path: str, image_url: str):
    from PIL import Image

    if image_path:
        return Image.open(image_path).convert("RGB")
    with urlopen(image_url, timeout=60) as response:
        data = response.read()
    return Image.open(BytesIO(data)).convert("RGB")


def load_audio(audio_path: str, audio_url: str, sampling_rate: int = 16000):
    import librosa

    if audio_path:
        audio, sr = librosa.load(audio_path, sr=sampling_rate)
        return audio, sr

    with urlopen(audio_url, timeout=60) as response:
        data = response.read()
    audio, sr = librosa.load(BytesIO(data), sr=sampling_rate)
    return audio, sr


def _build_image_prompt(processor: Any, question: str) -> str:
    if hasattr(processor, "apply_chat_template"):
        processor_name = processor.__class__.__name__.lower()
        if "qwen2vl" in processor_name or "qwen2_vl" in processor_name:
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


def _export_graph_artifacts(graph, modality: str, output_dir: Path, topn: int) -> tuple[GraphArtifacts, GraphStats]:
    full_json = output_dir / f"{modality}_graph_full.json"
    topn_json = output_dir / f"{modality}_graph_topn.json"
    topn_png = output_dir / f"{modality}_graph_topn.png"

    graph.reset(empty=False)
    graph.to_json(str(full_json))

    n_edges_total = int(graph.real_edge_mask.sum().item())
    n_keep = max(1, min(int(topn), n_edges_total))
    graph.apply_topn(n_keep, absolute=True, reset=True, prune=True)
    graph.to_json(str(topn_json))

    try:
        graph.to_image(str(topn_png))
    except Exception as exc:  # noqa: BLE001
        if "pygraphviz" in str(exc).lower() or isinstance(exc, ModuleNotFoundError):
            raise VisualizationDependencyError(
                "pygraphviz is required for graph PNG export. Install with `pip install pygraphviz`."
            ) from exc
        raise

    artifacts = GraphArtifacts(
        full_json=str(full_json),
        topn_json=str(topn_json),
        topn_png=str(topn_png),
    )
    stats = GraphStats(
        n_forward=int(graph.n_forward),
        n_backward=int(graph.n_backward),
        n_edges_total=n_edges_total,
        n_edges_in_topn_graph=int(graph.count_included_edges()),
        score_shape_0=int(graph.scores.shape[0]),
        score_shape_1=int(graph.scores.shape[1]),
    )
    return artifacts, stats


def _run_text(ctx: RunContext) -> ModalityResult:
    start = time.time()
    model_id = ctx.text_model
    artifacts: Dict[str, str] = {}
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=ctx.hf_token)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=ctx.dtype, token=ctx.hf_token).to(ctx.device)
        model.eval()

        labels = resolve_target_pair(tokenizer, model)
        prepared = prepare_text_batch(
            tokenizer,
            clean_text="The capital of France is",
            corrupt_text="The capital of Germany is",
            labels=labels,
        )

        backend = HFLLMBackend(model, tokenizer=tokenizer)
        result = attribute_from_dataloader(
            model=model,
            backend=backend,
            dataloader=[prepared],
            metric=metric_logit_diff,
            method=ctx.method,
            quiet=True,
        )

        graph_artifacts, graph_stats = _export_graph_artifacts(
            result.graph,
            modality="text",
            output_dir=ctx.output_dir,
            topn=ctx.topn,
        )
        artifacts = asdict(graph_artifacts)

        summary = {
            "modality": "text",
            "model_id": model_id,
            "method": ctx.method,
            "graph_stats": asdict(graph_stats),
            "artifacts": artifacts,
        }
        summary_path = ctx.output_dir / "text_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        artifacts["summary_json"] = str(summary_path)

        return ModalityResult(
            modality="text",
            model_id=model_id,
            method=ctx.method,
            status="pass",
            seconds=time.time() - start,
            error_type="",
            error_message="",
            graph_stats=asdict(graph_stats),
            artifacts=artifacts,
        )
    except Exception as exc:  # noqa: BLE001
        return ModalityResult(
            modality="text",
            model_id=model_id,
            method=ctx.method,
            status="fail",
            seconds=time.time() - start,
            error_type=classify_error(exc),
            error_message=str(exc),
            graph_stats=None,
            artifacts=artifacts,
        )


def _run_image(ctx: RunContext) -> ModalityResult:
    start = time.time()
    model_id = ctx.image_model
    artifacts: Dict[str, str] = {}
    try:
        processor = AutoProcessor.from_pretrained(model_id, token=ctx.hf_token, trust_remote_code=True)
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                dtype=ctx.dtype,
                token=ctx.hf_token,
                trust_remote_code=True,
            )
        except Exception:
            from transformers import AutoModelForVision2Seq

            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                dtype=ctx.dtype,
                token=ctx.hf_token,
                trust_remote_code=True,
            )
        model = model.to(ctx.device)
        model.eval()

        tokenizer = getattr(processor, "tokenizer", None)
        labels = resolve_target_pair(tokenizer, model)
        image = load_image(ctx.image_path, ctx.image_url)

        clean_prompt = _build_image_prompt(processor, "What animal is shown on the candy?")
        corrupt_prompt = _build_image_prompt(processor, "What object is this scene mainly about?")

        dataloader = [
            {
                "clean": [{"text": clean_prompt, "images": image}],
                "corrupt": [{"text": corrupt_prompt, "images": image}],
                "labels": labels,
            }
        ]

        backend = HFLLMBackend(model, tokenizer=tokenizer)
        result = attribute_from_dataloader(
            model=model,
            backend=backend,
            dataloader=dataloader,
            processor=processor,
            processor_kwargs={"padding": True},
            metric=metric_logit_diff,
            method=ctx.method,
            quiet=True,
        )

        graph_artifacts, graph_stats = _export_graph_artifacts(
            result.graph,
            modality="image",
            output_dir=ctx.output_dir,
            topn=ctx.topn,
        )
        artifacts = asdict(graph_artifacts)

        summary = {
            "modality": "image",
            "model_id": model_id,
            "method": ctx.method,
            "graph_stats": asdict(graph_stats),
            "artifacts": artifacts,
            "sample": {"image_path": ctx.image_path, "image_url": ctx.image_url},
        }
        summary_path = ctx.output_dir / "image_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        artifacts["summary_json"] = str(summary_path)

        return ModalityResult(
            modality="image",
            model_id=model_id,
            method=ctx.method,
            status="pass",
            seconds=time.time() - start,
            error_type="",
            error_message="",
            graph_stats=asdict(graph_stats),
            artifacts=artifacts,
        )
    except Exception as exc:  # noqa: BLE001
        return ModalityResult(
            modality="image",
            model_id=model_id,
            method=ctx.method,
            status="fail",
            seconds=time.time() - start,
            error_type=classify_error(exc),
            error_message=str(exc),
            graph_stats=None,
            artifacts=artifacts,
        )


def _run_audio(ctx: RunContext) -> ModalityResult:
    start = time.time()
    model_id = ctx.audio_model
    artifacts: Dict[str, str] = {}
    try:
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=ctx.dtype,
            token=ctx.hf_token,
        ).to(ctx.device)
        model.eval()

        backend = HFLLMBackend(model)
        infer_pipe = pipeline(model=model_id, trust_remote_code=True, token=ctx.hf_token)
        preparer = UltravoxPairPreparer(infer_pipe=infer_pipe, device=backend.config.device)

        audio, sr = load_audio(ctx.audio_path, ctx.audio_url, sampling_rate=16000)
        labels = resolve_target_pair(getattr(infer_pipe, "tokenizer", None), model)

        turns_clean = [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Summarize the spoken content in one sentence."},
        ]
        turns_corrupt = [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Transcribe the spoken content."},
        ]
        dataloader = [
            {
                "clean": {"audio": audio, "sampling_rate": sr, "turns": turns_clean},
                "corrupt": {"audio": audio, "sampling_rate": sr, "turns": turns_corrupt},
                "labels": labels,
            }
        ]

        result = attribute_from_dataloader(
            model=model,
            backend=backend,
            dataloader=dataloader,
            pair_batch_preparer=preparer,
            metric=metric_logit_diff,
            method=ctx.method,
            quiet=True,
        )

        graph_artifacts, graph_stats = _export_graph_artifacts(
            result.graph,
            modality="audio",
            output_dir=ctx.output_dir,
            topn=ctx.topn,
        )
        artifacts = asdict(graph_artifacts)

        summary = {
            "modality": "audio",
            "model_id": model_id,
            "method": ctx.method,
            "graph_stats": asdict(graph_stats),
            "artifacts": artifacts,
            "sample": {"audio_path": ctx.audio_path, "audio_url": ctx.audio_url, "sampling_rate": sr},
        }
        summary_path = ctx.output_dir / "audio_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        artifacts["summary_json"] = str(summary_path)

        return ModalityResult(
            modality="audio",
            model_id=model_id,
            method=ctx.method,
            status="pass",
            seconds=time.time() - start,
            error_type="",
            error_message="",
            graph_stats=asdict(graph_stats),
            artifacts=artifacts,
        )
    except Exception as exc:  # noqa: BLE001
        return ModalityResult(
            modality="audio",
            model_id=model_id,
            method=ctx.method,
            status="fail",
            seconds=time.time() - start,
            error_type=classify_error(exc),
            error_message=str(exc),
            graph_stats=None,
            artifacts=artifacts,
        )


def run_modalities(modalities: Iterable[str], ctx: RunContext, quiet: bool = False) -> List[ModalityResult]:
    runners = {
        "text": _run_text,
        "image": _run_image,
        "audio": _run_audio,
    }
    results: List[ModalityResult] = []
    for modality in modalities:
        if not quiet:
            print(f"[run] modality={modality} model={getattr(ctx, f'{modality}_model')}")
        results.append(runners[modality](ctx))
    return results


def _write_report(results: Sequence[ModalityResult], ctx: RunContext, modalities: Sequence[str]) -> Dict[str, Any]:
    report = {
        "report_type": REPORT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "modalities": list(modalities),
            "method": ctx.method,
            "device": ctx.device,
            "dtype": str(ctx.dtype).replace("torch.", ""),
            "topn": ctx.topn,
            "models": {
                "text": ctx.text_model,
                "image": ctx.image_model,
                "audio": ctx.audio_model,
            },
            "samples": {
                "image_path": ctx.image_path,
                "image_url": ctx.image_url,
                "audio_path": ctx.audio_path,
                "audio_url": ctx.audio_url,
            },
        },
        "all_passed": all(result.status == "pass" for result in results),
        "results": [asdict(result) for result in results],
    }
    output_path = ctx.output_dir / "real_attribution_modalities.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    modalities = parse_modalities(args.modalities)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx = RunContext(
        device=args.device,
        dtype=to_dtype(args.dtype),
        method=args.method,
        topn=args.topn,
        hf_token=args.hf_token,
        image_model=args.image_model,
        text_model=args.text_model,
        audio_model=args.audio_model,
        image_url=args.image_url,
        audio_url=args.audio_url,
        image_path=args.image_path,
        audio_path=args.audio_path,
        output_dir=output_dir,
    )

    results = run_modalities(modalities, ctx, quiet=args.quiet)
    report = _write_report(results, ctx, modalities)

    print(json.dumps(report, indent=2))
    print(f"Saved report to {output_dir / 'real_attribution_modalities.json'}")

    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
