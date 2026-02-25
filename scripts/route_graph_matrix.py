#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
from torch import Tensor
from transformers import AutoModel, AutoModelForImageTextToText

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from meap import (  # noqa: E402
    AttributionModel,
    PreparedBatch,
    inspect_model_architecture,
    list_official_models,
)
from meap.utils import make_hooks_and_matrices  # noqa: E402

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
REPORT_TYPE = "route_graph_matrix"
SCHEMA_VERSION = "1.0.0"


@dataclass
class RouteGraphRow:
    model_id: str
    modality: str
    entrypoint: str
    model_loader: str
    adapter_name: str
    language_trunk_path: str
    arch_kind: str
    status: str
    seconds: float
    error_type: str
    error_message: str
    resolution_error_hint: str
    graph_stats: Optional[Dict[str, int]]
    hook_stats: Optional[Dict[str, Any]]
    attribute_smoke: Optional[Dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate model -> language trunk -> graph/hook route for a text+multimodal matrix."
        )
    )
    parser.add_argument("--text-models", default=",".join(DEFAULT_TEXT_MODELS))
    parser.add_argument("--multimodal-models", default=",".join(DEFAULT_MULTIMODAL_MODELS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--hf-token", default=None)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--run-attribute-smoke",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run one minimal PreparedBatch attribution smoke per model (method=smoke).",
    )
    parser.add_argument("--output", default="reports/route_graph_matrix.json")
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
    if "from_pretrained" in text or "configuration class" in text:
        return "model_load"
    return "runtime"


def _model_kwargs(
    *,
    dtype: torch.dtype,
    token: Optional[str],
    trust_remote_code: bool,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": trust_remote_code,
    }
    if token:
        kwargs["token"] = token
    return kwargs


def _resolution_hint_from_diagnostics(diagnostics: Mapping[str, Any]) -> str:
    selected = diagnostics.get("selected")
    if isinstance(selected, Mapping):
        adapter = selected.get("adapter", "?")
        path = selected.get("path", "?")
        return f"selected={adapter}@{path}"

    attempts = diagnostics.get("adapter_attempts", [])
    if isinstance(attempts, list):
        for attempt in attempts:
            if isinstance(attempt, Mapping) and attempt.get("status") != "match":
                return (
                    f"attempt={attempt.get('adapter', '?')}@{attempt.get('path', '?')}: "
                    f"{attempt.get('detail', '')}"
                )
    return ""


def _hook_coverage(attribution_model: AttributionModel, graph) -> Dict[str, Any]:
    backend = attribution_model.backend
    scores = torch.zeros(
        (graph.n_forward, graph.n_backward),
        device=backend.config.device,
        dtype=backend.config.dtype,
    )
    (fwd_hooks_corrupt, fwd_hooks_clean, bwd_hooks), _ = make_hooks_and_matrices(
        backend,
        graph,
        batch_size=1,
        n_pos=2,
        scores=scores,
    )
    required_hooks = sorted(
        {
            name
            for name, _ in [*fwd_hooks_corrupt, *fwd_hooks_clean, *bwd_hooks]
        }
    )
    supported_hooks = sorted(backend.supported_hook_names)
    supported_set = set(supported_hooks)
    missing_hooks = [hook_name for hook_name in required_hooks if hook_name not in supported_set]
    return {
        "required_count": len(required_hooks),
        "supported_count": len(supported_hooks),
        "missing_count": len(missing_hooks),
        "missing_hooks": missing_hooks,
    }


def _graph_stats(graph) -> Dict[str, int]:
    return {
        "n_layers": int(graph.cfg["n_layers"]),
        "n_heads": int(graph.cfg["n_heads"]),
        "d_model": int(graph.cfg["d_model"]),
        "n_forward": int(graph.n_forward),
        "n_backward": int(graph.n_backward),
        "n_edges_total": int(len(graph.edges)),
    }


def _cleanup_model_ref(attr_model: Optional[AttributionModel]) -> None:
    if attr_model is not None:
        del attr_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _bounded_token_id(raw_value: Any, *, vocab_size: int, fallback: int) -> int:
    if isinstance(raw_value, int) and 0 <= raw_value < vocab_size:
        return int(raw_value)
    return int(fallback)


def _resolve_vocab_size(model: torch.nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Cannot build minimal PreparedBatch: model has no config")

    text_config = getattr(config, "text_config", None)
    for cfg in (text_config, config):
        if cfg is None:
            continue
        vocab_size = getattr(cfg, "vocab_size", None)
        if isinstance(vocab_size, int) and vocab_size > 2:
            return int(vocab_size)

    raise ValueError("Cannot infer vocab_size from model config/text_config")


def _build_minimal_prepared_batch(attr_model: AttributionModel) -> PreparedBatch:
    model = attr_model.model
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None) if config is not None else None
    token_source = text_config if text_config is not None else config
    vocab_size = _resolve_vocab_size(model)

    bos_token_id = _bounded_token_id(
        getattr(token_source, "bos_token_id", None),
        vocab_size=vocab_size,
        fallback=1,
    )
    eos_token_id = _bounded_token_id(
        getattr(token_source, "eos_token_id", None),
        vocab_size=vocab_size,
        fallback=min(2, vocab_size - 1),
    )
    token_a = min(3, vocab_size - 1)
    token_b = min(4, vocab_size - 1)
    if token_b == token_a:
        token_b = (token_a + 1) % vocab_size

    device = attr_model.backend.config.device
    clean_ids = torch.tensor([[bos_token_id, token_a, eos_token_id]], dtype=torch.long, device=device)
    corrupt_ids = torch.tensor([[bos_token_id, token_b, eos_token_id]], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(clean_ids, dtype=torch.long, device=device)
    return PreparedBatch(
        clean_inputs={"input_ids": clean_ids, "attention_mask": attention_mask},
        corrupt_inputs={"input_ids": corrupt_ids, "attention_mask": attention_mask.clone()},
        labels=None,
        input_lengths=attention_mask.sum(dim=-1),
    )


def _smoke_metric(logits: Tensor, clean_logits: Optional[Tensor], batch: PreparedBatch) -> Tensor:
    del clean_logits, batch
    return logits.sum()


def _run_attribute_smoke(attr_model: AttributionModel, graph) -> Dict[str, Any]:
    batch = _build_minimal_prepared_batch(attr_model)
    run = attr_model.attribute(
        batches=[batch],
        method="smoke",
        metric=_smoke_metric,
        graph=graph,
        quiet=True,
    )
    return {
        "status": "pass",
        "score_shape_0": int(run.scores.shape[0]),
        "score_shape_1": int(run.scores.shape[1]),
        "n_examples": int(run.run_info["n_examples"]),
    }


def run_text_route_graph_model(
    model_id: str,
    *,
    device: str,
    dtype: torch.dtype,
    token: Optional[str],
    trust_remote_code: bool,
    run_attribute_smoke: bool,
) -> RouteGraphRow:
    start = time.time()
    attr_model: Optional[AttributionModel] = None
    try:
        attr_model = AttributionModel.from_pretrained(
            model_id,
            device=device,
            dtype=dtype,
            model_kwargs=_model_kwargs(
                dtype=dtype,
                token=token,
                trust_remote_code=trust_remote_code,
            ),
            strict_arch=True,
            cache=False,
        )
        route_info = attr_model.route_info
        graph = attr_model.build_graph()
        hook_stats = _hook_coverage(attr_model, graph)
        status = "pass" if hook_stats["missing_count"] == 0 else "fail"
        error_type = ""
        error_message = ""
        if status != "pass":
            error_type = "hook_coverage"
            error_message = f"missing hooks: {hook_stats['missing_hooks']}"

        attribute_smoke = None
        if run_attribute_smoke:
            try:
                attribute_smoke = _run_attribute_smoke(attr_model, graph)
            except Exception as smoke_exc:
                status = "fail"
                if not error_type:
                    error_type = "attribute_smoke"
                    error_message = str(smoke_exc)
                attribute_smoke = {
                    "status": "fail",
                    "error_type": classify_error(smoke_exc),
                    "error_message": str(smoke_exc),
                }

        return RouteGraphRow(
            model_id=model_id,
            modality="text",
            entrypoint="AttributionModel.from_pretrained",
            model_loader="AutoModel",
            adapter_name=route_info.adapter_name,
            language_trunk_path=route_info.language_trunk_path,
            arch_kind=route_info.arch_kind,
            status=status,
            seconds=time.time() - start,
            error_type=error_type,
            error_message=error_message,
            resolution_error_hint=_resolution_hint_from_diagnostics(route_info.diagnostics),
            graph_stats=_graph_stats(graph),
            hook_stats=hook_stats,
            attribute_smoke=attribute_smoke,
        )
    except Exception as exc:
        return RouteGraphRow(
            model_id=model_id,
            modality="text",
            entrypoint="AttributionModel.from_pretrained",
            model_loader="AutoModel",
            adapter_name="",
            language_trunk_path="",
            arch_kind="",
            status="fail",
            seconds=time.time() - start,
            error_type=classify_error(exc),
            error_message=str(exc),
            resolution_error_hint="",
            graph_stats=None,
            hook_stats=None,
            attribute_smoke=None,
        )
    finally:
        _cleanup_model_ref(attr_model)


def _load_multimodal_model(
    model_id: str,
    *,
    dtype: torch.dtype,
    token: Optional[str],
    trust_remote_code: bool,
):
    kwargs = _model_kwargs(dtype=dtype, token=token, trust_remote_code=trust_remote_code)
    lowered = model_id.lower()
    if "qwen2-audio" in lowered:
        from transformers import Qwen2AudioForConditionalGeneration

        return (
            Qwen2AudioForConditionalGeneration.from_pretrained(model_id, **kwargs),
            "Qwen2AudioForConditionalGeneration",
        )
    if "ultravox" in model_id.lower():
        return AutoModel.from_pretrained(model_id, **kwargs), "AutoModel"

    try:
        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs), "AutoModelForImageTextToText"
    except Exception as img_txt_exc:
        try:
            from transformers import AutoModelForVision2Seq

            return AutoModelForVision2Seq.from_pretrained(model_id, **kwargs), "AutoModelForVision2Seq"
        except Exception as vision_seq_exc:
            try:
                return AutoModel.from_pretrained(model_id, **kwargs), "AutoModel"
            except Exception as auto_exc:
                raise RuntimeError(
                    "Failed to load multimodal model with AutoModelForImageTextToText, "
                    "AutoModelForVision2Seq, and AutoModel. "
                    f"errors=({img_txt_exc}; {vision_seq_exc}; {auto_exc})"
                ) from auto_exc


def _resolution_hint_for_model(model: torch.nn.Module) -> str:
    try:
        diagnostics = inspect_model_architecture(model)
    except Exception:
        return ""
    return _resolution_hint_from_diagnostics(diagnostics)


def run_multimodal_route_graph_model(
    model_id: str,
    *,
    device: str,
    dtype: torch.dtype,
    token: Optional[str],
    trust_remote_code: bool,
    run_attribute_smoke: bool,
) -> RouteGraphRow:
    start = time.time()
    attr_model: Optional[AttributionModel] = None
    model = None
    model_loader = ""
    try:
        model, model_loader = _load_multimodal_model(
            model_id,
            dtype=dtype,
            token=token,
            trust_remote_code=trust_remote_code,
        )
        model.eval()

        attr_model = AttributionModel.from_model(
            model,
            device=device,
            dtype=dtype,
            strict_arch=True,
        )
        route_info = attr_model.route_info
        graph = attr_model.build_graph()
        hook_stats = _hook_coverage(attr_model, graph)
        status = "pass" if hook_stats["missing_count"] == 0 else "fail"
        error_type = ""
        error_message = ""
        if status != "pass":
            error_type = "hook_coverage"
            error_message = f"missing hooks: {hook_stats['missing_hooks']}"

        attribute_smoke = None
        if run_attribute_smoke:
            try:
                attribute_smoke = _run_attribute_smoke(attr_model, graph)
            except Exception as smoke_exc:
                status = "fail"
                if not error_type:
                    error_type = "attribute_smoke"
                    error_message = str(smoke_exc)
                attribute_smoke = {
                    "status": "fail",
                    "error_type": classify_error(smoke_exc),
                    "error_message": str(smoke_exc),
                }

        return RouteGraphRow(
            model_id=model_id,
            modality="multimodal",
            entrypoint="AttributionModel.from_model",
            model_loader=model_loader,
            adapter_name=route_info.adapter_name,
            language_trunk_path=route_info.language_trunk_path,
            arch_kind=route_info.arch_kind,
            status=status,
            seconds=time.time() - start,
            error_type=error_type,
            error_message=error_message,
            resolution_error_hint=_resolution_hint_from_diagnostics(route_info.diagnostics),
            graph_stats=_graph_stats(graph),
            hook_stats=hook_stats,
            attribute_smoke=attribute_smoke,
        )
    except Exception as exc:
        return RouteGraphRow(
            model_id=model_id,
            modality="multimodal",
            entrypoint="AttributionModel.from_model",
            model_loader=(model_loader or "unknown"),
            adapter_name="",
            language_trunk_path="",
            arch_kind="",
            status="fail",
            seconds=time.time() - start,
            error_type=classify_error(exc),
            error_message=str(exc),
            resolution_error_hint=("" if model is None else _resolution_hint_for_model(model)),
            graph_stats=None,
            hook_stats=None,
            attribute_smoke=None,
        )
    finally:
        if model is not None:
            del model
        _cleanup_model_ref(attr_model)


def run_matrix(
    *,
    text_models: Iterable[str],
    multimodal_models: Iterable[str],
    device: str,
    dtype: torch.dtype,
    hf_token: Optional[str],
    trust_remote_code: bool,
    run_attribute_smoke: bool,
    quiet: bool,
) -> Dict[str, Any]:
    rows: List[RouteGraphRow] = []

    for model_id in text_models:
        if not quiet:
            print(f"[text] {model_id}")
        rows.append(
            run_text_route_graph_model(
                model_id,
                device=device,
                dtype=dtype,
                token=hf_token,
                trust_remote_code=trust_remote_code,
                run_attribute_smoke=run_attribute_smoke,
            )
        )

    for model_id in multimodal_models:
        if not quiet:
            print(f"[multimodal] {model_id}")
        rows.append(
            run_multimodal_route_graph_model(
                model_id,
                device=device,
                dtype=dtype,
                token=hf_token,
                trust_remote_code=trust_remote_code,
                run_attribute_smoke=run_attribute_smoke,
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
        trust_remote_code=args.trust_remote_code,
        run_attribute_smoke=args.run_attribute_smoke,
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
            "trust_remote_code": args.trust_remote_code,
            "run_attribute_smoke": args.run_attribute_smoke,
        },
        **core_report,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Saved route graph matrix report to {output_path}")

    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
