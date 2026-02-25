#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from meap import list_official_models  # noqa: E402

try:
    from huggingface_hub import snapshot_download
except Exception as exc:  # noqa: BLE001
    snapshot_download = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


REPORT_TYPE = "hf_model_download"
SCHEMA_VERSION = "1.0.0"

ROUTE_MATRIX_TEXT_MODELS = [
    "gpt2",
    "facebook/opt-125m",
    "Qwen/Qwen2-0.5B",
]
ROUTE_MATRIX_MULTIMODAL_MODELS = [
    "Qwen/Qwen2-VL-2B",
    "llava-hf/llava-1.5-7b-hf",
    "fixie-ai/ultravox-v0_5-llama-3_2-1b",
]

V15_TEXT_CANDIDATES = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "tiiuae/falcon-rw-1b",
    "google/gemma-2-2b",
    "microsoft/phi-2",
    "microsoft/Phi-3-mini-4k-instruct",
]
V15_MULTIMODAL_CANDIDATES = [
    "HuggingFaceTB/SmolVLM-Instruct",
    "HuggingFaceM4/idefics2-8b",
    "Qwen/Qwen2-Audio-7B",
]


@dataclass
class DownloadRow:
    model_id: str
    modality: str
    source: str
    status: str
    seconds: float
    snapshot_path: str
    error_type: str
    error_message: str


def _ordered_unique(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _parse_model_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _official_models() -> Tuple[List[str], List[str]]:
    rows = list_official_models()
    text = [row["model_id"] for row in rows if row["modality"] == "text"]
    multimodal = [row["model_id"] for row in rows if row["modality"] == "multimodal"]
    return text, multimodal


def _preset_models(preset: str) -> Tuple[List[str], List[str], Dict[str, str]]:
    official_text, official_multimodal = _official_models()
    source_map: Dict[str, str] = {}

    if preset == "official":
        text_models = official_text
        multimodal_models = official_multimodal
    elif preset == "route-matrix":
        text_models = ROUTE_MATRIX_TEXT_MODELS
        multimodal_models = ROUTE_MATRIX_MULTIMODAL_MODELS
    elif preset == "v1-5-candidates":
        text_models = V15_TEXT_CANDIDATES
        multimodal_models = V15_MULTIMODAL_CANDIDATES
    elif preset == "all-known":
        text_models = [
            *official_text,
            *ROUTE_MATRIX_TEXT_MODELS,
            *V15_TEXT_CANDIDATES,
        ]
        multimodal_models = [
            *official_multimodal,
            *ROUTE_MATRIX_MULTIMODAL_MODELS,
            *V15_MULTIMODAL_CANDIDATES,
        ]
    else:
        raise ValueError(f"Unknown preset: {preset}")

    text_models = _ordered_unique(text_models)
    multimodal_models = _ordered_unique(multimodal_models)
    for model_id in text_models:
        source_map[model_id] = preset
    for model_id in multimodal_models:
        source_map[model_id] = preset
    return text_models, multimodal_models, source_map


def _classify_error(exc: Exception) -> str:
    text = str(exc).lower()
    if "401" in text or "403" in text or "unauthorized" in text or "forbidden" in text:
        return "auth"
    if "repository not found" in text or "not found" in text:
        return "not_found"
    if "connection" in text or "timeout" in text or "network" in text:
        return "network"
    return "runtime"


def _require_snapshot_download() -> None:
    if snapshot_download is not None:
        return
    raise ModuleNotFoundError(
        "huggingface_hub is required for model download script. "
        "Install transformers/meap dependencies first. "
        f"Original import error: {_IMPORT_ERROR}"
    ) from _IMPORT_ERROR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-click Hugging Face model downloader. "
            "Default preset downloads all known models (official + route matrix + v1.5 candidates)."
        )
    )
    parser.add_argument(
        "--preset",
        default="all-known",
        choices=["official", "route-matrix", "v1-5-candidates", "all-known"],
    )
    parser.add_argument(
        "--text-models",
        default="",
        help="Extra comma-separated text model ids to append.",
    )
    parser.add_argument(
        "--multimodal-models",
        default="",
        help="Extra comma-separated multimodal model ids to append.",
    )
    parser.add_argument("--hf-token", default=None, help="Hugging Face token for private/gated models.")
    parser.add_argument("--cache-dir", default="", help="Optional Hugging Face cache directory.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="reports/hf_model_download.json")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _require_snapshot_download()

    text_models, multimodal_models, source_map = _preset_models(args.preset)
    extra_text = _parse_model_list(args.text_models)
    extra_multimodal = _parse_model_list(args.multimodal_models)

    for model_id in extra_text:
        source_map[model_id] = "custom"
    for model_id in extra_multimodal:
        source_map[model_id] = "custom"

    text_models = _ordered_unique([*text_models, *extra_text])
    multimodal_models = _ordered_unique([*multimodal_models, *extra_multimodal])

    items: List[Tuple[str, str]] = []
    items.extend((model_id, "text") for model_id in text_models)
    items.extend((model_id, "multimodal") for model_id in multimodal_models)
    items = _ordered_unique([f"{model_id}|||{modality}" for model_id, modality in items])
    pairs = [tuple(item.split("|||", 1)) for item in items]

    if not args.quiet:
        print(f"Preset: {args.preset}")
        print(f"Text models ({len(text_models)}): {text_models}")
        print(f"Multimodal models ({len(multimodal_models)}): {multimodal_models}")
        if args.dry_run:
            print("Dry-run mode enabled. No downloads will be performed.")

    rows: List[DownloadRow] = []
    cache_dir = args.cache_dir or None
    for model_id, modality in pairs:
        start = time.time()
        if args.dry_run:
            rows.append(
                DownloadRow(
                    model_id=model_id,
                    modality=modality,
                    source=source_map.get(model_id, args.preset),
                    status="dry_run",
                    seconds=0.0,
                    snapshot_path="",
                    error_type="",
                    error_message="",
                )
            )
            continue

        try:
            snapshot_path = snapshot_download(
                repo_id=model_id,
                token=args.hf_token,
                cache_dir=cache_dir,
                local_files_only=args.local_files_only,
            )
            row = DownloadRow(
                model_id=model_id,
                modality=modality,
                source=source_map.get(model_id, args.preset),
                status="pass",
                seconds=time.time() - start,
                snapshot_path=str(snapshot_path),
                error_type="",
                error_message="",
            )
        except Exception as exc:  # noqa: BLE001
            row = DownloadRow(
                model_id=model_id,
                modality=modality,
                source=source_map.get(model_id, args.preset),
                status="fail",
                seconds=time.time() - start,
                snapshot_path="",
                error_type=_classify_error(exc),
                error_message=str(exc),
            )
        rows.append(row)

        if not args.quiet:
            print(
                f"[{row.status}] {row.model_id} ({row.modality}) "
                f"{row.seconds:.2f}s {row.error_type}".rstrip()
            )

    all_passed = all(row.status in {"pass", "dry_run"} for row in rows)
    payload = {
        "report_type": REPORT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "preset": args.preset,
            "text_models": text_models,
            "multimodal_models": multimodal_models,
            "cache_dir": args.cache_dir,
            "local_files_only": bool(args.local_files_only),
            "dry_run": bool(args.dry_run),
        },
        "all_passed": all_passed,
        "n_models": len(rows),
        "n_pass": sum(1 for row in rows if row.status == "pass"),
        "n_fail": sum(1 for row in rows if row.status == "fail"),
        "results": [asdict(row) for row in rows],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.quiet:
        print(f"Saved report to {output_path}")
    print(json.dumps({"all_passed": all_passed, "output": str(output_path)}))

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
