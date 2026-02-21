#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from meap import list_official_models

REPORT_TYPE = "stage_matrix"
SCHEMA_VERSION = "1.0.0"
_OFFICIAL_MODELS = list_official_models()
_DEFAULT_TEXT_MODELS = ",".join([row["model_id"] for row in _OFFICIAL_MODELS if row["modality"] == "text"])
_DEFAULT_MULTIMODAL_MODELS = ",".join(
    [row["model_id"] for row in _OFFICIAL_MODELS if row["modality"] == "multimodal"]
)


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _parse_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _slug(model_id: str) -> str:
    return (
        model_id.replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace(":", "_")
    )


def _classify_error_message(message: str) -> str:
    lower = (message or "").lower()
    if "out of memory" in lower or "cuda oom" in lower:
        return "oom"
    if "forbidden" in lower or "unauthorized" in lower or "403" in lower or "401" in lower:
        return "auth"
    if "unsupported" in lower:
        return "unsupported"
    if "skip" in lower:
        return "skip"
    if "mismatch" in lower or "parity" in lower:
        return "parity_mismatch"
    if lower.strip() == "":
        return ""
    return "runtime"


def _classify_parity_row(row: Dict[str, Any]) -> str:
    status = str(row.get("status", ""))
    if status == "pass":
        return ""
    if status == "skip":
        return "skip"
    if status == "fail":
        return "parity_mismatch"
    if status == "error":
        return _classify_error_message(str(row.get("note", "")))
    return "runtime"


@dataclass
class CommandResult:
    name: str
    command: List[str]
    returncode: int
    seconds: float
    output_path: str


@dataclass
class ModelLevelRow:
    model_id: str
    track: str
    status: str
    seconds: float
    error_type: str
    error_message: str
    details: Dict[str, Any]


@dataclass
class MethodLevelRow:
    model_id: str
    track: str
    backend: str
    method: str
    status: str
    error_type: str
    note: str
    metrics: Optional[Dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified PR4 matrix: text parity + multimodal smoke.")
    parser.add_argument("--output", default="reports/stage_matrix.json")

    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--skip-ruff", action="store_true")
    parser.add_argument("--skip-text-parity", action="store_true")
    parser.add_argument("--skip-multimodal-smoke", action="store_true")

    parser.add_argument(
        "--text-models",
        default=_DEFAULT_TEXT_MODELS,
    )
    parser.add_argument(
        "--multimodal-models",
        default=_DEFAULT_MULTIMODAL_MODELS,
    )
    parser.add_argument(
        "--parity-methods",
        default="EAP,EAP-IG-inputs,clean-corrupted,EAP-IG-activations,exact",
    )
    parser.add_argument("--device", default="cuda" if _has_cuda() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--audio-fallback-model", default="fixie-ai/ultravox-v0_5-llama-3_2-1b")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--ig-steps", type=int, default=4)
    parser.add_argument("--max-exact-edges", type=int, default=30_000)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _run(name: str, command: List[str], *, output_path: str = "") -> CommandResult:
    start = time.time()
    completed = subprocess.run(command, check=False)
    return CommandResult(
        name=name,
        command=command,
        returncode=completed.returncode,
        seconds=time.time() - start,
        output_path=output_path,
    )


def _load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _summarize_text_model(model_id: str, rows: List[Dict[str, Any]]) -> ModelLevelRow:
    if len(rows) == 0:
        return ModelLevelRow(
            model_id=model_id,
            track="text_parity",
            status="fail",
            seconds=0.0,
            error_type="runtime",
            error_message="No method-level parity rows produced.",
            details={"n_methods": 0},
        )

    statuses = [str(row.get("status", "error")) for row in rows]
    total_seconds = float(sum(float(row.get("seconds", 0.0)) for row in rows))

    if all(status in {"pass", "skip"} for status in statuses):
        return ModelLevelRow(
            model_id=model_id,
            track="text_parity",
            status="pass",
            seconds=total_seconds,
            error_type="",
            error_message="",
            details={
                "n_methods": len(rows),
                "n_pass": sum(1 for status in statuses if status == "pass"),
                "n_skip": sum(1 for status in statuses if status == "skip"),
            },
        )

    failed = next((row for row in rows if row.get("status") in {"fail", "error"}), rows[0])
    note = str(failed.get("skip_reason") or failed.get("note") or "")
    error_type = _classify_parity_row(failed)
    if error_type == "":
        error_type = "runtime"
    return ModelLevelRow(
        model_id=model_id,
        track="text_parity",
        status="fail",
        seconds=total_seconds,
        error_type=error_type,
        error_message=note,
        details={
            "n_methods": len(rows),
            "n_fail": sum(1 for status in statuses if status in {"fail", "error"}),
            "n_skip": sum(1 for status in statuses if status == "skip"),
        },
    )


def _build_error_summary(model_rows: List[ModelLevelRow], method_rows: List[MethodLevelRow]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in model_rows:
        if row.error_type:
            counts[row.error_type] = counts.get(row.error_type, 0) + 1
    for row in method_rows:
        if row.error_type:
            counts[row.error_type] = counts.get(row.error_type, 0) + 1
    return counts


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command_rows: List[CommandResult] = []
    model_rows: List[ModelLevelRow] = []
    method_rows: List[MethodLevelRow] = []

    if not args.skip_ruff:
        command_rows.append(
            _run(
                "ruff",
                [sys.executable, "-m", "ruff", "check", "meap", "tests", "scripts"],
            )
        )

    if not args.skip_pytest:
        command_rows.append(_run("pytest", [sys.executable, "-m", "pytest", "-q"]))

    if not args.skip_text_parity:
        for text_model in _parse_list(args.text_models):
            parity_output = output_path.parent / f"text_parity_{_slug(text_model)}.json"
            command = [
                sys.executable,
                "scripts/test_text_vendor_parity.py",
                "--models",
                text_model,
                "--methods",
                args.parity_methods,
                "--n-samples",
                str(args.n_samples),
                "--batch-size",
                str(args.batch_size),
                "--ig-steps",
                str(args.ig_steps),
                "--max-exact-edges",
                str(args.max_exact_edges),
                "--device",
                args.device,
                "--dtype",
                args.dtype,
                "--output",
                str(parity_output),
            ]
            if args.strict:
                command.append("--strict")
            if args.quiet:
                command.append("--quiet")

            result = _run(f"text_parity:{text_model}", command, output_path=str(parity_output))
            command_rows.append(result)

            payload = _load_json_if_exists(parity_output)
            if payload is None:
                model_rows.append(
                    ModelLevelRow(
                        model_id=text_model,
                        track="text_parity",
                        status="fail",
                        seconds=result.seconds,
                        error_type="runtime",
                        error_message="Parity command failed and produced no report artifact.",
                        details={"returncode": result.returncode},
                    )
                )
                method_rows.append(
                    MethodLevelRow(
                        model_id=text_model,
                        track="text_parity",
                        backend="hf_vs_vendor",
                        method="all",
                        status="error",
                        error_type="runtime",
                        note="Missing parity JSON artifact.",
                        metrics=None,
                    )
                )
                continue

            rows = payload.get("results", [])
            for row in rows:
                method_rows.append(
                    MethodLevelRow(
                        model_id=text_model,
                        track="text_parity",
                        backend="hf_vs_vendor",
                        method=str(row.get("method", "")),
                        status=str(row.get("status", "error")),
                        error_type=_classify_parity_row(row),
                        note=str(row.get("skip_reason") or row.get("note") or ""),
                        metrics=row.get("vendor_vs_ours_hf"),
                    )
                )
            model_rows.append(_summarize_text_model(text_model, rows))

    if not args.skip_multimodal_smoke:
        smoke_output = output_path.parent / "multimodal_smoke_matrix.json"
        smoke_command = [
            sys.executable,
            "scripts/smoke_hf_matrix.py",
            "--text-models",
            "",
            "--multimodal-models",
            args.multimodal_models,
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--audio-fallback-model",
            args.audio_fallback_model,
            "--output",
            str(smoke_output),
        ]
        if args.hf_token is not None:
            smoke_command.extend(["--hf-token", args.hf_token])
        if args.quiet:
            smoke_command.append("--quiet")

        result = _run("multimodal_smoke", smoke_command, output_path=str(smoke_output))
        command_rows.append(result)

        payload = _load_json_if_exists(smoke_output)
        if payload is None:
            model_rows.append(
                ModelLevelRow(
                    model_id="multimodal_smoke",
                    track="multimodal_smoke",
                    status="fail",
                    seconds=result.seconds,
                    error_type="runtime",
                    error_message="Smoke command failed and produced no report artifact.",
                    details={"returncode": result.returncode},
                )
            )
            method_rows.append(
                MethodLevelRow(
                    model_id="multimodal_smoke",
                    track="multimodal_smoke",
                    backend="hf",
                    method="smoke",
                    status="error",
                    error_type="runtime",
                    note="Missing smoke JSON artifact.",
                    metrics=None,
                )
            )
        else:
            for row in payload.get("results", []):
                model_rows.append(
                    ModelLevelRow(
                        model_id=str(row.get("model_id", "")),
                        track="multimodal_smoke",
                        status=str(row.get("status", "fail")),
                        seconds=float(row.get("seconds", 0.0)),
                        error_type=str(row.get("error_type", "")),
                        error_message=str(row.get("error_message", "")),
                        details={
                            "adapter_name": row.get("adapter_name", ""),
                            "language_trunk_path": row.get("language_trunk_path", ""),
                            "arch_kind": row.get("arch_kind", ""),
                            "resolution_error_hint": row.get("resolution_error_hint", ""),
                            "graph_stats": row.get("graph_stats"),
                        },
                    )
                )
                method_rows.append(
                    MethodLevelRow(
                        model_id=str(row.get("model_id", "")),
                        track="multimodal_smoke",
                        backend="hf",
                        method="smoke",
                        status=str(row.get("status", "fail")),
                        error_type=str(row.get("error_type", "")),
                        note=str(row.get("error_message", "")),
                        metrics=row.get("graph_stats"),
                    )
                )

    errors = _build_error_summary(model_rows, method_rows)
    checks_passed = all(result.returncode == 0 for result in command_rows)
    models_passed = all(row.status in {"pass", "skip"} for row in model_rows)

    report = {
        "report_type": REPORT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "text_models": _parse_list(args.text_models),
            "multimodal_models": _parse_list(args.multimodal_models),
            "parity_methods": _parse_list(args.parity_methods),
            "device": args.device,
            "dtype": args.dtype,
            "strict": bool(args.strict),
            "n_samples": args.n_samples,
            "batch_size": args.batch_size,
            "ig_steps": args.ig_steps,
            "max_exact_edges": args.max_exact_edges,
            "audio_fallback_model": args.audio_fallback_model,
        },
        "checks": [asdict(result) for result in command_rows],
        "model_level": [asdict(row) for row in model_rows],
        "method_level": [asdict(row) for row in method_rows],
        "error_summary": errors,
        "all_passed": bool(checks_passed and models_passed),
    }

    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved stage matrix report to {output_path}")

    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
