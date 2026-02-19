#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import List


@dataclass
class CommandResult:
    name: str
    command: List[str]
    returncode: int
    seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged validation matrix for backend migration.")
    parser.add_argument("--output", default="reports/stage_matrix.json")

    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--skip-ruff", action="store_true")
    parser.add_argument("--skip-smoke-hf-matrix", action="store_true")

    parser.add_argument(
        "--text-models",
        default="gpt2,distilgpt2,facebook/opt-125m,Qwen/Qwen2-0.5B,TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )
    parser.add_argument(
        "--multimodal-models",
        default="Qwen/Qwen2-VL-2B,llava-hf/llava-1.5-7b-hf,HuggingFaceTB/SmolVLM-Instruct",
    )
    parser.add_argument("--device", default="cuda" if _has_cuda() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])

    parser.add_argument("--hf-token", default=None)

    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _run(name: str, command: List[str]) -> CommandResult:
    start = time.time()
    completed = subprocess.run(command, check=False)
    return CommandResult(
        name=name,
        command=command,
        returncode=completed.returncode,
        seconds=time.time() - start,
    )


def main() -> None:
    args = parse_args()

    results: List[CommandResult] = []

    if not args.skip_ruff:
        results.append(
            _run(
                "ruff",
                [sys.executable, "-m", "ruff", "check", "multimodal_lm_eap_ig", "tests", "scripts"],
            )
        )

    if not args.skip_pytest:
        results.append(_run("pytest", [sys.executable, "-m", "pytest", "-q"]))

    if not args.skip_smoke_hf_matrix:
        smoke_command = [
            sys.executable,
            "scripts/smoke_hf_matrix.py",
            "--text-models",
            args.text_models,
            "--multimodal-models",
            args.multimodal_models,
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--output",
            "reports/smoke_hf_matrix_stage.json",
        ]
        if args.hf_token is not None:
            smoke_command.extend(["--hf-token", args.hf_token])
        if args.quiet:
            smoke_command.append("--quiet")
        results.append(
            _run(
                "smoke_hf_matrix",
                smoke_command,
            )
        )

    report = {
        "all_passed": all(r.returncode == 0 for r in results),
        "results": [asdict(r) for r in results],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Saved stage matrix report to {output_path}")

    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
