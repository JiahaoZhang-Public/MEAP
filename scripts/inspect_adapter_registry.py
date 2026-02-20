#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional

from transformers import AutoModel, AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multimodal_lm_eap_ig.backend import HFLLMBackend, inspect_model_architecture  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect architecture adapter resolution for a HF model.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", default="")
    parser.add_argument("--causal-lm", action="store_true", help="Load via AutoModelForCausalLM instead of AutoModel")
    return parser.parse_args()


def _load_model(model_id: str, *, trust_remote_code: bool, causal_lm: bool):
    kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if causal_lm:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except Exception:
        return AutoModel.from_pretrained(model_id, **kwargs)


def main() -> None:
    args = parse_args()
    model = _load_model(args.model_id, trust_remote_code=args.trust_remote_code, causal_lm=args.causal_lm)

    report = inspect_model_architecture(model)
    try:
        backend = HFLLMBackend(model, strict_arch=True)
        report["backend_init"] = {
            "status": "ok",
            "adapter_name": backend.adapter_name,
            "n_layers": backend.config.n_layers,
            "n_heads": backend.config.n_heads,
            "d_model": backend.config.d_model,
        }
    except Exception as exc:
        report["backend_init"] = {"status": "error", "message": str(exc)}

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        print(f"Saved registry report to {output}")


if __name__ == "__main__":
    main()
