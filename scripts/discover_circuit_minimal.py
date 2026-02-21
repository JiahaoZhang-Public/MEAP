"""Minimal one-shot circuit discovery examples for HF model id or local path."""

from __future__ import annotations

import argparse

from meap.api import discover_circuit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-shot discover_circuit")
    parser.add_argument("--model-ref", required=True, help="HF model id or local model directory")
    parser.add_argument("--task", default="next_token", choices=["next_token", "choice_classification"])
    parser.add_argument(
        "--method",
        default="EAP",
        choices=["EAP", "EAP-IG-inputs", "clean-corrupted", "EAP-IG-activations", "exact"],
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    clean = {"text": ["The capital of France is"]}
    corrupt = {"text": ["The capital of Germany is"]}

    try:
        if args.task == "next_token":
            result = discover_circuit(
                model_id_or_path=args.model_ref,
                clean_samples=clean,
                corrupt_samples=corrupt,
                task="next_token",
                labels=[357],
                method=args.method,
                top_k=args.top_k,
                device=args.device,
                dtype=args.dtype,
            )
        else:
            result = discover_circuit(
                model_id_or_path=args.model_ref,
                clean_samples=clean,
                corrupt_samples=corrupt,
                task="choice_classification",
                labels={"targets": [0], "choice_token_ids": [357, 11]},
                method=args.method,
                top_k=args.top_k,
                device=args.device,
                dtype=args.dtype,
            )
    except Exception as exc:
        print("discover_circuit failed with diagnostics:")
        print(exc)
        raise

    print("Backend:", result.backend_info)
    print("Run info:", result.run_info)
    print("Top edges:")
    for edge in result.top_edges:
        print(f"  {edge.edge_name}: score={edge.score:.6f}")


if __name__ == "__main__":
    main()
