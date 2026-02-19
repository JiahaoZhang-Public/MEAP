#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Dict

import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multimodal_lm_eap_ig.attribute import attribute  # noqa: E402
from multimodal_lm_eap_ig.backend import HFLLMBackend, TLensBackend  # noqa: E402
from multimodal_lm_eap_ig.batch import PreparedBatch, text_batch_to_prepared_batch  # noqa: E402
from multimodal_lm_eap_ig.graph import Graph  # noqa: E402
from multimodal_lm_eap_ig.utils import forward_with_hooks, resolve_run_inputs  # noqa: E402

DEFAULT_METHODS = [
    "smoke",
    "EAP",
    "EAP-IG-inputs",
    "clean-corrupted",
    "EAP-IG-activations",
    "exact",
]


@dataclass
class ModelLevelResult:
    model_tlens: str
    model_hf: str
    graph_match: bool
    embedding_match: bool
    embedding_shape: tuple[int, ...]
    embedding_max_abs_diff: float


@dataclass
class MethodRow:
    model_tlens: str
    model_hf: str
    backend: str
    method: str
    status: str
    seconds: float
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified backend test script: model-level + method-level checks.",
    )
    parser.add_argument(
        "--models",
        default="gpt2-small,Qwen/Qwen2-0.5B",
        help="Comma-separated TLens model ids.",
    )
    parser.add_argument(
        "--levels",
        default="model,method",
        help="Comma-separated levels to run: model,method",
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated methods for method-level test.",
    )
    parser.add_argument("--prompt", default="The quick brown fox")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--ig-steps", type=int, default=1)
    parser.add_argument("--skip-exact", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def to_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def resolve_hf_model_id(model_id: str) -> str:
    if model_id == "gpt2-small":
        return "gpt2"
    return model_id


def load_tlens_model(
    model_id: str,
    *,
    tokenizer,
    device: str,
    dtype: torch.dtype,
) -> HookedTransformer:
    try:
        model = HookedTransformer.from_pretrained(
            model_id,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            fold_value_biases=False,
            default_padding_side="right",
        )
    except Exception:
        model = HookedTransformer.from_pretrained_no_processing(
            model_id,
            device=device,
            dtype=dtype,
            default_padding_side="right",
        )

    model.eval()
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    if model.cfg.n_key_value_heads is not None:
        model.cfg.ungroup_grouped_query_attention = True
    return model


def build_records(tokenizer, n_samples: int) -> list[tuple[str, str, list[int]]]:
    candidates = [
        "John",
        "Mary",
        "James",
        "Robert",
        "Michael",
        "David",
        "Thomas",
        "William",
    ]
    names = []
    for name in candidates:
        token_ids = tokenizer.encode(f" {name}", add_special_tokens=False)
        if len(token_ids) == 1:
            names.append(name)

    if len(names) < 4:
        raise RuntimeError(f"Need >=4 single-token names, found {len(names)}: {names}")

    records: list[tuple[str, str, list[int]]] = []
    for i in range(n_samples):
        subject = names[i % len(names)]
        indirect_obj = names[(i + 1) % len(names)]

        clean = (
            f"When {subject} and {indirect_obj} were in the office, "
            f"{subject} handed a letter to"
        )
        corrupt = (
            f"When {subject} and {indirect_obj} were in the office, "
            f"{indirect_obj} handed a letter to"
        )

        correct = tokenizer.encode(f" {indirect_obj}", add_special_tokens=False)[0]
        incorrect = tokenizer.encode(f" {subject}", add_special_tokens=False)[0]
        records.append((clean, corrupt, [correct, incorrect]))
    return records


def build_prepared_batches(
    tlens_tokenization_model: HookedTransformer,
    records: list[tuple[str, str, list[int]]],
    batch_size: int,
) -> list[PreparedBatch]:
    batches: list[PreparedBatch] = []
    for i in range(0, len(records), batch_size):
        chunk = records[i : i + batch_size]
        clean = [x[0] for x in chunk]
        corrupt = [x[1] for x in chunk]
        labels = torch.tensor([x[2] for x in chunk], dtype=torch.long)
        prepared = text_batch_to_prepared_batch(
            tlens_tokenization_model,
            clean,
            corrupt,
            labels,
        )
        batches.append(prepared)
    return batches


def metric_fn(logits, clean_logits, batch: PreparedBatch):
    del clean_logits
    labels = batch.labels
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels, dtype=torch.long, device=logits.device)
    labels = labels.to(logits.device)

    idx = torch.arange(logits.size(0), device=logits.device)
    last_logits = logits[idx, batch.input_lengths.to(logits.device) - 1]
    pair = torch.gather(last_logits, -1, labels)
    return -(pair[:, 0] - pair[:, 1]).mean()


def run_model_level(
    model_id_tlens: str,
    model_id_hf: str,
    tlens_backend: TLensBackend,
    hf_backend: HFLLMBackend,
    prompt: str,
    atol: float,
    rtol: float,
) -> ModelLevelResult:
    tokenizer = hf_backend.tokenizer
    encoded = tokenizer(prompt, return_tensors="pt")

    model_inputs = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }

    tlens_graph = Graph.from_model(tlens_backend.config)
    hf_graph = Graph.from_model(hf_backend.config)
    graph_match = (
        tlens_backend.config.n_layers == hf_backend.config.n_layers
        and tlens_backend.config.n_heads == hf_backend.config.n_heads
        and tlens_backend.config.d_model == hf_backend.config.d_model
        and tlens_backend.config.parallel_attn_mlp == hf_backend.config.parallel_attn_mlp
        and tlens_graph.n_forward == hf_graph.n_forward
        and tlens_graph.n_backward == hf_graph.n_backward
        and set(tlens_graph.nodes.keys()) == set(hf_graph.nodes.keys())
        and set(tlens_graph.edges.keys()) == set(hf_graph.edges.keys())
    )

    captures: Dict[str, torch.Tensor] = {}

    def capture_hf_embed(activations, hook):
        del hook
        captures["hf"] = activations.detach().cpu()
        return activations

    def capture_tlens_embed(activations, hook):
        del hook
        captures["tlens"] = activations.detach().cpu()
        return activations

    hf_inputs = resolve_run_inputs(hf_backend, model_inputs)
    tlens_inputs = resolve_run_inputs(tlens_backend, model_inputs)

    with torch.inference_mode():
        _ = forward_with_hooks(hf_backend, hf_inputs, fwd_hooks=[("hook_embed", capture_hf_embed)])
        _ = forward_with_hooks(
            tlens_backend,
            tlens_inputs,
            fwd_hooks=[("hook_embed", capture_tlens_embed)],
        )

    if "hf" not in captures or "tlens" not in captures:
        raise RuntimeError("Failed to capture hook_embed activations from one or both backends")

    diff = (captures["hf"] - captures["tlens"]).abs()
    max_abs_diff = float(diff.max().item())
    embed_match = torch.allclose(captures["hf"], captures["tlens"], atol=atol, rtol=rtol)

    return ModelLevelResult(
        model_tlens=model_id_tlens,
        model_hf=model_id_hf,
        graph_match=graph_match,
        embedding_match=embed_match,
        embedding_shape=tuple(captures["hf"].shape),
        embedding_max_abs_diff=max_abs_diff,
    )


def run_method_level(
    model_id_tlens: str,
    model_id_hf: str,
    methods: list[str],
    batches: list[PreparedBatch],
    tlens_backend: TLensBackend,
    hf_backend: HFLLMBackend,
    tlens_model: HookedTransformer,
    hf_model: torch.nn.Module,
    ig_steps: int,
    skip_exact: bool,
    quiet: bool,
) -> list[MethodRow]:
    out: list[MethodRow] = []

    backend_specs = [
        ("tlens", tlens_backend, tlens_model),
        ("hf", hf_backend, hf_model),
    ]

    for backend_name, backend, model in backend_specs:
        for method in methods:
            graph = Graph.from_model(backend.config)
            if method == "exact" and skip_exact:
                out.append(
                    MethodRow(
                        model_tlens=model_id_tlens,
                        model_hf=model_id_hf,
                        backend=backend_name,
                        method=method,
                        status="skip",
                        seconds=0.0,
                        message=f"skipped (edges={len(graph.edges)})",
                    )
                )
                continue

            start = time.time()
            try:
                kwargs = {}
                if method in {"EAP-IG-inputs", "EAP-IG-activations"}:
                    kwargs["ig_steps"] = ig_steps

                _ = attribute(
                    model=model,
                    backend=backend,
                    graph=graph,
                    batches=batches,
                    metric=metric_fn,
                    method=method,
                    quiet=quiet,
                    **kwargs,
                )
                out.append(
                    MethodRow(
                        model_tlens=model_id_tlens,
                        model_hf=model_id_hf,
                        backend=backend_name,
                        method=method,
                        status="pass",
                        seconds=round(time.time() - start, 3),
                        message="",
                    )
                )
            except Exception as exc:
                out.append(
                    MethodRow(
                        model_tlens=model_id_tlens,
                        model_hf=model_id_hf,
                        backend=backend_name,
                        method=method,
                        status="fail",
                        seconds=round(time.time() - start, 3),
                        message=str(exc)[:280],
                    )
                )

    return out


def print_report(
    model_results: list[ModelLevelResult],
    method_rows: list[MethodRow],
) -> None:
    if model_results:
        print("=== Model Level ===")
        for r in model_results:
            print(f"model_tlens={r.model_tlens} model_hf={r.model_hf}")
            print(f"  graph_match={r.graph_match}")
            print(f"  embedding_match={r.embedding_match}")
            print(f"  embedding_shape={r.embedding_shape}")
            print(f"  embedding_max_abs_diff={r.embedding_max_abs_diff:.8f}")

    if method_rows:
        print("=== Method Level ===")
        for r in method_rows:
            print(
                f"model={r.model_tlens} backend={r.backend} method={r.method} "
                f"status={r.status} time={r.seconds:.3f}s"
            )
            if r.message:
                print(f"  message={r.message}")


def main() -> None:
    args = parse_args()
    dtype = to_dtype(args.dtype)

    if args.device == "cpu" and dtype == torch.float16:
        raise ValueError("float16 on CPU is unstable; use float32 or bfloat16")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    levels = {x.strip() for x in args.levels.split(",") if x.strip()}
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    model_results: list[ModelLevelResult] = []
    method_rows: list[MethodRow] = []

    for model_id in models:
        hf_model_id = resolve_hf_model_id(model_id)

        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        tlens_model = load_tlens_model(
            model_id,
            tokenizer=tokenizer,
            device=args.device,
            dtype=dtype,
        )
        hf_model = AutoModelForCausalLM.from_pretrained(hf_model_id, dtype=dtype).to(args.device)
        hf_model.eval()

        tlens_backend = TLensBackend(tlens_model)
        hf_backend = HFLLMBackend(hf_model, tokenizer=tokenizer)

        if "model" in levels:
            model_results.append(
                run_model_level(
                    model_id_tlens=model_id,
                    model_id_hf=hf_model_id,
                    tlens_backend=tlens_backend,
                    hf_backend=hf_backend,
                    prompt=args.prompt,
                    atol=args.atol,
                    rtol=args.rtol,
                )
            )

        if "method" in levels:
            records = build_records(tokenizer, args.n_samples)
            batches = build_prepared_batches(tlens_model, records, args.batch_size)
            method_rows.extend(
                run_method_level(
                    model_id_tlens=model_id,
                    model_id_hf=hf_model_id,
                    methods=methods,
                    batches=batches,
                    tlens_backend=tlens_backend,
                    hf_backend=hf_backend,
                    tlens_model=tlens_model,
                    hf_model=hf_model,
                    ig_steps=args.ig_steps,
                    skip_exact=args.skip_exact,
                    quiet=args.quiet,
                )
            )

    if args.json:
        data = {
            "model_level": [asdict(x) for x in model_results],
            "method_level": [asdict(x) for x in method_rows],
        }
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    print_report(model_results, method_rows)


if __name__ == "__main__":
    main()
