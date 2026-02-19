#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VENDOR_SRC = REPO_ROOT / "vendor" / "eap-ig" / "src"
if str(VENDOR_SRC) not in sys.path:
    sys.path.insert(0, str(VENDOR_SRC))

import eap.attribute as vendor_attribute_module  # noqa: E402
from eap.attribute import attribute as vendor_attribute  # noqa: E402
import eap.evaluate as vendor_evaluate_module  # noqa: E402
from eap.graph import Graph as VendorGraph  # noqa: E402
import eap.utils as vendor_utils_module  # noqa: E402

from multimodal_lm_eap_ig.attribute import attribute as ours_attribute  # noqa: E402
from multimodal_lm_eap_ig.backend import HFLLMBackend, TLensBackend  # noqa: E402
from multimodal_lm_eap_ig.batch import PreparedBatch, text_batch_to_prepared_batch  # noqa: E402
from multimodal_lm_eap_ig.graph import Graph as OursGraph  # noqa: E402

DEFAULT_METHODS = [
    "EAP",
    "EAP-IG-inputs",
    "clean-corrupted",
    "EAP-IG-activations",
    "exact",
]


@dataclass
class DiffStats:
    max_abs: float
    mean_abs: float
    cosine: float


@dataclass
class MethodParityRow:
    model_tlens: str
    model_hf: str
    method: str
    status: str
    seconds: float
    note: str
    vendor_vs_ours_tlens: DiffStats | None
    vendor_vs_ours_hf: DiffStats | None
    ours_tlens_vs_ours_hf: DiffStats | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parity test between vendor EAP-IG and multimodal_lm_eap_ig on text models. "
            "Runs vendor, ours+TLens backend, and ours+HF backend on the same batches."
        )
    )
    parser.add_argument(
        "--models",
        default="gpt2-small,Qwen/Qwen2-0.5B",
        help="Comma-separated TLens model ids.",
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated method list.",
    )
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--ig-steps", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument(
        "--max-exact-edges",
        type=int,
        default=30_000,
        help="Skip exact if graph edge count exceeds this threshold.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-abs-threshold", type=float, default=5e-3)
    parser.add_argument("--cosine-threshold", type=float, default=0.995)
    parser.add_argument("--output", default="")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def resolve_hf_model_id(model_id: str) -> str:
    if model_id == "gpt2-small":
        return "gpt2"
    return model_id


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def patch_vendor_cuda_literals() -> None:
    if torch.cuda.is_available():
        return

    def _patch_zeros(module: Any) -> None:
        if getattr(module, "_mm_eap_ig_cpu_patch_applied", False):
            return
        original_zeros = module.torch.zeros

        def _zeros(*args, **kwargs):
            if kwargs.get("device") == "cuda":
                kwargs["device"] = "cpu"
            return original_zeros(*args, **kwargs)

        module.torch.zeros = _zeros
        module._mm_eap_ig_cpu_patch_applied = True

    _patch_zeros(vendor_attribute_module)
    _patch_zeros(vendor_evaluate_module)
    _patch_zeros(vendor_utils_module)


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
    names: list[str] = []
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


def build_vendor_batches(
    records: Sequence[tuple[str, str, list[int]]],
    batch_size: int,
) -> list[tuple[list[str], list[str], Tensor]]:
    batches: list[tuple[list[str], list[str], Tensor]] = []
    for i in range(0, len(records), batch_size):
        chunk = records[i : i + batch_size]
        clean = [x[0] for x in chunk]
        corrupt = [x[1] for x in chunk]
        labels = torch.tensor([x[2] for x in chunk], dtype=torch.long)
        batches.append((clean, corrupt, labels))
    return batches


def build_prepared_batches(
    tlens_tokenization_model: HookedTransformer,
    records: Sequence[tuple[str, str, list[int]]],
    batch_size: int,
) -> list[PreparedBatch]:
    out: list[PreparedBatch] = []
    for i in range(0, len(records), batch_size):
        chunk = records[i : i + batch_size]
        clean = [x[0] for x in chunk]
        corrupt = [x[1] for x in chunk]
        labels = torch.tensor([x[2] for x in chunk], dtype=torch.long)
        out.append(
            text_batch_to_prepared_batch(
                tlens_tokenization_model,
                clean,
                corrupt,
                labels,
            )
        )
    return out


def vendor_metric(logits: Tensor, clean_logits: Tensor | None, input_lengths: Tensor, labels: Tensor) -> Tensor:
    del clean_logits
    input_lengths = input_lengths.to(device=logits.device)
    labels = labels.to(device=logits.device)
    batch_idx = torch.arange(logits.size(0), device=logits.device)
    last_logits = logits[batch_idx, input_lengths - 1]
    pair = torch.gather(last_logits, -1, labels)
    return -(pair[:, 0] - pair[:, 1]).mean()


def ours_metric(logits: Tensor, clean_logits: Tensor | None, batch: PreparedBatch) -> Tensor:
    del clean_logits
    labels = batch.labels
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels, dtype=torch.long, device=logits.device)
    labels = labels.to(device=logits.device)
    batch_idx = torch.arange(logits.size(0), device=logits.device)
    last_logits = logits[batch_idx, batch.input_lengths.to(logits.device) - 1]
    pair = torch.gather(last_logits, -1, labels)
    return -(pair[:, 0] - pair[:, 1]).mean()


def _as_float_cpu(x: Tensor) -> Tensor:
    return x.detach().to(dtype=torch.float32, device="cpu")


def compute_diff_stats(a: Tensor, b: Tensor) -> DiffStats:
    a_cpu = _as_float_cpu(a).reshape(-1)
    b_cpu = _as_float_cpu(b).reshape(-1)
    diff = (a_cpu - b_cpu).abs()
    cosine = 1.0
    if torch.linalg.vector_norm(a_cpu).item() > 0 and torch.linalg.vector_norm(b_cpu).item() > 0:
        cosine = float(F.cosine_similarity(a_cpu, b_cpu, dim=0).item())
    return DiffStats(
        max_abs=float(diff.max().item()),
        mean_abs=float(diff.mean().item()),
        cosine=cosine,
    )


def run_method(
    *,
    method: str,
    tlens_model: HookedTransformer,
    hf_model: torch.nn.Module,
    tlens_backend: TLensBackend,
    hf_backend: HFLLMBackend,
    vendor_batches: Iterable[tuple[list[str], list[str], Tensor]],
    prepared_batches: Iterable[PreparedBatch],
    ig_steps: int,
    max_exact_edges: int,
    max_abs_threshold: float,
    cosine_threshold: float,
    quiet: bool,
    model_id_tlens: str,
    model_id_hf: str,
) -> MethodParityRow:
    start = time.time()

    vendor_graph = VendorGraph.from_model(tlens_model)
    if method == "exact" and len(vendor_graph.edges) > max_exact_edges:
        return MethodParityRow(
            model_tlens=model_id_tlens,
            model_hf=model_id_hf,
            method=method,
            status="skip",
            seconds=0.0,
            note=(
                f"exact skipped: edge_count={len(vendor_graph.edges)} exceeds "
                f"--max-exact-edges={max_exact_edges}"
            ),
            vendor_vs_ours_tlens=None,
            vendor_vs_ours_hf=None,
            ours_tlens_vs_ours_hf=None,
        )

    kwargs: Dict[str, Any] = {"method": method, "quiet": quiet}
    if method in {"EAP-IG-inputs", "EAP-IG-activations"}:
        kwargs["ig_steps"] = ig_steps

    vendor_result = vendor_attribute(
        model=tlens_model,
        graph=vendor_graph,
        dataloader=vendor_batches,
        metric=vendor_metric,
        **kwargs,
    )
    vendor_scores = vendor_graph.scores if vendor_result is None else vendor_result

    ours_tlens_graph = OursGraph.from_model(tlens_backend.config)
    ours_tlens_scores = ours_attribute(
        model=tlens_model,
        backend=tlens_backend,
        graph=ours_tlens_graph,
        batches=prepared_batches,
        metric=ours_metric,
        **kwargs,
    )

    ours_hf_graph = OursGraph.from_model(hf_backend.config)
    ours_hf_scores = ours_attribute(
        model=hf_model,
        backend=hf_backend,
        graph=ours_hf_graph,
        batches=prepared_batches,
        metric=ours_metric,
        **kwargs,
    )

    d_vendor_tlens = compute_diff_stats(vendor_scores, ours_tlens_scores)
    d_vendor_hf = compute_diff_stats(vendor_scores, ours_hf_scores)
    d_tlens_hf = compute_diff_stats(ours_tlens_scores, ours_hf_scores)

    pass_tlens = (
        d_vendor_tlens.max_abs <= max_abs_threshold or d_vendor_tlens.cosine >= cosine_threshold
    )
    pass_hf = d_vendor_hf.max_abs <= max_abs_threshold or d_vendor_hf.cosine >= cosine_threshold
    status = "pass" if (pass_tlens and pass_hf) else "fail"

    return MethodParityRow(
        model_tlens=model_id_tlens,
        model_hf=model_id_hf,
        method=method,
        status=status,
        seconds=round(time.time() - start, 3),
        note="",
        vendor_vs_ours_tlens=d_vendor_tlens,
        vendor_vs_ours_hf=d_vendor_hf,
        ours_tlens_vs_ours_hf=d_tlens_hf,
    )


def main() -> None:
    args = parse_args()
    dtype = dtype_from_name(args.dtype)
    if args.device == "cpu" and dtype == torch.float16:
        raise ValueError("float16 on CPU is unstable; use float32/bfloat16")

    set_seed(args.seed)
    patch_vendor_cuda_literals()

    model_ids = [x.strip() for x in args.models.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]

    all_rows: list[MethodParityRow] = []
    for tlens_id in model_ids:
        hf_id = resolve_hf_model_id(tlens_id)

        print(f"\\n=== Loading model pair: tlens={tlens_id} hf={hf_id} ===")
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        tlens_model = load_tlens_model(
            tlens_id,
            tokenizer=tokenizer,
            device=args.device,
            dtype=dtype,
        )
        tlens_model.eval()

        hf_model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=dtype).to(args.device)
        hf_model.eval()

        tlens_backend = TLensBackend(tlens_model)
        hf_backend = HFLLMBackend(hf_model, tokenizer=tokenizer)

        records = build_records(tokenizer, args.n_samples)
        vendor_batches = build_vendor_batches(records, args.batch_size)
        prepared_batches = build_prepared_batches(tlens_model, records, args.batch_size)

        for method in methods:
            print(f"Running method={method} ...")
            try:
                row = run_method(
                    method=method,
                    tlens_model=tlens_model,
                    hf_model=hf_model,
                    tlens_backend=tlens_backend,
                    hf_backend=hf_backend,
                    vendor_batches=vendor_batches,
                    prepared_batches=prepared_batches,
                    ig_steps=args.ig_steps,
                    max_exact_edges=args.max_exact_edges,
                    max_abs_threshold=args.max_abs_threshold,
                    cosine_threshold=args.cosine_threshold,
                    quiet=args.quiet,
                    model_id_tlens=tlens_id,
                    model_id_hf=hf_id,
                )
            except Exception as exc:
                row = MethodParityRow(
                    model_tlens=tlens_id,
                    model_hf=hf_id,
                    method=method,
                    status="error",
                    seconds=0.0,
                    note=str(exc)[:320],
                    vendor_vs_ours_tlens=None,
                    vendor_vs_ours_hf=None,
                    ours_tlens_vs_ours_hf=None,
                )
            all_rows.append(row)

            if row.status in {"pass", "fail"} and row.vendor_vs_ours_hf is not None:
                print(
                    "  status={status} vendor_vs_hf(max_abs={max_abs:.4e}, cosine={cosine:.6f})".format(
                        status=row.status,
                        max_abs=row.vendor_vs_ours_hf.max_abs,
                        cosine=row.vendor_vs_ours_hf.cosine,
                    )
                )
            elif row.status == "skip":
                print(f"  status=skip note={row.note}")
            else:
                print(f"  status={row.status} note={row.note}")

        if torch.cuda.is_available() and args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    payload = {
        "config": {
            "models": model_ids,
            "methods": methods,
            "device": args.device,
            "dtype": args.dtype,
            "n_samples": args.n_samples,
            "batch_size": args.batch_size,
            "ig_steps": args.ig_steps,
            "max_exact_edges": args.max_exact_edges,
            "max_abs_threshold": args.max_abs_threshold,
            "cosine_threshold": args.cosine_threshold,
        },
        "all_passed": all(row.status in {"pass", "skip"} for row in all_rows),
        "results": [asdict(row) for row in all_rows],
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print("\n=== Summary JSON ===")
    print(text)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
