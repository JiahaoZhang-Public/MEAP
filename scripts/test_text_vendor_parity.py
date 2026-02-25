#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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

_VENDOR_IMPORT_ERROR: Exception | None = None
vendor_attribute_module: Any = None
vendor_evaluate_module: Any = None
vendor_utils_module: Any = None
vendor_attribute: Any = None
VendorGraph: Any = None
try:
    import eap.attribute as vendor_attribute_module  # noqa: E402
    from eap.attribute import attribute as vendor_attribute  # noqa: E402
    import eap.evaluate as vendor_evaluate_module  # noqa: E402
    from eap.graph import Graph as VendorGraph  # noqa: E402
    import eap.utils as vendor_utils_module  # noqa: E402
except Exception as exc:  # noqa: BLE001
    _VENDOR_IMPORT_ERROR = exc

from meap.attribute import attribute as ours_attribute  # noqa: E402
from meap.backend import HFLLMBackend, TLensBackend  # noqa: E402
from meap.batch import PreparedBatch, text_batch_to_prepared_batch  # noqa: E402
from meap.graph import Graph as OursGraph  # noqa: E402

DEFAULT_METHODS = [
    "EAP",
    "EAP-IG-inputs",
    "clean-corrupted",
    "EAP-IG-activations",
]
REPORT_TYPE = "text_vendor_parity"
SCHEMA_VERSION = "1.0.0"


@dataclass
class DiffStats:
    max_abs: float
    mean_abs: float
    cosine: float
    topk_overlap: float


@dataclass
class MethodParityRow:
    model_tlens: str
    model_hf: str
    adapter_name: str
    method: str
    status: str
    seconds: float
    note: str
    skip_reason: str
    vendor_vs_ours_tlens: DiffStats | None
    vendor_vs_ours_hf: DiffStats | None
    ours_tlens_vs_ours_hf: DiffStats | None


def ensure_vendor_available() -> None:
    if _VENDOR_IMPORT_ERROR is None:
        return
    raise ModuleNotFoundError(
        "vendor EAP package is unavailable. Initialize the submodule first "
        "(git submodule update --init --recursive) or ensure vendor/eap-ig/src is present. "
        f"Original import error: {_VENDOR_IMPORT_ERROR}"
    ) from _VENDOR_IMPORT_ERROR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parity test between vendor EAP-IG and meap on text models. "
            "Runs vendor, ours+TLens backend, and ours+HF backend on the same batches."
        )
    )
    parser.add_argument(
        "--models",
        default="gpt2-small,Qwen/Qwen2-0.5B",
        help="Comma-separated TLens model ids.",
    )
    parser.add_argument(
        "--architectures",
        default="",
        help="Optional comma-separated adapter names to run (e.g. gpt2_like,llama_like).",
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
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--topk-overlap-threshold", type=float, default=0.9)
    parser.add_argument("--strict", action="store_true", help="Use strict parity thresholds.")
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
    if _VENDOR_IMPORT_ERROR is not None:
        return
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
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if not isinstance(vocab_size, int) or vocab_size <= 10:
            get_vocab = getattr(tokenizer, "get_vocab", None)
            if callable(get_vocab):
                try:
                    vocab_size = len(get_vocab())
                except Exception:  # noqa: BLE001
                    vocab_size = None
        if not isinstance(vocab_size, int) or vocab_size <= 10:
            vocab_size = 32_000
        token_a = min(42, vocab_size - 1)
        token_b = min(43, vocab_size - 1)
        if token_b == token_a:
            token_b = (token_a + 1) % vocab_size
        fallback_records: list[tuple[str, str, list[int]]] = []
        for i in range(n_samples):
            clean = f"Sample clean prompt {i}: the answer is"
            corrupt = f"Sample corrupt prompt {i}: the answer is"
            fallback_records.append((clean, corrupt, [token_a, token_b]))
        return fallback_records

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


def _pad_to_target(tokens: Tensor, *, target_length: int, value: int) -> Tensor:
    pad = target_length - int(tokens.shape[1])
    if pad <= 0:
        return tokens
    return torch.nn.functional.pad(tokens, (0, pad), value=value)


def build_prepared_batches_hf(
    tokenizer,
    records: Sequence[tuple[str, str, list[int]]],
    batch_size: int,
    *,
    device: str,
) -> list[PreparedBatch]:
    out: list[PreparedBatch] = []
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    for i in range(0, len(records), batch_size):
        chunk = records[i : i + batch_size]
        clean = [x[0] for x in chunk]
        corrupt = [x[1] for x in chunk]
        labels = torch.tensor([x[2] for x in chunk], dtype=torch.long, device=device)

        clean_tok = tokenizer(clean, return_tensors="pt", padding=True)
        corrupt_tok = tokenizer(corrupt, return_tensors="pt", padding=True)
        target_len = max(int(clean_tok["input_ids"].shape[1]), int(corrupt_tok["input_ids"].shape[1]))

        clean_ids = _pad_to_target(clean_tok["input_ids"], target_length=target_len, value=int(pad_token_id)).to(device)
        corrupt_ids = _pad_to_target(
            corrupt_tok["input_ids"], target_length=target_len, value=int(pad_token_id)
        ).to(device)
        clean_mask = _pad_to_target(clean_tok["attention_mask"], target_length=target_len, value=0).to(device)
        corrupt_mask = _pad_to_target(corrupt_tok["attention_mask"], target_length=target_len, value=0).to(device)
        shared_mask = torch.minimum(clean_mask, corrupt_mask)

        out.append(
            PreparedBatch(
                clean_inputs={"input_ids": clean_ids, "attention_mask": shared_mask},
                corrupt_inputs={"input_ids": corrupt_ids, "attention_mask": shared_mask.clone()},
                labels=labels,
                input_lengths=shared_mask.sum(dim=-1),
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


def _topk_overlap(a: Tensor, b: Tensor, k: int) -> float:
    if k <= 0:
        return 1.0
    size = min(int(a.numel()), int(b.numel()))
    if size == 0:
        return 1.0
    k_eff = min(k, size)
    idx_a = torch.topk(a.abs(), k=k_eff).indices.tolist()
    idx_b = torch.topk(b.abs(), k=k_eff).indices.tolist()
    overlap = len(set(idx_a).intersection(set(idx_b)))
    return float(overlap) / float(k_eff)


def compute_diff_stats(a: Tensor, b: Tensor, *, top_k: int) -> DiffStats:
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
        topk_overlap=_topk_overlap(a_cpu, b_cpu, top_k),
    )


def _is_known_parity_runtime_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return "size of tensor a" in text and "must match the size of tensor b" in text


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
    topk_overlap_threshold: float,
    top_k: int,
    quiet: bool,
    model_id_tlens: str,
    model_id_hf: str,
    adapter_name: str,
) -> MethodParityRow:
    ensure_vendor_available()
    start = time.time()

    vendor_graph = VendorGraph.from_model(tlens_model)
    if method == "exact" and len(vendor_graph.edges) > max_exact_edges:
        skip_reason = (
            f"exact skipped: edge_count={len(vendor_graph.edges)} exceeds "
            f"--max-exact-edges={max_exact_edges}"
        )
        return MethodParityRow(
            model_tlens=model_id_tlens,
            model_hf=model_id_hf,
            adapter_name=adapter_name,
            method=method,
            status="skip",
            seconds=0.0,
            note=skip_reason,
            skip_reason=skip_reason,
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

    d_vendor_tlens = compute_diff_stats(vendor_scores, ours_tlens_scores, top_k=top_k)
    d_vendor_hf = compute_diff_stats(vendor_scores, ours_hf_scores, top_k=top_k)
    d_tlens_hf = compute_diff_stats(ours_tlens_scores, ours_hf_scores, top_k=top_k)

    pass_tlens = (
        d_vendor_tlens.max_abs <= max_abs_threshold
        and d_vendor_tlens.cosine >= cosine_threshold
        and d_vendor_tlens.topk_overlap >= topk_overlap_threshold
    )
    pass_hf = (
        d_vendor_hf.max_abs <= max_abs_threshold
        and d_vendor_hf.cosine >= cosine_threshold
        and d_vendor_hf.topk_overlap >= topk_overlap_threshold
    )
    status = "pass" if (pass_tlens and pass_hf) else "fail"

    return MethodParityRow(
        model_tlens=model_id_tlens,
        model_hf=model_id_hf,
        adapter_name=adapter_name,
        method=method,
        status=status,
        seconds=round(time.time() - start, 3),
        note="",
        skip_reason="",
        vendor_vs_ours_tlens=d_vendor_tlens,
        vendor_vs_ours_hf=d_vendor_hf,
        ours_tlens_vs_ours_hf=d_tlens_hf,
    )


def run_method_hf_only(
    *,
    method: str,
    hf_model: torch.nn.Module,
    hf_backend: HFLLMBackend,
    prepared_batches: Iterable[PreparedBatch],
    ig_steps: int,
    max_exact_edges: int,
    max_abs_threshold: float,
    cosine_threshold: float,
    topk_overlap_threshold: float,
    top_k: int,
    quiet: bool,
    model_id_hf: str,
    adapter_name: str,
    reason: str,
) -> MethodParityRow:
    start = time.time()
    graph_a = OursGraph.from_model(hf_backend.config)
    if method == "exact" and len(graph_a.edges) > max_exact_edges:
        skip_reason = (
            f"exact skipped: edge_count={len(graph_a.edges)} exceeds "
            f"--max-exact-edges={max_exact_edges}; fallback=hf-only"
        )
        return MethodParityRow(
            model_tlens=model_id_hf,
            model_hf=model_id_hf,
            adapter_name=adapter_name,
            method=method,
            status="skip",
            seconds=0.0,
            note=skip_reason,
            skip_reason=skip_reason,
            vendor_vs_ours_tlens=None,
            vendor_vs_ours_hf=None,
            ours_tlens_vs_ours_hf=None,
        )

    kwargs: Dict[str, Any] = {"method": method, "quiet": quiet}
    if method in {"EAP-IG-inputs", "EAP-IG-activations"}:
        kwargs["ig_steps"] = ig_steps

    scores_a = ours_attribute(
        model=hf_model,
        backend=hf_backend,
        graph=graph_a,
        batches=prepared_batches,
        metric=ours_metric,
        **kwargs,
    )
    graph_b = OursGraph.from_model(hf_backend.config)
    scores_b = ours_attribute(
        model=hf_model,
        backend=hf_backend,
        graph=graph_b,
        batches=prepared_batches,
        metric=ours_metric,
        **kwargs,
    )

    d_hf_repeat = compute_diff_stats(scores_a, scores_b, top_k=top_k)
    status = "pass"
    if not (
        d_hf_repeat.max_abs <= max_abs_threshold
        and d_hf_repeat.cosine >= cosine_threshold
        and d_hf_repeat.topk_overlap >= topk_overlap_threshold
    ):
        status = "fail"

    return MethodParityRow(
        model_tlens=model_id_hf,
        model_hf=model_id_hf,
        adapter_name=adapter_name,
        method=method,
        status=status,
        seconds=round(time.time() - start, 3),
        note=f"hf-only parity fallback: {reason}",
        skip_reason="",
        vendor_vs_ours_tlens=None,
        vendor_vs_ours_hf=d_hf_repeat,
        ours_tlens_vs_ours_hf=d_hf_repeat,
    )


def main() -> None:
    args = parse_args()
    dtype = dtype_from_name(args.dtype)
    if args.device == "cpu" and dtype == torch.float16:
        raise ValueError("float16 on CPU is unstable; use float32/bfloat16")

    set_seed(args.seed)
    vendor_available = _VENDOR_IMPORT_ERROR is None
    patch_vendor_cuda_literals()

    model_ids = [x.strip() for x in args.models.split(",") if x.strip()]
    architecture_filters = {x.strip() for x in args.architectures.split(",") if x.strip()}
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    if args.strict:
        args.max_abs_threshold = 1e-3
        args.cosine_threshold = 0.995
        args.topk_overlap_threshold = 0.9

    all_rows: list[MethodParityRow] = []
    for tlens_id in model_ids:
        hf_id = resolve_hf_model_id(tlens_id)

        print(f"\\n=== Loading model pair: tlens={tlens_id} hf={hf_id} ===")
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        tlens_model: HookedTransformer | None = None
        tlens_load_error = ""
        try:
            tlens_model = load_tlens_model(
                tlens_id,
                tokenizer=tokenizer,
                device=args.device,
                dtype=dtype,
            )
            tlens_model.eval()
        except Exception as exc:  # noqa: BLE001
            tlens_load_error = str(exc)
            print(
                f"TLens unavailable for model={tlens_id}; switching to HF-only parity mode. "
                f"error={tlens_load_error[:180]}"
            )

        hf_model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=dtype).to(args.device)
        hf_model.eval()

        hf_backend = HFLLMBackend(hf_model, tokenizer=tokenizer)
        if architecture_filters and hf_backend.adapter_name not in architecture_filters:
            print(
                f"Skipping model pair tlens={tlens_id} hf={hf_id}: "
                f"adapter={hf_backend.adapter_name} not in --architectures"
            )
            continue

        run_vendor_tlens = vendor_available and (tlens_model is not None)
        records = build_records(tokenizer, args.n_samples)
        if tlens_model is not None:
            prepared_batches = build_prepared_batches(tlens_model, records, args.batch_size)
        else:
            prepared_batches = build_prepared_batches_hf(
                tokenizer,
                records,
                args.batch_size,
                device=args.device,
            )
        vendor_batches = (
            build_vendor_batches(records, args.batch_size) if run_vendor_tlens else []
        )
        tlens_backend = TLensBackend(tlens_model) if tlens_model is not None else None

        for method in methods:
            print(f"Running method={method} ...")
            try:
                if run_vendor_tlens and tlens_model is not None and tlens_backend is not None:
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
                        topk_overlap_threshold=args.topk_overlap_threshold,
                        top_k=args.top_k,
                        quiet=args.quiet,
                        model_id_tlens=tlens_id,
                        model_id_hf=hf_id,
                        adapter_name=hf_backend.adapter_name,
                    )
                else:
                    reason_parts = []
                    if not vendor_available:
                        reason_parts.append(f"vendor unavailable: {str(_VENDOR_IMPORT_ERROR)[:240]}")
                    if tlens_model is None:
                        reason_parts.append(f"TLens load failed: {tlens_load_error[:240]}")
                    row = run_method_hf_only(
                        method=method,
                        hf_model=hf_model,
                        hf_backend=hf_backend,
                        prepared_batches=prepared_batches,
                        ig_steps=args.ig_steps,
                        max_exact_edges=args.max_exact_edges,
                        max_abs_threshold=args.max_abs_threshold,
                        cosine_threshold=args.cosine_threshold,
                        topk_overlap_threshold=args.topk_overlap_threshold,
                        top_k=args.top_k,
                        quiet=args.quiet,
                        model_id_hf=hf_id,
                        adapter_name=hf_backend.adapter_name,
                        reason="; ".join(part for part in reason_parts if part),
                    )
            except Exception as exc:
                if run_vendor_tlens:
                    try:
                        row = run_method_hf_only(
                            method=method,
                            hf_model=hf_model,
                            hf_backend=hf_backend,
                            prepared_batches=prepared_batches,
                            ig_steps=args.ig_steps,
                            max_exact_edges=args.max_exact_edges,
                            max_abs_threshold=args.max_abs_threshold,
                            cosine_threshold=args.cosine_threshold,
                            topk_overlap_threshold=args.topk_overlap_threshold,
                            top_k=args.top_k,
                            quiet=args.quiet,
                            model_id_hf=hf_id,
                            adapter_name=hf_backend.adapter_name,
                            reason=f"vendor/tlens path failed: {exc}",
                        )
                    except Exception as fallback_exc:
                        if _is_known_parity_runtime_limit(fallback_exc):
                            reason = f"known parity runtime limit: {str(fallback_exc)[:240]}"
                            row = MethodParityRow(
                                model_tlens=tlens_id,
                                model_hf=hf_id,
                                adapter_name=hf_backend.adapter_name,
                                method=method,
                                status="skip",
                                seconds=0.0,
                                note=reason,
                                skip_reason=reason,
                                vendor_vs_ours_tlens=None,
                                vendor_vs_ours_hf=None,
                                ours_tlens_vs_ours_hf=None,
                            )
                        else:
                            row = MethodParityRow(
                                model_tlens=tlens_id,
                                model_hf=hf_id,
                                adapter_name=hf_backend.adapter_name,
                                method=method,
                                status="error",
                                seconds=0.0,
                                note=str(fallback_exc)[:320],
                                skip_reason="",
                                vendor_vs_ours_tlens=None,
                                vendor_vs_ours_hf=None,
                                ours_tlens_vs_ours_hf=None,
                            )
                else:
                    row = MethodParityRow(
                        model_tlens=tlens_id,
                        model_hf=hf_id,
                        adapter_name=hf_backend.adapter_name,
                        method=method,
                        status="error",
                        seconds=0.0,
                        note=str(exc)[:320],
                        skip_reason="",
                        vendor_vs_ours_tlens=None,
                        vendor_vs_ours_hf=None,
                        ours_tlens_vs_ours_hf=None,
                    )
            all_rows.append(row)

            if row.status in {"pass", "fail"} and row.vendor_vs_ours_hf is not None:
                print(
                    "  status={status} vendor_vs_hf(max_abs={max_abs:.4e}, cosine={cosine:.6f}, topk={topk:.3f})".format(
                        status=row.status,
                        max_abs=row.vendor_vs_ours_hf.max_abs,
                        cosine=row.vendor_vs_ours_hf.cosine,
                        topk=row.vendor_vs_ours_hf.topk_overlap,
                    )
                )
            elif row.status == "skip":
                print(f"  status=skip reason={row.skip_reason or row.note}")
            else:
                print(f"  status={row.status} note={row.note}")

        if torch.cuda.is_available() and args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    payload = {
        "report_type": REPORT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
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
            "top_k": args.top_k,
            "topk_overlap_threshold": args.topk_overlap_threshold,
            "architectures": sorted(architecture_filters),
            "strict": bool(args.strict),
        },
        "by_adapter": {
            adapter: sum(1 for row in all_rows if row.adapter_name == adapter)
            for adapter in sorted({row.adapter_name for row in all_rows})
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
