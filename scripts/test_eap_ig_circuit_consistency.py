#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VENDOR_SRC = REPO_ROOT / "vendor" / "eap-ig" / "src"
if str(VENDOR_SRC) not in sys.path:
    sys.path.insert(0, str(VENDOR_SRC))

from eap.attribute import attribute as vendor_attribute  # noqa: E402
from eap.graph import Graph as VendorGraph  # noqa: E402

from multimodal_lm_eap_ig.attribute import attribute as ours_attribute  # noqa: E402
from multimodal_lm_eap_ig.backend import HFLLMBackend, TLensBackend  # noqa: E402
from multimodal_lm_eap_ig.batch import (  # noqa: E402
    PreparedBatch,
    text_batch_to_prepared_batch,
    validate_prepared_batch,
)
from multimodal_lm_eap_ig.graph import Graph as OursGraph  # noqa: E402


@dataclass
class IOIRecord:
    clean: str
    corrupt: str
    labels: list[int]


class IOIDataset(Dataset):
    def __init__(self, records: Sequence[IOIRecord]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        return record.clean, record.corrupt, record.labels


def collate_eap(batch):
    clean, corrupt, labels = zip(*batch)
    return list(clean), list(corrupt), torch.tensor(labels, dtype=torch.long)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare vendor EAP-IG vs multimodal_lm_eap_ig on simple IOI-style GPT2 data.",
    )
    parser.add_argument("--model-id", default="gpt2-small")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--ig-steps", type=int, default=2)
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--quiet", action="store_true")
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


def logit_positions(logits: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
    idx = torch.arange(logits.size(0), device=logits.device)
    return logits[idx, input_lengths.to(logits.device) - 1]


def vendor_metric(logits, clean_logits, input_lengths, labels):
    del clean_logits
    last_logits = logit_positions(logits, input_lengths)
    pair = torch.gather(last_logits, -1, labels.to(logits.device))
    return -(pair[:, 0] - pair[:, 1]).mean()


def ours_metric(logits, clean_logits, batch: PreparedBatch):
    del clean_logits
    labels = batch.labels
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels, dtype=torch.long, device=logits.device)
    else:
        labels = labels.to(logits.device)

    last_logits = logit_positions(logits, batch.input_lengths)
    pair = torch.gather(last_logits, -1, labels)
    return -(pair[:, 0] - pair[:, 1]).mean()


def choose_single_token_names(tokenizer, minimum: int) -> list[str]:
    candidates = [
        "John",
        "Mary",
        "James",
        "Robert",
        "Michael",
        "David",
        "Thomas",
        "William",
        "Daniel",
        "Joseph",
    ]
    names = []
    for name in candidates:
        token_ids = tokenizer.encode(f" {name}", add_special_tokens=False)
        if len(token_ids) == 1:
            names.append(name)
    if len(names) < minimum:
        raise RuntimeError(
            f"Need at least {minimum} single-token names, found {len(names)}: {names}"
        )
    return names


def build_records(tokenizer, n_samples: int) -> list[IOIRecord]:
    names = choose_single_token_names(tokenizer, minimum=4)
    records: list[IOIRecord] = []
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

        correct_token = tokenizer.encode(f" {indirect_obj}", add_special_tokens=False)[0]
        incorrect_token = tokenizer.encode(f" {subject}", add_special_tokens=False)[0]
        records.append(
            IOIRecord(
                clean=clean,
                corrupt=corrupt,
                labels=[correct_token, incorrect_token],
            )
        )
    return records


def build_prepared_batches(
    tokenization_model,
    records: Sequence[IOIRecord],
    batch_size: int,
) -> list[PreparedBatch]:
    batches: list[PreparedBatch] = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        clean_text = [x.clean for x in chunk]
        corrupt_text = [x.corrupt for x in chunk]
        labels = torch.tensor([x.labels for x in chunk], dtype=torch.long)
        prepared = text_batch_to_prepared_batch(
            tokenization_model,
            clean_text,
            corrupt_text,
            labels,
        )
        validate_prepared_batch(prepared)
        batches.append(prepared)
    return batches


@contextmanager
def patch_vendor_cuda_allocations(target_device: str):
    if torch.cuda.is_available():
        yield
        return

    original_zeros = torch.zeros

    def patched_zeros(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = target_device
        return original_zeros(*args, **kwargs)

    torch.zeros = patched_zeros
    try:
        yield
    finally:
        torch.zeros = original_zeros


def topk_edges(graph, k: int) -> list[str]:
    edges = list(graph.edges.values())
    edges = sorted(edges, key=lambda e: abs(float(e.score)), reverse=True)
    return [edge.name for edge in edges[:k]]


def compare_graphs(reference_graph, candidate_graph, *, topk: int):
    shared_edges = sorted(set(reference_graph.edges.keys()) & set(candidate_graph.edges.keys()))
    if not shared_edges:
        raise RuntimeError("No shared edge names to compare between graphs")

    ref_vec = torch.tensor(
        [float(reference_graph.edges[name].score) for name in shared_edges],
        dtype=torch.float32,
    )
    cand_vec = torch.tensor(
        [float(candidate_graph.edges[name].score) for name in shared_edges],
        dtype=torch.float32,
    )

    diff = (ref_vec - cand_vec).abs()
    corr = float(torch.corrcoef(torch.stack([ref_vec, cand_vec]))[0, 1].item())

    ref_topk = topk_edges(reference_graph, topk)
    cand_topk = topk_edges(candidate_graph, topk)
    ref_set = set(ref_topk)
    cand_set = set(cand_topk)
    intersection = len(ref_set & cand_set)
    union = len(ref_set | cand_set)
    jaccard = float(intersection / union) if union else 1.0

    return {
        "shared_edges": len(shared_edges),
        "max_score_abs_diff": float(diff.max().item()),
        "mean_score_abs_diff": float(diff.mean().item()),
        "score_pearson": corr,
        "topk": topk,
        "topk_overlap": intersection,
        "topk_jaccard": jaccard,
        "reference_topk_preview": ref_topk[:10],
        "candidate_topk_preview": cand_topk[:10],
    }


def run_vendor_eap_ig(model, dataloader, ig_steps: int, quiet: bool):
    vendor_graph = VendorGraph.from_model(model)
    with patch_vendor_cuda_allocations(str(model.cfg.device)):
        vendor_attribute(
            model,
            vendor_graph,
            dataloader,
            vendor_metric,
            method="EAP-IG-inputs",
            ig_steps=ig_steps,
            quiet=quiet,
        )
    return vendor_graph


def run_ours_tlens_eap_ig(model, dataloader, ig_steps: int, quiet: bool):
    ours_graph = OursGraph.from_model(model)
    ours_attribute(
        model=model,
        backend=TLensBackend(model),
        graph=ours_graph,
        batches=dataloader,
        metric=ours_metric,
        method="EAP-IG-inputs",
        ig_steps=ig_steps,
        quiet=quiet,
    )
    return ours_graph


def run_ours_hf_eap_ig(
    hf_model,
    tlens_tokenization_model,
    tokenizer,
    records: Sequence[IOIRecord],
    batch_size: int,
    ig_steps: int,
    quiet: bool,
):
    hf_backend = HFLLMBackend(hf_model, tokenizer=tokenizer)
    hf_graph = OursGraph.from_model(hf_backend.config)
    prepared_batches = build_prepared_batches(tlens_tokenization_model, records, batch_size)

    ours_attribute(
        model=hf_model,
        backend=hf_backend,
        graph=hf_graph,
        batches=prepared_batches,
        metric=ours_metric,
        method="EAP-IG-inputs",
        ig_steps=ig_steps,
        quiet=quiet,
    )
    return hf_graph


def main() -> None:
    args = parse_args()
    dtype = to_dtype(args.dtype)

    if args.device == "cpu" and dtype == torch.float16:
        raise ValueError("float16 on CPU is unstable; use float32 or bfloat16")

    hf_model_id = resolve_hf_model_id(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = build_records(tokenizer, args.n_samples)
    dataset = IOIDataset(records)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_eap)

    tlens_model = HookedTransformer.from_pretrained(
        args.model_id,
        tokenizer=tokenizer,
        device=args.device,
        dtype=dtype,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        default_padding_side="right",
    )
    tlens_model.eval()
    tlens_model.cfg.use_attn_result = True
    tlens_model.cfg.use_split_qkv_input = True
    tlens_model.cfg.use_hook_mlp_in = True

    vendor_graph = run_vendor_eap_ig(
        model=tlens_model,
        dataloader=dataloader,
        ig_steps=args.ig_steps,
        quiet=args.quiet,
    )
    ours_tlens_graph = run_ours_tlens_eap_ig(
        model=tlens_model,
        dataloader=dataloader,
        ig_steps=args.ig_steps,
        quiet=args.quiet,
    )

    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_id, dtype=dtype).to(args.device)
    hf_model.eval()
    ours_hf_graph = run_ours_hf_eap_ig(
        hf_model=hf_model,
        tlens_tokenization_model=tlens_model,
        tokenizer=tokenizer,
        records=records,
        batch_size=args.batch_size,
        ig_steps=args.ig_steps,
        quiet=args.quiet,
    )

    tlens_compare = compare_graphs(vendor_graph, ours_tlens_graph, topk=args.topk)
    hf_compare = compare_graphs(vendor_graph, ours_hf_graph, topk=args.topk)

    print(f"Model (TLens): {args.model_id}")
    print(f"Model (HF): {hf_model_id}")
    print(f"Samples: {args.n_samples}, batch_size={args.batch_size}, ig_steps={args.ig_steps}")

    print("--- Vendor vs Ours (TLens backend) ---")
    print(f"shared_edges={tlens_compare['shared_edges']}")
    print(f"max_score_abs_diff={tlens_compare['max_score_abs_diff']:.8f}")
    print(f"mean_score_abs_diff={tlens_compare['mean_score_abs_diff']:.8f}")
    print(f"score_pearson={tlens_compare['score_pearson']:.8f}")
    print(
        f"topk_overlap={tlens_compare['topk_overlap']}/{tlens_compare['topk']} "
        f"(jaccard={tlens_compare['topk_jaccard']:.8f})"
    )
    print(f"vendor_topk_preview={tlens_compare['reference_topk_preview']}")
    print(f"ours_tlens_topk_preview={tlens_compare['candidate_topk_preview']}")

    print("--- Vendor vs Ours (HF backend) ---")
    print(f"shared_edges={hf_compare['shared_edges']}")
    print(f"max_score_abs_diff={hf_compare['max_score_abs_diff']:.8f}")
    print(f"mean_score_abs_diff={hf_compare['mean_score_abs_diff']:.8f}")
    print(f"score_pearson={hf_compare['score_pearson']:.8f}")
    print(
        f"topk_overlap={hf_compare['topk_overlap']}/{hf_compare['topk']} "
        f"(jaccard={hf_compare['topk_jaccard']:.8f})"
    )
    print(f"vendor_topk_preview={hf_compare['reference_topk_preview']}")
    print(f"ours_hf_topk_preview={hf_compare['candidate_topk_preview']}")


if __name__ == "__main__":
    main()
