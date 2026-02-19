#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multimodal_lm_eap_ig.backend import HFLLMBackend, TLensBackend  # noqa: E402
from multimodal_lm_eap_ig.graph import Graph  # noqa: E402
from multimodal_lm_eap_ig.utils import forward_with_hooks, resolve_run_inputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare GPT-style backend consistency between HF and TLens backends."
    )
    parser.add_argument("--model-id", default="gpt2-small")
    parser.add_argument("--prompt", default="The quick brown fox")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    return parser.parse_args()


def _to_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _resolve_hf_model_id(model_id: str) -> str:
    # TLens uses "gpt2-small", while HF hub uses "gpt2" for the same checkpoint.
    if model_id == "gpt2-small":
        return "gpt2"
    return model_id


def _load_tlens_model(
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
    return model


def main() -> None:
    args = parse_args()
    dtype = _to_dtype(args.dtype)
    hf_model_id = _resolve_hf_model_id(args.model_id)

    if args.device == "cpu" and dtype == torch.float16:
        raise ValueError("float16 on CPU is unsupported/unstable; use float32 or bfloat16.")

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(args.prompt, return_tensors="pt")
    model_inputs = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }

    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_id, dtype=dtype)
    hf_model = hf_model.to(device=args.device)
    hf_model.eval()

    tlens_model = _load_tlens_model(
        args.model_id,
        tokenizer=tokenizer,
        device=args.device,
        dtype=dtype,
    )

    hf_backend = HFLLMBackend(hf_model, tokenizer=tokenizer)
    tlens_backend = TLensBackend(tlens_model)

    hf_graph = Graph.from_model(hf_backend.config)
    tlens_graph = Graph.from_model(tlens_backend.config)

    graph_matches = (
        hf_backend.config.n_layers == tlens_backend.config.n_layers
        and hf_backend.config.n_heads == tlens_backend.config.n_heads
        and hf_backend.config.d_model == tlens_backend.config.d_model
        and hf_backend.config.parallel_attn_mlp == tlens_backend.config.parallel_attn_mlp
        and hf_graph.n_forward == tlens_graph.n_forward
        and hf_graph.n_backward == tlens_graph.n_backward
        and set(hf_graph.nodes.keys()) == set(tlens_graph.nodes.keys())
        and set(hf_graph.edges.keys()) == set(tlens_graph.edges.keys())
    )

    captures: dict[str, torch.Tensor] = {}

    def capture_hf_embed(activations, hook):
        del hook
        captures["hf"] = activations.detach().cpu()
        return activations

    def capture_tlens_embed(activations, hook):
        del hook
        captures["tlens"] = activations.detach().cpu()
        return activations

    hf_run_inputs = resolve_run_inputs(hf_backend, model_inputs)
    tlens_run_inputs = resolve_run_inputs(tlens_backend, model_inputs)

    with torch.inference_mode():
        _ = forward_with_hooks(
            hf_backend,
            hf_run_inputs,
            fwd_hooks=[("hook_embed", capture_hf_embed)],
        )
        _ = forward_with_hooks(
            tlens_backend,
            tlens_run_inputs,
            fwd_hooks=[("hook_embed", capture_tlens_embed)],
        )

    if "hf" not in captures or "tlens" not in captures:
        raise RuntimeError("Failed to capture hook_embed activations from one or both backends")

    diff = (captures["hf"] - captures["tlens"]).abs()
    max_abs_diff = float(diff.max().item())
    embed_matches = torch.allclose(captures["hf"], captures["tlens"], atol=args.atol, rtol=args.rtol)

    print(f"Model (TLens): {args.model_id}")
    print(f"Model (HF): {hf_model_id}")
    print(f"Graph match: {graph_matches}")
    print(f"Embedding shape: {tuple(captures['hf'].shape)}")
    print(f"Embedding allclose (atol={args.atol}, rtol={args.rtol}): {embed_matches}")
    print(f"Embedding max abs diff: {max_abs_diff:.8f}")

    if not graph_matches or not embed_matches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
