import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformers import GPT2Config, GPT2LMHeadModel

from meap.backend import HFLLMBackend, TLensBackend
from meap.graph import Graph
from meap.utils import forward_with_hooks, resolve_run_inputs


def _build_tiny_gpt2_models() -> tuple[GPT2LMHeadModel, HookedTransformer]:
    hf_config = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=16,
        n_positions=32,
        n_ctx=32,
        vocab_size=64,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    hf_model = GPT2LMHeadModel(hf_config)
    hf_model.eval()

    tlens_config = HookedTransformerConfig(
        n_layers=2,
        n_heads=2,
        d_model=16,
        d_head=8,
        d_mlp=32,
        n_ctx=32,
        d_vocab=64,
        act_fn="gelu_new",
        normalization_type="LN",
        parallel_attn_mlp=False,
    )
    tlens_model = HookedTransformer(tlens_config)
    tlens_model.eval()
    tlens_model.cfg.use_attn_result = True
    tlens_model.cfg.use_split_qkv_input = True
    tlens_model.cfg.use_hook_mlp_in = True

    with torch.no_grad():
        tlens_model.W_E.copy_(hf_model.transformer.wte.weight.to(tlens_model.W_E.dtype))

    return hf_model, tlens_model


def test_gpt2_graph_is_consistent_between_hf_and_tlens_backends():
    hf_model, tlens_model = _build_tiny_gpt2_models()

    hf_backend = HFLLMBackend(hf_model)
    tlens_backend = TLensBackend(tlens_model)

    hf_graph = Graph.from_model(hf_backend.config)
    tlens_graph = Graph.from_model(tlens_backend.config)

    assert hf_backend.config.n_layers == tlens_backend.config.n_layers
    assert hf_backend.config.n_heads == tlens_backend.config.n_heads
    assert hf_backend.config.d_model == tlens_backend.config.d_model
    assert hf_backend.config.parallel_attn_mlp == tlens_backend.config.parallel_attn_mlp

    assert hf_graph.n_forward == tlens_graph.n_forward
    assert hf_graph.n_backward == tlens_graph.n_backward
    assert set(hf_graph.nodes.keys()) == set(tlens_graph.nodes.keys())
    assert set(hf_graph.edges.keys()) == set(tlens_graph.edges.keys())


def test_gpt2_hook_embed_is_consistent_between_hf_and_tlens_backends():
    hf_model, tlens_model = _build_tiny_gpt2_models()

    hf_backend = HFLLMBackend(hf_model)
    tlens_backend = TLensBackend(tlens_model)

    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    captures: dict[str, torch.Tensor] = {}

    def capture_hf_embed(activations, hook):
        del hook
        captures["hf"] = activations.detach().cpu()
        return activations

    def capture_tlens_embed(activations, hook):
        del hook
        captures["tlens"] = activations.detach().cpu()
        return activations

    hf_inputs = resolve_run_inputs(
        hf_backend,
        {"input_ids": input_ids.clone(), "attention_mask": attention_mask.clone()},
    )
    tlens_inputs = resolve_run_inputs(
        tlens_backend,
        {"input_ids": input_ids.clone(), "attention_mask": attention_mask.clone()},
    )

    with torch.inference_mode():
        _ = forward_with_hooks(hf_backend, hf_inputs, fwd_hooks=[("hook_embed", capture_hf_embed)])
        _ = forward_with_hooks(
            tlens_backend,
            tlens_inputs,
            fwd_hooks=[("hook_embed", capture_tlens_embed)],
        )

    assert "hf" in captures
    assert "tlens" in captures
    assert captures["hf"].shape == captures["tlens"].shape
    assert torch.allclose(captures["hf"], captures["tlens"], atol=1e-6, rtol=1e-6)
