import pytest
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    OPTConfig,
    OPTForCausalLM,
)

from meap.attribute import attribute
from meap.backend import HFLLMBackend
from meap.batch import PreparedBatch
from meap.graph import Graph


def _tiny_llama_lm() -> LlamaForCausalLM:
    cfg = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=32,
    )
    return LlamaForCausalLM(cfg)


def _tiny_opt_lm() -> OPTForCausalLM:
    cfg = OPTConfig(
        hidden_size=16,
        ffn_dim=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        vocab_size=128,
        max_position_embeddings=32,
    )
    return OPTForCausalLM(cfg)


def _prepared_batch() -> PreparedBatch:
    clean_ids = torch.tensor([[1, 2, 3, 4]])
    corrupt_ids = torch.tensor([[1, 2, 9, 4]])
    mask = torch.ones_like(clean_ids)
    return PreparedBatch(
        clean_inputs={"input_ids": clean_ids, "attention_mask": mask},
        corrupt_inputs={"input_ids": corrupt_ids, "attention_mask": mask.clone()},
        labels=None,
        input_lengths=mask.sum(dim=-1),
    )


def _metric(logits, clean_logits, batch):
    del clean_logits, batch
    return logits.sum()


def test_hf_backend_normalizes_llama_config_and_hook_scaffold():
    model = _tiny_llama_lm()
    backend = HFLLMBackend(model)

    assert backend.language_trunk_path == "model.model"
    assert backend.arch_kind == "llama_like"
    assert backend.config.n_layers == 1
    assert backend.config.n_heads == 4
    assert backend.config.n_key_value_heads == 2
    assert backend.config.d_model == 16
    assert backend.config.parallel_attn_mlp is False

    names = set(backend.supported_hook_names)
    assert "hook_embed" in names
    assert "blocks.0.attn.hook_result" in names
    assert "blocks.0.hook_q_input" in names
    assert "blocks.0.hook_k_input" in names
    assert "blocks.0.hook_v_input" in names
    assert "blocks.0.hook_mlp_in" in names
    assert "blocks.0.hook_mlp_out" in names
    assert "blocks.0.hook_resid_post" in names


def test_hf_backend_supports_gpt2_and_builds_hook_scaffold():
    gpt2 = GPT2LMHeadModel(
        GPT2Config(n_layer=1, n_head=2, n_embd=16, n_positions=32, vocab_size=128)
    )
    backend = HFLLMBackend(gpt2)

    assert backend.language_trunk_path == "model.transformer"
    assert backend.arch_kind == "gpt2_like"
    assert backend.config.n_layers == 1
    assert backend.config.n_heads == 2
    assert backend.config.d_model == 16
    assert backend.config.parallel_attn_mlp is False

    names = set(backend.supported_hook_names)
    assert "hook_embed" in names
    assert "blocks.0.attn.hook_result" in names
    assert "blocks.0.hook_q_input" in names
    assert "blocks.0.hook_k_input" in names
    assert "blocks.0.hook_v_input" in names
    assert "blocks.0.hook_mlp_in" in names
    assert "blocks.0.hook_mlp_out" in names
    assert "blocks.0.hook_resid_post" in names


def test_hf_backend_supports_adapter_name_override():
    gpt2 = GPT2LMHeadModel(
        GPT2Config(n_layer=1, n_head=2, n_embd=16, n_positions=32, vocab_size=128)
    )
    backend = HFLLMBackend(gpt2, adapter_name="gpt2_like")
    assert backend.adapter_name == "gpt2_like"


def test_hf_backend_supports_explicit_language_trunk_path():
    gpt2 = GPT2LMHeadModel(
        GPT2Config(n_layer=1, n_head=2, n_embd=16, n_positions=32, vocab_size=128)
    )
    backend = HFLLMBackend(gpt2, adapter_name="gpt2_like", language_trunk_path="model.transformer")
    assert backend.adapter_name == "gpt2_like"
    assert backend.language_trunk_path == "model.transformer"


def test_hf_backend_rejects_invalid_language_trunk_path():
    gpt2 = GPT2LMHeadModel(
        GPT2Config(n_layer=1, n_head=2, n_embd=16, n_positions=32, vocab_size=128)
    )
    with pytest.raises(ValueError, match="language_trunk_path"):
        HFLLMBackend(gpt2, language_trunk_path="model.not_a_real_path")


def test_hf_backend_rejects_unknown_adapter_name():
    gpt2 = GPT2LMHeadModel(
        GPT2Config(n_layer=1, n_head=2, n_embd=16, n_positions=32, vocab_size=128)
    )
    with pytest.raises(ValueError, match="Unknown adapter_name"):
        HFLLMBackend(gpt2, adapter_name="does_not_exist")


def test_hf_backend_supports_opt_and_builds_hook_scaffold():
    opt_model = _tiny_opt_lm()
    backend = HFLLMBackend(opt_model)

    assert backend.config.n_layers == 1
    assert backend.config.n_heads == 4
    assert backend.config.d_model == 16
    assert backend.config.parallel_attn_mlp is False

    names = set(backend.supported_hook_names)
    assert "hook_embed" in names
    assert "blocks.0.attn.hook_result" in names
    assert "blocks.0.hook_q_input" in names
    assert "blocks.0.hook_k_input" in names
    assert "blocks.0.hook_v_input" in names
    assert "blocks.0.hook_mlp_in" in names
    assert "blocks.0.hook_mlp_out" in names
    assert "blocks.0.hook_resid_post" in names


def test_hf_backend_ungroups_gqa_without_changing_logits():
    cfg = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=32,
    )
    baseline = LlamaForCausalLM(cfg).eval()
    model = LlamaForCausalLM(cfg).eval()
    model.load_state_dict(baseline.state_dict())

    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        baseline_logits = baseline(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).logits

    backend = HFLLMBackend(model)
    attn = model.model.layers[0].self_attn
    assert getattr(attn, "num_key_value_groups", None) == 1
    assert attn.k_proj.out_features == cfg.num_attention_heads * attn.head_dim
    assert attn.v_proj.out_features == cfg.num_attention_heads * attn.head_dim
    assert backend.config.n_key_value_heads == cfg.num_key_value_heads

    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).logits
    assert torch.allclose(logits, baseline_logits, atol=1e-6, rtol=1e-6)


def test_hf_backend_rejects_non_decoder_backbone():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)
            self.config = type("Cfg", (), {"hidden_size": 4, "num_hidden_layers": 1, "num_attention_heads": 1})()

    with pytest.raises(ValueError, match="Unsupported HF architecture") as exc_info:
        HFLLMBackend(DummyModel())
    assert "Resolution summary" in str(exc_info.value)


def test_hf_backend_reports_candidate_paths_for_invalid_decoder_shape():
    class BadDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = torch.nn.Linear(4, 4)

    class BadDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([BadDecoderLayer()])
            self.embed_tokens = torch.nn.Embedding(8, 4)

    class BadModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = BadDecoder()
            self.config = type(
                "Cfg",
                (),
                {"hidden_size": 4, "num_hidden_layers": 1, "num_attention_heads": 1},
            )()

    with pytest.raises(ValueError, match="Tried backbones:") as exc_info:
        HFLLMBackend(BadModel())
    text = str(exc_info.value)
    assert "missing attrs" in text or "missing required attrs" in text


def test_hf_backend_exposes_resolution_diagnostics_schema():
    model = _tiny_llama_lm()
    backend = HFLLMBackend(model)

    diagnostics = backend.resolution_diagnostics
    assert "candidate_backbones" in diagnostics
    assert "adapter_attempts" in diagnostics
    assert "selected" in diagnostics
    selected = diagnostics["selected"]
    assert selected["adapter"] == backend.adapter_name
    assert selected["path"] == backend.language_trunk_path


def test_attribute_smoke_with_hf_backend_succeeds():
    model = _tiny_llama_lm()
    backend = HFLLMBackend(model)
    graph = Graph.from_model(backend.config)

    scores = attribute(
        model=model,
        backend=backend,
        graph=graph,
        batches=[_prepared_batch()],
        metric=_metric,
        method="smoke",
    )

    assert scores.shape == (graph.n_forward, graph.n_backward)
    assert torch.all(scores == 0)


def test_attribute_smoke_with_hf_opt_backend_succeeds():
    model = _tiny_opt_lm()
    backend = HFLLMBackend(model)
    graph = Graph.from_model(backend.config)

    scores = attribute(
        model=model,
        backend=backend,
        graph=graph,
        batches=[_prepared_batch()],
        metric=_metric,
        method="smoke",
    )

    assert scores.shape == (graph.n_forward, graph.n_backward)
    assert torch.all(scores == 0)


def test_attribute_eap_ig_inputs_with_hf_backend_succeeds():
    model = _tiny_llama_lm()
    backend = HFLLMBackend(model)
    graph = Graph.from_model(backend.config)

    scores = attribute(
        model=model,
        backend=backend,
        graph=graph,
        batches=[_prepared_batch()],
        metric=_metric,
        method="EAP-IG-inputs",
        ig_steps=2,
    )

    assert scores.shape == (graph.n_forward, graph.n_backward)
    assert torch.isfinite(scores).all()


@pytest.mark.parametrize(
    ("method", "extra_kwargs"),
    [
        ("EAP", {}),
        ("EAP-IG-inputs", {"ig_steps": 2}),
        ("clean-corrupted", {}),
        ("EAP-IG-activations", {"ig_steps": 2}),
        ("exact", {}),
    ],
)
def test_attribute_non_smoke_methods_with_hf_backend_succeed(method, extra_kwargs):
    model = _tiny_llama_lm()
    backend = HFLLMBackend(model)
    graph = Graph.from_model(backend.config)

    scores = attribute(
        model=model,
        backend=backend,
        graph=graph,
        batches=[_prepared_batch()],
        metric=_metric,
        method=method,
        **extra_kwargs,
    )

    assert scores.shape == (graph.n_forward, graph.n_backward)
    assert torch.isfinite(scores).all()


def test_eap_ig_activations_is_batch_duplication_invariant():
    model = _tiny_llama_lm()
    backend = HFLLMBackend(model)
    graph = Graph.from_model(backend.config)
    batch = _prepared_batch()

    scores_one = attribute(
        model=model,
        backend=backend,
        graph=graph,
        batches=[batch],
        metric=_metric,
        method="EAP-IG-activations",
        ig_steps=2,
    ).clone()

    graph_two = Graph.from_model(backend.config)
    scores_two = attribute(
        model=model,
        backend=backend,
        graph=graph_two,
        batches=[batch, batch],
        metric=_metric,
        method="EAP-IG-activations",
        ig_steps=2,
    )

    assert torch.allclose(scores_one, scores_two, atol=1e-6, rtol=1e-6)


def test_attribute_requires_backend_for_non_tlens_model():
    graph = Graph.from_model(
        {
            "n_layers": 1,
            "n_heads": 1,
            "parallel_attn_mlp": True,
            "d_model": 16,
        }
    )
    with pytest.raises(ValueError, match="backend must be provided|A backend must be provided"):
        attribute(
            model=object(),
            graph=graph,
            batches=[_prepared_batch()],
            metric=_metric,
            method="smoke",
        )
