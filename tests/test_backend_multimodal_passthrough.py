import torch
from transformers import LlamaConfig, LlamaForCausalLM

from multimodal_lm_eap_ig.backend import HFLLMBackend


class TinyQwen2VLLikeModel(torch.nn.Module):
    """Qwen2-VL-like wrapper exposing language_model plus multimodal kwargs."""

    def __init__(self):
        super().__init__()
        cfg = LlamaConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=64,
            max_position_embeddings=32,
        )
        self.language_model = LlamaForCausalLM(cfg)
        self.model = self.language_model.model
        self.lm_head = self.language_model.lm_head
        self.config = self.language_model.config
        self.last_seen = {}

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        pixel_values=None,
        image_grid_thw=None,
        use_cache=False,
        return_dict=True,
        **kwargs,
    ):
        self.last_seen = {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "extra_keys": sorted(kwargs.keys()),
        }
        return self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=return_dict,
            **kwargs,
        )


def test_hf_backend_accepts_language_model_wrapper_and_multimodal_kwargs():
    model = TinyQwen2VLLikeModel()
    backend = HFLLMBackend(model)

    assert backend.config.n_layers == 1
    assert backend.config.n_heads == 4

    run_inputs = backend.prepare_inputs(
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "pixel_values": torch.randn(1, 3, 2, 2),
            "image_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }
    )

    assert "pixel_values" in run_inputs.model_kwargs
    assert "image_grid_thw" in run_inputs.model_kwargs

    with torch.inference_mode():
        logits = backend.forward(run_inputs)

    assert logits.shape == (1, 3, model.config.vocab_size)
    assert model.last_seen["pixel_values"] is not None
    assert model.last_seen["image_grid_thw"] is not None
