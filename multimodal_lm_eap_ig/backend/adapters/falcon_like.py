from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from ..base import BaseArchitectureAdapter, ProjectionSpec


class FalconLikeAdapter(BaseArchitectureAdapter):
    name = "falcon_like"
    arch_kind = "falcon_like"
    layer_accessors = {
        "attn": "self_attention",
        "mlp": "mlp",
        "ln1": "input_layernorm",
        "ln2": "input_layernorm",
    }
    required_attn_attrs = ("query_key_value", "dense")

    def match(self, backbone: torch.nn.Module) -> bool:
        if not hasattr(backbone, "h"):
            return False
        layers = getattr(backbone, "h", None)
        return bool(layers) and hasattr(layers[0], "self_attention")

    def get_layers(self, backbone: torch.nn.Module) -> Optional[Sequence[torch.nn.Module]]:
        return getattr(backbone, "h", None)

    def get_embed_module(self, backbone: torch.nn.Module) -> Optional[torch.nn.Module]:
        return getattr(backbone, "word_embeddings", None)

    def get_resid_module(self, backbone: torch.nn.Module, layers: List[torch.nn.Module]) -> torch.nn.Module:
        return getattr(backbone, "ln_f", layers[-1])

    def attn_result_module(self, attn_module: torch.nn.Module) -> torch.nn.Module:
        return attn_module.dense

    def qkv_hook_modules(self, attn_module: torch.nn.Module) -> Dict[str, torch.nn.Module]:
        return {
            "q": attn_module.query_key_value,
            "k": attn_module.query_key_value,
            "v": attn_module.query_key_value,
        }

    def projection_spec(self, attn_module: torch.nn.Module, qkv: str) -> ProjectionSpec:
        del qkv
        if bool(getattr(attn_module, "new_decoder_architecture", False)):
            return ProjectionSpec(kind="fused_linear_interleaved")
        return ProjectionSpec(kind="fused_linear")

    def required_modules(self) -> Sequence[str]:
        return [
            "h",
            "word_embeddings",
            "ln_f",
            "self_attention.query_key_value",
            "self_attention.dense",
            "mlp",
            "input_layernorm",
        ]
