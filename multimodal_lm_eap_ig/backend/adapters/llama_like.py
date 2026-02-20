from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from ..base import BaseArchitectureAdapter, ProjectionSpec


class LlamaLikeAdapter(BaseArchitectureAdapter):
    name = "llama_like"
    arch_kind = "llama_like"
    layer_accessors = {
        "attn": "self_attn",
        "mlp": "mlp",
        "ln1": "input_layernorm",
        "ln2": "post_attention_layernorm",
    }
    required_attn_attrs = ("q_proj", "k_proj", "v_proj", "o_proj")

    def match(self, backbone: torch.nn.Module) -> bool:
        return hasattr(backbone, "layers")

    def get_layers(self, backbone: torch.nn.Module) -> Optional[Sequence[torch.nn.Module]]:
        return getattr(backbone, "layers", None)

    def get_embed_module(self, backbone: torch.nn.Module) -> Optional[torch.nn.Module]:
        return getattr(backbone, "embed_tokens", None)

    def get_resid_module(self, backbone: torch.nn.Module, layers: List[torch.nn.Module]) -> torch.nn.Module:
        return getattr(backbone, "norm", layers[-1])

    def attn_result_module(self, attn_module: torch.nn.Module) -> torch.nn.Module:
        return attn_module.o_proj

    def qkv_hook_modules(self, attn_module: torch.nn.Module) -> Dict[str, torch.nn.Module]:
        return {
            "q": attn_module.q_proj,
            "k": attn_module.k_proj,
            "v": attn_module.v_proj,
        }

    def projection_spec(self, attn_module: torch.nn.Module, qkv: str) -> ProjectionSpec:
        del attn_module, qkv
        return ProjectionSpec(kind="separate")

    @property
    def supports_gqa_ungroup(self) -> bool:
        return True

    def required_modules(self) -> Sequence[str]:
        return [
            "layers",
            "embed_tokens",
            "norm",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp",
            "input_layernorm",
            "post_attention_layernorm",
        ]
