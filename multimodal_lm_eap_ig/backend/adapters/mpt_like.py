from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from ..base import BaseArchitectureAdapter, ProjectionSpec


class MPTLikeAdapter(BaseArchitectureAdapter):
    name = "mpt_like"
    arch_kind = "mpt_like"
    layer_accessors = {
        "attn": "attn",
        "mlp": "ffn",
        "ln1": "norm_1",
        "ln2": "norm_2",
    }
    required_attn_attrs = ("Wqkv", "out_proj")

    def match(self, backbone: torch.nn.Module) -> bool:
        if not hasattr(backbone, "blocks"):
            return False
        blocks = getattr(backbone, "blocks", None)
        return bool(blocks) and hasattr(blocks[0], "attn")

    def get_layers(self, backbone: torch.nn.Module) -> Optional[Sequence[torch.nn.Module]]:
        return getattr(backbone, "blocks", None)

    def get_embed_module(self, backbone: torch.nn.Module) -> Optional[torch.nn.Module]:
        return getattr(backbone, "wte", None)

    def get_resid_module(self, backbone: torch.nn.Module, layers: List[torch.nn.Module]) -> torch.nn.Module:
        return getattr(backbone, "norm_f", layers[-1])

    def attn_result_module(self, attn_module: torch.nn.Module) -> torch.nn.Module:
        return attn_module.out_proj

    def qkv_hook_modules(self, attn_module: torch.nn.Module) -> Dict[str, torch.nn.Module]:
        return {
            "q": attn_module.Wqkv,
            "k": attn_module.Wqkv,
            "v": attn_module.Wqkv,
        }

    def projection_spec(self, attn_module: torch.nn.Module, qkv: str) -> ProjectionSpec:
        del attn_module, qkv
        return ProjectionSpec(kind="fused_linear")

    def required_modules(self) -> Sequence[str]:
        return [
            "blocks",
            "wte",
            "norm_f",
            "attn.Wqkv",
            "attn.out_proj",
            "ffn",
            "norm_1",
            "norm_2",
        ]
