from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from ..base import BaseArchitectureAdapter, ProjectionSpec


class GPT2LikeAdapter(BaseArchitectureAdapter):
    name = "gpt2_like"
    arch_kind = "gpt2_like"
    layer_accessors = {
        "attn": "attn",
        "mlp": "mlp",
        "ln1": "ln_1",
        "ln2": "ln_2",
    }
    required_attn_attrs = ("c_attn", "c_proj")

    def match(self, backbone: torch.nn.Module) -> bool:
        return hasattr(backbone, "h")

    def get_layers(self, backbone: torch.nn.Module) -> Optional[Sequence[torch.nn.Module]]:
        return getattr(backbone, "h", None)

    def get_embed_module(self, backbone: torch.nn.Module) -> Optional[torch.nn.Module]:
        return getattr(backbone, "wte", None)

    def get_resid_module(self, backbone: torch.nn.Module, layers: List[torch.nn.Module]) -> torch.nn.Module:
        return getattr(backbone, "ln_f", layers[-1])

    def attn_result_module(self, attn_module: torch.nn.Module) -> torch.nn.Module:
        return attn_module.c_proj

    def qkv_hook_modules(self, attn_module: torch.nn.Module) -> Dict[str, torch.nn.Module]:
        return {
            "q": attn_module.c_attn,
            "k": attn_module.c_attn,
            "v": attn_module.c_attn,
        }

    def projection_spec(self, attn_module: torch.nn.Module, qkv: str) -> ProjectionSpec:
        del attn_module, qkv
        return ProjectionSpec(kind="fused_conv1d")

    def required_modules(self) -> Sequence[str]:
        return [
            "h",
            "wte",
            "ln_f",
            "attn.c_attn",
            "attn.c_proj",
            "mlp",
            "ln_1",
            "ln_2",
        ]
