from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from ..base import BackendConfig, BaseArchitectureAdapter, ProjectionSpec
from ..operators import ProjectionAttentionOperator


class GemmaLikeAdapter(BaseArchitectureAdapter):
    name = "gemma_like"
    arch_kind = "gemma_like"
    layer_accessors = {
        "attn": "self_attn",
        "mlp": "mlp",
        "ln1": "input_layernorm",
        "ln2": "post_attention_layernorm",
    }
    required_attn_attrs = ("q_proj", "k_proj", "v_proj", "o_proj")

    def match(self, backbone: torch.nn.Module) -> bool:
        if not hasattr(backbone, "layers"):
            return False
        class_name = backbone.__class__.__name__.lower()
        if "gemma" in class_name:
            return True
        if hasattr(backbone, "config"):
            model_type = str(getattr(backbone.config, "model_type", "")).lower()
            return "gemma" in model_type
        return False

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

    def attention_operator(
        self,
        attn_module: torch.nn.Module,
        *,
        qkv: str,
        backend_config: BackendConfig,
    ) -> ProjectionAttentionOperator:
        return ProjectionAttentionOperator(
            n_heads=int(backend_config.n_heads),
            d_model=int(backend_config.d_model),
            arch_kind=self.arch_kind,
            projection_kind=self.projection_spec(attn_module, qkv).kind,
        )

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
