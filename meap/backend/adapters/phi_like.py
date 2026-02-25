from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from ..base import (
    AdapterResolution,
    BackendConfig,
    BaseArchitectureAdapter,
    ProjectionSpec,
)
from ..operators import ProjectionAttentionOperator


class PhiLikeAdapter(BaseArchitectureAdapter):
    name = "phi_like"
    arch_kind = "phi_like"
    layer_accessors = {
        "attn": "self_attn",
        "mlp": "mlp",
        "ln1": "input_layernorm",
        "ln2": "input_layernorm",
    }
    required_attn_attrs: Sequence[str] = ()

    def match(self, backbone: torch.nn.Module) -> bool:
        if not hasattr(backbone, "layers"):
            return False
        class_name = backbone.__class__.__name__.lower()
        if "phi" in class_name:
            return True
        if hasattr(backbone, "config"):
            model_type = str(getattr(backbone.config, "model_type", "")).lower()
            return model_type.startswith("phi")
        return False

    def get_layers(self, backbone: torch.nn.Module) -> Optional[Sequence[torch.nn.Module]]:
        return getattr(backbone, "layers", None)

    def get_embed_module(self, backbone: torch.nn.Module) -> Optional[torch.nn.Module]:
        return getattr(backbone, "embed_tokens", None)

    def get_resid_module(self, backbone: torch.nn.Module, layers: List[torch.nn.Module]) -> torch.nn.Module:
        if hasattr(backbone, "norm"):
            return backbone.norm
        if hasattr(backbone, "final_layernorm"):
            return backbone.final_layernorm
        return layers[-1]

    def resolve(
        self,
        path: str,
        backbone: torch.nn.Module,
    ) -> Optional[AdapterResolution]:
        if not self.match(backbone):
            return None

        layers_raw = self.get_layers(backbone)
        if layers_raw is None:
            raise ValueError(f"{self.name} @ {path}: missing decoder layer stack")
        layers = list(layers_raw)
        if len(layers) == 0:
            raise ValueError(f"{self.name} @ {path}: decoder layer stack is empty")

        if all(hasattr(layer, "post_attention_layernorm") for layer in layers):
            ln2_attr = "post_attention_layernorm"
        elif all(hasattr(layer, "input_layernorm") for layer in layers):
            ln2_attr = "input_layernorm"
        else:
            raise ValueError(f"{self.name} @ {path}: cannot resolve ln2 attribute for phi family")

        layer_accessors = {
            "attn": "self_attn",
            "mlp": "mlp",
            "ln1": "input_layernorm",
            "ln2": ln2_attr,
        }
        for idx, layer in enumerate(layers):
            missing = [attr for attr in layer_accessors.values() if not hasattr(layer, attr)]
            if missing:
                raise ValueError(f"{self.name} @ {path}: layer {idx} missing required attrs {missing}")

            attn = getattr(layer, layer_accessors["attn"])
            has_separate = all(hasattr(attn, key) for key in ("q_proj", "k_proj", "v_proj"))
            has_fused = hasattr(attn, "qkv_proj")
            if not has_separate and not has_fused:
                raise ValueError(
                    f"{self.name} @ {path}: layer {idx} attention must expose "
                    "q_proj/k_proj/v_proj or qkv_proj"
                )
            if not (hasattr(attn, "o_proj") or hasattr(attn, "dense")):
                raise ValueError(f"{self.name} @ {path}: layer {idx} attention missing output projection")

        embed_module = self.get_embed_module(backbone)
        if embed_module is None:
            raise ValueError(f"{self.name} @ {path}: missing embedding module")

        resid_module = self.get_resid_module(backbone, layers)
        return AdapterResolution(
            path=path,
            base_model=backbone,
            arch_kind=self.arch_kind,
            layers=layers,
            layer_accessors=layer_accessors,
            embed_module=embed_module,
            resid_module=resid_module,
        )

    def attn_result_module(self, attn_module: torch.nn.Module) -> torch.nn.Module:
        if hasattr(attn_module, "o_proj"):
            return attn_module.o_proj
        return attn_module.dense

    def qkv_hook_modules(self, attn_module: torch.nn.Module) -> Dict[str, torch.nn.Module]:
        if hasattr(attn_module, "qkv_proj"):
            return {
                "q": attn_module.qkv_proj,
                "k": attn_module.qkv_proj,
                "v": attn_module.qkv_proj,
            }
        return {
            "q": attn_module.q_proj,
            "k": attn_module.k_proj,
            "v": attn_module.v_proj,
        }

    def projection_spec(self, attn_module: torch.nn.Module, qkv: str) -> ProjectionSpec:
        del qkv
        if hasattr(attn_module, "qkv_proj"):
            return ProjectionSpec(kind="fused_linear")
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

    def required_modules(self) -> Sequence[str]:
        return [
            "layers",
            "embed_tokens",
            "self_attn",
            "mlp",
            "input_layernorm",
        ]
