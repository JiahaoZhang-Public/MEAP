"""Template for adding a new HF architecture adapter.

Usage:
1. Copy this file and rename class/file to your architecture.
2. Implement `match`, layer/module getters, and attention hook mapping.
3. Register the adapter via `register_architecture_adapter`.
4. Add unit tests similar to tests/test_backend_falcon_like.py.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from ..base import BaseArchitectureAdapter, ProjectionSpec


class TemplateArchitectureAdapter(BaseArchitectureAdapter):
    name = "template_like"
    arch_kind = "template_like"
    layer_accessors = {
        "attn": "<attention_module_attr>",
        "mlp": "<mlp_attr>",
        "ln1": "<norm1_attr>",
        "ln2": "<norm2_attr>",
    }
    required_attn_attrs = ("<q_proj_like>", "<k_proj_like>", "<v_proj_like>", "<o_proj_like>")

    def match(self, backbone: torch.nn.Module) -> bool:
        # Keep this strict: check both layer stack and one representative layer attr.
        return False

    def get_layers(self, backbone: torch.nn.Module) -> Optional[Sequence[torch.nn.Module]]:
        return None

    def get_embed_module(self, backbone: torch.nn.Module) -> Optional[torch.nn.Module]:
        return None

    def get_resid_module(self, backbone: torch.nn.Module, layers: List[torch.nn.Module]) -> torch.nn.Module:
        del backbone
        return layers[-1]

    def attn_result_module(self, attn_module: torch.nn.Module) -> torch.nn.Module:
        raise NotImplementedError

    def qkv_hook_modules(self, attn_module: torch.nn.Module) -> Dict[str, torch.nn.Module]:
        raise NotImplementedError

    def projection_spec(self, attn_module: torch.nn.Module, qkv: str) -> ProjectionSpec:
        del attn_module, qkv
        # One of: separate | fused_conv1d | fused_linear | fused_linear_interleaved
        return ProjectionSpec(kind="separate")

    @property
    def supports_gqa_ungroup(self) -> bool:
        return False

    def required_modules(self) -> Sequence[str]:
        return [
            "<layer_stack_path>",
            "<embedding_path>",
            "<residual_norm_path>",
            "<attention_path>",
            "<mlp_path>",
        ]
