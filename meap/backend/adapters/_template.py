"""Template for adding a new HF architecture adapter.

Five-minute onboarding checklist:
1. Copy this file to `meap/backend/adapters/<arch>_like.py`.
2. Rename class and set:
   - `name`
   - `arch_kind`
   - `layer_accessors`
   - `required_attn_attrs`
3. Implement:
   - `match`
   - `get_layers`
   - `get_embed_module`
   - `get_resid_module`
   - `attn_result_module`
   - `qkv_hook_modules`
   - `projection_spec`
4. Register the adapter:
   - add export in `meap/backend/adapters/__init__.py`
   - include adapter class in `_default_adapter_classes()` inside
     `meap/backend/registry.py`
5. Run required checks:
   - `python scripts/inspect_adapter_registry.py --model-id <model_id>`
   - `python scripts/smoke_hf_matrix.py --text-models <model_id> --multimodal-models ""`
   - `ruff check meap tests scripts`
   - `pytest -q`
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
        # Keep this strict: check layer stack + representative layer attrs.
        return False

    def get_layers(self, backbone: torch.nn.Module) -> Optional[Sequence[torch.nn.Module]]:
        return None

    def get_embed_module(self, backbone: torch.nn.Module) -> Optional[torch.nn.Module]:
        return None

    def get_resid_module(self, backbone: torch.nn.Module, layers: List[torch.nn.Module]) -> torch.nn.Module:
        del backbone
        return layers[-1]

    def attn_result_module(self, attn_module: torch.nn.Module) -> torch.nn.Module:
        # Return the module whose forward output corresponds to attention result.
        raise NotImplementedError

    def qkv_hook_modules(self, attn_module: torch.nn.Module) -> Dict[str, torch.nn.Module]:
        # Return {"q": module, "k": module, "v": module}.
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
