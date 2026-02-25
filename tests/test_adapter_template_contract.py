import torch

from meap.backend.adapters._template import TemplateArchitectureAdapter
from meap.backend.base import BackendConfig


def test_template_adapter_contract_surface():
    adapter = TemplateArchitectureAdapter()

    assert hasattr(adapter, "name")
    assert hasattr(adapter, "arch_kind")
    assert callable(getattr(adapter, "match"))
    assert callable(getattr(adapter, "resolve"))
    assert callable(getattr(adapter, "projection_spec"))
    assert callable(getattr(adapter, "attention_operator"))
    assert callable(getattr(adapter, "required_modules"))

    spec = adapter.projection_spec(None, "q")
    assert spec.kind in {"separate", "fused_conv1d", "fused_linear", "fused_linear_interleaved"}
    operator = adapter.attention_operator(
        None,
        qkv="q",
        backend_config=BackendConfig(
            device=torch.device("cpu"),
            dtype=torch.float32,
            d_model=8,
            n_layers=1,
            n_heads=2,
            parallel_attn_mlp=False,
            n_key_value_heads=None,
            use_attn_result=True,
            use_split_qkv_input=True,
            use_hook_mlp_in=True,
        ),
    )
    assert operator.projection_kind in {"separate", "fused_conv1d", "fused_linear", "fused_linear_interleaved"}
    assert isinstance(adapter.required_modules(), list)
