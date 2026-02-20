from multimodal_lm_eap_ig.backend.adapters._template import TemplateArchitectureAdapter


def test_template_adapter_contract_surface():
    adapter = TemplateArchitectureAdapter()

    assert hasattr(adapter, "name")
    assert hasattr(adapter, "arch_kind")
    assert callable(getattr(adapter, "match"))
    assert callable(getattr(adapter, "resolve"))
    assert callable(getattr(adapter, "projection_spec"))
    assert callable(getattr(adapter, "required_modules"))

    spec = adapter.projection_spec(None, "q")
    assert spec.kind in {"separate", "fused_conv1d", "fused_linear", "fused_linear_interleaved"}
    assert isinstance(adapter.required_modules(), list)
