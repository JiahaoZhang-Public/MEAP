from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .operators import ProjectionAttentionOperator

BackendHookSpec = Tuple[str, Callable]


@dataclass
class BackendRunInputs:
    input_ids: Optional[Tensor]
    inputs_embeds: Optional[Tensor]
    attention_mask: Optional[Tensor]
    extra_fwd_hooks: List[BackendHookSpec] = field(default_factory=list)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendConfig:
    device: torch.device
    dtype: torch.dtype
    d_model: int
    n_layers: int
    n_heads: int
    parallel_attn_mlp: bool
    n_key_value_heads: Optional[int]
    use_attn_result: bool
    use_split_qkv_input: bool
    use_hook_mlp_in: bool
    use_normalization_before_and_after: bool = False

    def to_graph_dict(self) -> Dict[str, Any]:
        return {
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "parallel_attn_mlp": self.parallel_attn_mlp,
            "d_model": self.d_model,
        }


@runtime_checkable
class ModelBackend(Protocol):
    kind: str

    @property
    def config(self) -> BackendConfig:
        ...

    @property
    def tokenizer(self):
        ...

    @property
    def tokenization_model(self):
        ...

    def prepare_inputs(self, inputs: Dict[str, Tensor]) -> BackendRunInputs:
        ...

    def forward(
        self,
        run_inputs: BackendRunInputs,
        *,
        fwd_hooks: Optional[List[BackendHookSpec]] = None,
        bwd_hooks: Optional[List[BackendHookSpec]] = None,
    ) -> Tensor:
        ...

    def zero_grad(self) -> None:
        ...

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        ...


@dataclass
class HookTarget:
    module: torch.nn.Module
    mode: str
    output_index: Optional[int] = None


@dataclass
class AdapterResolution:
    path: str
    base_model: torch.nn.Module
    arch_kind: str
    layers: List[torch.nn.Module]
    layer_accessors: Dict[str, str]
    embed_module: torch.nn.Module
    resid_module: torch.nn.Module


@dataclass
class ProjectionSpec:
    kind: str


@runtime_checkable
class ArchitectureAdapter(Protocol):
    name: str
    arch_kind: str

    def match(self, backbone: torch.nn.Module) -> bool:
        ...

    def resolve(
        self,
        path: str,
        backbone: torch.nn.Module,
    ) -> Optional[AdapterResolution]:
        ...

    def attn_result_module(self, attn_module: torch.nn.Module) -> torch.nn.Module:
        ...

    def qkv_hook_modules(self, attn_module: torch.nn.Module) -> Dict[str, torch.nn.Module]:
        ...

    def projection_spec(self, attn_module: torch.nn.Module, qkv: str) -> ProjectionSpec:
        ...

    def attention_operator(
        self,
        attn_module: torch.nn.Module,
        *,
        qkv: str,
        backend_config: BackendConfig,
    ) -> "ProjectionAttentionOperator":
        ...

    @property
    def supports_gqa_ungroup(self) -> bool:
        ...

    def required_modules(self) -> Sequence[str]:
        ...


class BaseArchitectureAdapter:
    name: str
    arch_kind: str
    layer_accessors: Dict[str, str]
    required_attn_attrs: Sequence[str]

    def match(self, backbone: torch.nn.Module) -> bool:
        raise NotImplementedError

    def get_layers(self, backbone: torch.nn.Module) -> Optional[Sequence[torch.nn.Module]]:
        raise NotImplementedError

    def get_embed_module(self, backbone: torch.nn.Module) -> Optional[torch.nn.Module]:
        raise NotImplementedError

    def get_resid_module(
        self,
        backbone: torch.nn.Module,
        layers: List[torch.nn.Module],
    ) -> torch.nn.Module:
        raise NotImplementedError

    def attn_result_module(self, attn_module: torch.nn.Module) -> torch.nn.Module:
        raise NotImplementedError

    def qkv_hook_modules(self, attn_module: torch.nn.Module) -> Dict[str, torch.nn.Module]:
        raise NotImplementedError

    def projection_spec(self, attn_module: torch.nn.Module, qkv: str) -> ProjectionSpec:
        del attn_module, qkv
        return ProjectionSpec(kind="separate")

    def attention_operator(
        self,
        attn_module: torch.nn.Module,
        *,
        qkv: str,
        backend_config: BackendConfig,
    ) -> "ProjectionAttentionOperator":
        from .operators import ProjectionAttentionOperator

        projection_kind = self.projection_spec(attn_module, qkv).kind
        return ProjectionAttentionOperator(
            n_heads=int(backend_config.n_heads),
            d_model=int(backend_config.d_model),
            arch_kind=self.arch_kind,
            projection_kind=projection_kind,
        )

    @property
    def supports_gqa_ungroup(self) -> bool:
        return False

    def required_modules(self) -> Sequence[str]:
        fields = list(self.layer_accessors.values())
        fields.extend(self.required_attn_attrs)
        return fields

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

        for idx, layer in enumerate(layers):
            missing = [attr for attr in self.layer_accessors.values() if not hasattr(layer, attr)]
            if missing:
                raise ValueError(f"{self.name} @ {path}: layer {idx} missing required attrs {missing}")
            attn = getattr(layer, self.layer_accessors["attn"])
            missing_attn = [attr for attr in self.required_attn_attrs if not hasattr(attn, attr)]
            if missing_attn:
                raise ValueError(
                    f"{self.name} @ {path}: layer {idx} attention missing attrs {missing_attn}"
                )

        embed_module = self.get_embed_module(backbone)
        if embed_module is None:
            raise ValueError(f"{self.name} @ {path}: missing embedding module")

        resid_module = self.get_resid_module(backbone, layers)
        return AdapterResolution(
            path=path,
            base_model=backbone,
            arch_kind=self.arch_kind,
            layers=layers,
            layer_accessors=dict(self.layer_accessors),
            embed_module=embed_module,
            resid_module=resid_module,
        )


def parse_torch_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str) and hasattr(torch, dtype):
        parsed = getattr(torch, dtype)
        if isinstance(parsed, torch.dtype):
            return parsed
    return torch.float32


def first_parameter_device_dtype(model: torch.nn.Module) -> Tuple[torch.device, torch.dtype]:
    first_param = next(model.parameters(), None)
    if first_param is None:
        return torch.device("cpu"), torch.float32
    return first_param.device, first_param.dtype


def is_hooked_transformer_model(model: Any) -> bool:
    try:
        from transformer_lens import HookedTransformer
    except Exception:
        return False
    return isinstance(model, HookedTransformer)
