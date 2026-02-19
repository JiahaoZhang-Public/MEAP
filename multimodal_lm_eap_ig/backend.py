from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import (
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
class _HookTarget:
    module: torch.nn.Module
    mode: str  # 'forward' or 'pre'
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


@runtime_checkable
class ArchitectureAdapter(Protocol):
    name: str
    arch_kind: str

    def resolve_from_backbone(
        self,
        path: str,
        backbone: torch.nn.Module,
    ) -> Optional[AdapterResolution]:
        ...

    def attn_result_module(self, attn_module: torch.nn.Module) -> torch.nn.Module:
        ...

    def qkv_hook_modules(self, attn_module: torch.nn.Module) -> Dict[str, torch.nn.Module]:
        ...


def _iter_decoder_backbone_candidates(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    candidates: List[Tuple[str, torch.nn.Module]] = []

    def _add(path: str, module: Any) -> None:
        if isinstance(module, torch.nn.Module):
            candidates.append((path, module))

    _add("model", model)

    if hasattr(model, "language_model"):
        lm = model.language_model
        _add("model.language_model", lm)
        _add("model.language_model.model", getattr(lm, "model", None))
        _add("model.language_model.model.decoder", getattr(getattr(lm, "model", None), "decoder", None))
        _add("model.language_model.transformer", getattr(lm, "transformer", None))
        _add("model.language_model.decoder", getattr(lm, "decoder", None))

    _add("model.model", getattr(model, "model", None))
    _add("model.model.decoder", getattr(getattr(model, "model", None), "decoder", None))
    _add("model.transformer", getattr(model, "transformer", None))
    _add("model.decoder", getattr(model, "decoder", None))

    deduped: List[Tuple[str, torch.nn.Module]] = []
    seen = set()
    for path, module in candidates:
        marker = id(module)
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append((path, module))
    return deduped


class _BaseArchitectureAdapter:
    name: str
    arch_kind: str
    layer_accessors: Dict[str, str]
    required_attn_attrs: Sequence[str]

    def matches_backbone(self, backbone: torch.nn.Module) -> bool:
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

    def resolve_from_backbone(
        self,
        path: str,
        backbone: torch.nn.Module,
    ) -> Optional[AdapterResolution]:
        if not self.matches_backbone(backbone):
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
                raise ValueError(
                    f"{self.name} @ {path}: layer {idx} missing required attrs {missing}"
                )
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


class LlamaLikeAdapter(_BaseArchitectureAdapter):
    name = "llama_like"
    arch_kind = "llama_like"
    layer_accessors = {
        "attn": "self_attn",
        "mlp": "mlp",
        "ln1": "input_layernorm",
        "ln2": "post_attention_layernorm",
    }
    required_attn_attrs = ("q_proj", "k_proj", "v_proj", "o_proj")

    def matches_backbone(self, backbone: torch.nn.Module) -> bool:
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


class GPT2LikeAdapter(_BaseArchitectureAdapter):
    name = "gpt2_like"
    arch_kind = "gpt2_like"
    layer_accessors = {
        "attn": "attn",
        "mlp": "mlp",
        "ln1": "ln_1",
        "ln2": "ln_2",
    }
    required_attn_attrs = ("c_attn", "c_proj")

    def matches_backbone(self, backbone: torch.nn.Module) -> bool:
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


class OPTLikeAdapter(_BaseArchitectureAdapter):
    name = "opt_like"
    arch_kind = "opt_like"
    layer_accessors = {
        "attn": "self_attn",
        "mlp": "fc2",
        "ln1": "self_attn_layer_norm",
        "ln2": "final_layer_norm",
    }
    required_attn_attrs = ("q_proj", "k_proj", "v_proj", "out_proj")

    def matches_backbone(self, backbone: torch.nn.Module) -> bool:
        return hasattr(backbone, "layers")

    def get_layers(self, backbone: torch.nn.Module) -> Optional[Sequence[torch.nn.Module]]:
        return getattr(backbone, "layers", None)

    def get_embed_module(self, backbone: torch.nn.Module) -> Optional[torch.nn.Module]:
        return getattr(backbone, "embed_tokens", None)

    def get_resid_module(self, backbone: torch.nn.Module, layers: List[torch.nn.Module]) -> torch.nn.Module:
        if hasattr(backbone, "final_layer_norm"):
            return backbone.final_layer_norm
        if hasattr(backbone, "project_out"):
            return backbone.project_out
        return layers[-1]

    def attn_result_module(self, attn_module: torch.nn.Module) -> torch.nn.Module:
        return attn_module.out_proj

    def qkv_hook_modules(self, attn_module: torch.nn.Module) -> Dict[str, torch.nn.Module]:
        return {
            "q": attn_module.q_proj,
            "k": attn_module.k_proj,
            "v": attn_module.v_proj,
        }


def _parse_torch_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str) and hasattr(torch, dtype):
        parsed = getattr(torch, dtype)
        if isinstance(parsed, torch.dtype):
            return parsed
    return torch.float32


def _first_parameter_device_dtype(model: torch.nn.Module) -> Tuple[torch.device, torch.dtype]:
    first_param = next(model.parameters(), None)
    if first_param is None:
        return torch.device("cpu"), torch.float32
    return first_param.device, first_param.dtype


def _is_hooked_transformer_model(model: Any) -> bool:
    try:
        from transformer_lens import HookedTransformer
    except Exception:
        return False
    return isinstance(model, HookedTransformer)


class TLensBackend:
    kind = "tlens"

    def __init__(self, model: Any):
        if not _is_hooked_transformer_model(model):
            raise TypeError("TLensBackend requires a transformer_lens.HookedTransformer instance")
        self.model = model
        cfg = model.cfg
        device = torch.device(getattr(cfg, "device", "cpu"))
        dtype = _parse_torch_dtype(getattr(cfg, "dtype", torch.float32))
        self._config = BackendConfig(
            device=device,
            dtype=dtype,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            parallel_attn_mlp=cfg.parallel_attn_mlp,
            n_key_value_heads=cfg.n_key_value_heads,
            use_attn_result=bool(getattr(cfg, "use_attn_result", False)),
            use_split_qkv_input=bool(getattr(cfg, "use_split_qkv_input", False)),
            use_hook_mlp_in=bool(getattr(cfg, "use_hook_mlp_in", False)),
            use_normalization_before_and_after=bool(
                getattr(cfg, "use_normalization_before_and_after", False)
            ),
        )

    @property
    def config(self) -> BackendConfig:
        return self._config

    @property
    def tokenizer(self):
        return self.model.tokenizer

    @property
    def tokenization_model(self):
        return self.model

    def prepare_inputs(self, inputs: Dict[str, Tensor]) -> BackendRunInputs:
        device = self._config.device
        dtype = self._config.dtype

        tokens = inputs.get("input_ids")
        if tokens is None:
            tokens = inputs.get("tokens")

        input_embeds = inputs.get("inputs_embeds")

        if tokens is None and input_embeds is None:
            raise ValueError("Model inputs must include input_ids/tokens or inputs_embeds")

        if input_embeds is not None:
            input_embeds = input_embeds.to(device=device, dtype=dtype)

        if tokens is None:
            if input_embeds is None or input_embeds.ndim != 3:
                raise ValueError("inputs_embeds must be rank-3 [batch, seq, d_model]")
            pad_token_id = getattr(self.model.tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(self.model.tokenizer, "eos_token_id", 0)
            tokens = torch.full(
                input_embeds.shape[:2],
                int(pad_token_id),
                device=device,
                dtype=torch.long,
            )
        else:
            tokens = tokens.to(device=device, dtype=torch.long)

        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=device)
        else:
            attention_mask = torch.ones(tokens.shape, device=device, dtype=torch.long)

        if attention_mask.shape != tokens.shape:
            raise ValueError(
                "attention_mask shape must match tokens shape. "
                f"Got attention_mask={tuple(attention_mask.shape)} tokens={tuple(tokens.shape)}"
            )

        extra_fwd_hooks: List[BackendHookSpec] = []
        if input_embeds is not None:

            def _embed_override_hook(
                activations: Tensor,
                hook,
                prepared_embeds: Tensor = input_embeds,
            ) -> Tensor:
                return prepared_embeds

            extra_fwd_hooks.append(("hook_embed", _embed_override_hook))

        return BackendRunInputs(
            input_ids=tokens,
            inputs_embeds=None,
            attention_mask=attention_mask,
            extra_fwd_hooks=extra_fwd_hooks,
            model_kwargs={},
        )

    def forward(
        self,
        run_inputs: BackendRunInputs,
        *,
        fwd_hooks: Optional[List[BackendHookSpec]] = None,
        bwd_hooks: Optional[List[BackendHookSpec]] = None,
    ) -> Tensor:
        merged_fwd_hooks = list(run_inputs.extra_fwd_hooks)
        if fwd_hooks:
            merged_fwd_hooks.extend(fwd_hooks)

        kwargs = {"attention_mask": run_inputs.attention_mask}

        if not merged_fwd_hooks and not bwd_hooks:
            return self.model(run_inputs.input_ids, **kwargs)

        hook_kwargs = {"fwd_hooks": merged_fwd_hooks}
        if bwd_hooks is not None:
            hook_kwargs["bwd_hooks"] = bwd_hooks

        with self.model.hooks(**hook_kwargs):
            return self.model(run_inputs.input_ids, **kwargs)

    def zero_grad(self) -> None:
        self.model.zero_grad(set_to_none=True)

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return self.model.parameters()


class HFLLMBackend:
    kind = "hf"

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        tokenizer=None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if not hasattr(model, "config"):
            raise TypeError("HFLLMBackend requires a model with a HuggingFace-like .config")

        self.model = model
        if device is not None or dtype is not None:
            cast_kwargs: Dict[str, Any] = {}
            if device is not None:
                cast_kwargs["device"] = device
            if dtype is not None:
                cast_kwargs["dtype"] = dtype
            self.model.to(**cast_kwargs)

        self._tokenizer = tokenizer
        self._adapter_registry: List[ArchitectureAdapter] = [
            LlamaLikeAdapter(),
            GPT2LikeAdapter(),
            OPTLikeAdapter(),
        ]
        self._adapter, resolution = self._resolve_architecture(self.model)
        self._base_model = resolution.base_model
        self._arch_kind = resolution.arch_kind
        self._layers = resolution.layers
        self._layer_accessors = resolution.layer_accessors
        self._embed_module = resolution.embed_module
        self._resid_module = resolution.resid_module
        self._hook_targets = self._build_hook_targets()
        self._config = self._normalize_config(self.model)
        self._ln1_residual_cache: Dict[int, Tensor] = {}

    @property
    def config(self) -> BackendConfig:
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def tokenization_model(self):
        return None

    @property
    def supported_hook_names(self) -> List[str]:
        return sorted(self._hook_targets.keys())

    def _resolve_architecture(
        self,
        model: torch.nn.Module,
    ) -> Tuple[ArchitectureAdapter, AdapterResolution]:
        candidates = _iter_decoder_backbone_candidates(model)
        errors: List[str] = []

        for path, backbone in candidates:
            for adapter in self._adapter_registry:
                try:
                    resolved = adapter.resolve_from_backbone(path, backbone)
                except ValueError as exc:
                    errors.append(str(exc))
                    continue
                if resolved is not None:
                    return adapter, resolved

        candidate_paths = [path for path, _ in candidates]
        details = "; ".join(errors[:6]) if errors else "no adapter produced a compatible backbone"
        raise ValueError(
            "Unsupported HF architecture for HFLLMBackend. "
            f"Tried backbones: {candidate_paths}. "
            f"Expected decoder-like structure with attention, MLP, and norm modules. Details: {details}"
        )

    def _normalize_config(self, model: torch.nn.Module) -> BackendConfig:
        cfg = model.config
        d_model = (
            getattr(cfg, "hidden_size", None)
            if getattr(cfg, "hidden_size", None) is not None
            else getattr(cfg, "n_embd", None)
        )
        n_layers = (
            getattr(cfg, "num_hidden_layers", None)
            if getattr(cfg, "num_hidden_layers", None) is not None
            else getattr(cfg, "n_layer", None)
        )
        n_heads = (
            getattr(cfg, "num_attention_heads", None)
            if getattr(cfg, "num_attention_heads", None) is not None
            else getattr(cfg, "n_head", None)
        )
        missing_cfg = []
        if d_model is None:
            missing_cfg.append("hidden_size/n_embd")
        if n_layers is None:
            missing_cfg.append("num_hidden_layers/n_layer")
        if n_heads is None:
            missing_cfg.append("num_attention_heads/n_head")
        if missing_cfg:
            raise ValueError(
                "Unsupported HF config for HFLLMBackend. Missing fields: "
                f"{missing_cfg}."
            )

        device, model_dtype = _first_parameter_device_dtype(self.model)
        n_key_value_heads = getattr(cfg, "num_key_value_heads", None)

        return BackendConfig(
            device=device,
            dtype=model_dtype,
            d_model=int(d_model),
            n_layers=int(n_layers),
            n_heads=int(n_heads),
            parallel_attn_mlp=False,
            n_key_value_heads=(None if n_key_value_heads is None else int(n_key_value_heads)),
            use_attn_result=True,
            use_split_qkv_input=True,
            use_hook_mlp_in=True,
            use_normalization_before_and_after=False,
        )

    def _build_hook_targets(self) -> Dict[str, _HookTarget]:
        targets: Dict[str, _HookTarget] = {"hook_embed": _HookTarget(self._embed_module, "forward")}

        attn_attr = self._layer_accessors["attn"]
        mlp_attr = self._layer_accessors["mlp"]
        for layer_idx, layer in enumerate(self._layers):
            attn_module = getattr(layer, attn_attr)
            mlp_module = getattr(layer, mlp_attr)
            ln2_module = getattr(layer, self._layer_accessors["ln2"])

            attn_result_module = self._adapter.attn_result_module(attn_module)
            targets[f"blocks.{layer_idx}.attn.hook_result"] = _HookTarget(
                attn_result_module,
                "pre",
            )
            qkv_modules = self._adapter.qkv_hook_modules(attn_module)
            targets[f"blocks.{layer_idx}.hook_q_input"] = _HookTarget(qkv_modules["q"], "forward")
            targets[f"blocks.{layer_idx}.hook_k_input"] = _HookTarget(qkv_modules["k"], "forward")
            targets[f"blocks.{layer_idx}.hook_v_input"] = _HookTarget(qkv_modules["v"], "forward")

            targets[f"blocks.{layer_idx}.hook_mlp_out"] = _HookTarget(mlp_module, "forward")
            targets[f"blocks.{layer_idx}.hook_mlp_in"] = _HookTarget(ln2_module, "forward")

        targets[f"blocks.{len(self._layers) - 1}.hook_resid_post"] = _HookTarget(
            self._resid_module,
            "pre",
        )

        return targets

    def prepare_inputs(self, inputs: Dict[str, Tensor]) -> BackendRunInputs:
        device = self._config.device
        dtype = self._config.dtype

        input_ids = inputs.get("input_ids")
        if input_ids is None:
            input_ids = inputs.get("tokens")
        inputs_embeds = inputs.get("inputs_embeds")

        if input_ids is None and inputs_embeds is None:
            raise ValueError("Model inputs must include input_ids/tokens or inputs_embeds")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("HFLLMBackend expects exactly one of input_ids/tokens or inputs_embeds")

        if input_ids is not None:
            input_ids = input_ids.to(device=device, dtype=torch.long)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(device=device, dtype=dtype)

        if input_ids is not None:
            batch_shape = input_ids.shape
        else:
            if inputs_embeds is None or inputs_embeds.ndim != 3:
                raise ValueError("inputs_embeds must be rank-3 [batch, seq, d_model]")
            batch_shape = inputs_embeds.shape[:2]

        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=device)
        else:
            attention_mask = torch.ones(batch_shape, device=device, dtype=torch.long)

        if attention_mask.shape != batch_shape:
            raise ValueError(
                "attention_mask shape must match token/embed sequence shape. "
                f"Got attention_mask={tuple(attention_mask.shape)} expected={tuple(batch_shape)}"
            )

        model_kwargs: Dict[str, Any] = {}
        core_keys = {"input_ids", "tokens", "inputs_embeds", "attention_mask"}
        for key, value in inputs.items():
            if key in core_keys:
                continue
            if torch.is_tensor(value):
                if torch.is_floating_point(value):
                    model_kwargs[key] = value.to(device=device, dtype=dtype)
                else:
                    model_kwargs[key] = value.to(device=device)
            else:
                model_kwargs[key] = value

        return BackendRunInputs(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            extra_fwd_hooks=[],
            model_kwargs=model_kwargs,
        )

    def _make_forward_hook(self, name: str, hook_fn: Callable, target: _HookTarget):
        return self._make_composed_forward_hook(name, target, [hook_fn], [])

    def _apply_forward_hook_chain(
        self,
        activation: Tensor,
        hook_ctx: SimpleNamespace,
        fwd_hook_fns: List[Callable],
    ) -> Tensor:
        updated = activation
        for hook_fn in fwd_hook_fns:
            maybe_output = hook_fn(updated, hook_ctx)
            if maybe_output is not None:
                updated = maybe_output
        return updated

    def _attach_backward_hook_chain(
        self,
        activation: Tensor,
        hook_ctx: SimpleNamespace,
        bwd_hook_fns: List[Callable],
    ) -> None:
        if not bwd_hook_fns:
            return
        if not torch.is_tensor(activation):
            raise RuntimeError(
                f"Backward hooks for {hook_ctx.name} require tensor activation, got {type(activation)}"
            )
        if not activation.requires_grad:
            return

        def _bwd_chain(grad: Tensor) -> Tensor:
            updated = grad
            for hook_fn in bwd_hook_fns:
                maybe_grad = hook_fn(updated, hook_ctx)
                if maybe_grad is not None:
                    updated = maybe_grad
            return updated

        activation.register_hook(_bwd_chain)

    def _make_composed_forward_hook(
        self,
        name: str,
        target: _HookTarget,
        fwd_hook_fns: List[Callable],
        bwd_hook_fns: List[Callable],
    ):
        hook_ctx = SimpleNamespace(name=name)

        def _wrapped(module, inputs, output):
            if name.endswith(".hook_mlp_in"):
                if len(inputs) == 0:
                    raise RuntimeError(f"{name} expects at least one module input tensor")
                ln_input = inputs[0]
                updated = self._apply_forward_hook_chain(ln_input, hook_ctx, fwd_hook_fns)
                if updated is not ln_input:
                    raise RuntimeError(
                        "HFLLMBackend hook_mlp_in does not support activation replacement"
                    )
                self._attach_mlp_in_backward_hook_from_ln_output(
                    ln_output=output,
                    ln_input=ln_input,
                    ln_module=module,
                    hook_name=name,
                    hook_ctx=hook_ctx,
                    bwd_hook_fns=bwd_hook_fns,
                )
                return output

            if (
                name.endswith(".hook_q_input")
                or name.endswith(".hook_k_input")
                or name.endswith(".hook_v_input")
            ):
                if len(inputs) == 0:
                    raise RuntimeError(f"{name} expects at least one module input tensor")
                projected = self._project_qkv_input_for_hook(inputs[0], name)
                updated = self._apply_forward_hook_chain(projected, hook_ctx, fwd_hook_fns)
                if updated is not projected:
                    raise RuntimeError(
                        "HFLLMBackend q/k/v input hook does not support activation replacement"
                    )
                self._attach_qkv_backward_hook_from_projection_output(
                    projection_output=output,
                    projection_module=module,
                    hook_name=name,
                    hook_ctx=hook_ctx,
                    bwd_hook_fns=bwd_hook_fns,
                )
                return output

            if target.output_index is None:
                updated = self._apply_forward_hook_chain(output, hook_ctx, fwd_hook_fns)
                self._attach_backward_hook_chain(updated, hook_ctx, bwd_hook_fns)
                return updated

            if not isinstance(output, (tuple, list)):
                raise RuntimeError(
                    f"Hook point {name} expected tuple/list output for index "
                    f"{target.output_index}, got {type(output)}"
                )
            if not (0 <= target.output_index < len(output)):
                raise RuntimeError(
                    f"Hook point {name} output_index={target.output_index} out of range "
                    f"for output length {len(output)}"
                )

            selected = output[target.output_index]
            updated = self._apply_forward_hook_chain(selected, hook_ctx, fwd_hook_fns)
            self._attach_backward_hook_chain(updated, hook_ctx, bwd_hook_fns)
            if updated is selected:
                return output

            output_items = list(output)
            output_items[target.output_index] = updated
            if isinstance(output, tuple):
                return tuple(output_items)
            return output_items

        return _wrapped

    def _make_composed_pre_hook(
        self,
        name: str,
        fwd_hook_fns: List[Callable],
        bwd_hook_fns: List[Callable],
    ):
        hook_ctx = SimpleNamespace(name=name)

        def _wrapped(module, inputs):
            if len(inputs) == 0:
                return None

            if name.endswith(".attn.hook_result"):
                current = inputs[0]
                projected = self._project_attention_result_from_proj_input(current, module, name)
                updated = self._apply_forward_hook_chain(projected, hook_ctx, fwd_hook_fns)
                self._attach_backward_hook_chain(updated, hook_ctx, bwd_hook_fns)
                if updated is not projected:
                    raise RuntimeError(
                        "HFLLMBackend attn.hook_result hook does not support activation replacement"
                    )
                return None

            if (
                name.endswith(".hook_q_input")
                or name.endswith(".hook_k_input")
                or name.endswith(".hook_v_input")
            ):
                raise RuntimeError(f"{name} expected forward hook target, got pre-hook target")

            current = inputs[0]
            updated = self._apply_forward_hook_chain(current, hook_ctx, fwd_hook_fns)
            self._attach_backward_hook_chain(updated, hook_ctx, bwd_hook_fns)
            if updated is current:
                return None
            return (updated, *inputs[1:])

        return _wrapped

    def _project_attention_result_from_proj_input(
        self,
        projection_input: Tensor,
        projection_module: torch.nn.Module,
        hook_name: str,
    ) -> Tensor:
        if projection_input.ndim != 3:
            raise RuntimeError(
                f"{hook_name} expected rank-3 projection input [batch, seq, d_model], "
                f"got {tuple(projection_input.shape)}"
            )
        if not hasattr(projection_module, "weight"):
            raise RuntimeError(f"{hook_name} projection module missing weight parameter")

        weight = projection_module.weight
        if weight.ndim != 2:
            raise RuntimeError(f"{hook_name} projection weight must be rank-2, got {weight.ndim}")

        d_model = int(projection_input.shape[-1])
        if d_model % self._config.n_heads != 0:
            raise RuntimeError(
                f"{hook_name} cannot split d_model={d_model} into n_heads={self._config.n_heads}"
            )
        d_head = d_model // self._config.n_heads
        if int(weight.shape[1]) != d_model:
            if not (
                self._arch_kind == "gpt2_like"
                and projection_module.__class__.__name__ == "Conv1D"
                and int(weight.shape[0]) == d_model
                and int(weight.shape[1]) == d_model
            ):
                raise RuntimeError(
                    f"{hook_name} projection weight input dim mismatch: "
                    f"weight_in={int(weight.shape[1])}, d_model={d_model}"
                )

        projection_heads = projection_input.view(
            projection_input.shape[0],
            projection_input.shape[1],
            self._config.n_heads,
            d_head,
        )
        if self._arch_kind == "gpt2_like" and projection_module.__class__.__name__ == "Conv1D":
            weight_heads = weight.view(self._config.n_heads, d_head, d_model)
            return torch.einsum("bphd,hdm->bphm", projection_heads, weight_heads)

        weight_heads = weight.view(int(weight.shape[0]), self._config.n_heads, d_head)
        return torch.einsum("bphd,ohd->bpho", projection_heads, weight_heads)

    def _project_qkv_input_for_hook(
        self,
        qkv_input: Tensor,
        hook_name: str,
    ) -> Tensor:
        if qkv_input.ndim != 3:
            raise RuntimeError(
                f"{hook_name} expected rank-3 projection input [batch, seq, d_model], "
                f"got {tuple(qkv_input.shape)}"
            )
        return qkv_input.unsqueeze(2).expand(
            qkv_input.shape[0],
            qkv_input.shape[1],
            self._config.n_heads,
            qkv_input.shape[2],
        )

    def _qkv_letter_from_hook_name(self, hook_name: str) -> str:
        if hook_name.endswith(".hook_q_input"):
            return "q"
        if hook_name.endswith(".hook_k_input"):
            return "k"
        if hook_name.endswith(".hook_v_input"):
            return "v"
        raise ValueError(f"Cannot infer qkv letter from hook name: {hook_name}")

    def _layer_index_from_hook_name(self, hook_name: str) -> int:
        parts = hook_name.split(".")
        if len(parts) < 2 or parts[0] != "blocks":
            raise ValueError(f"Cannot infer layer index from hook name: {hook_name}")
        return int(parts[1])

    def _layernorm_input_grads(
        self,
        grads_wrt_ln_out: Tensor,
        ln_input: Tensor,
        layernorm_module: torch.nn.Module,
    ) -> Tensor:
        if not isinstance(layernorm_module, torch.nn.LayerNorm):
            return grads_wrt_ln_out

        ln_input_fp = ln_input.float()
        grads_fp = grads_wrt_ln_out.float()
        weight = layernorm_module.weight.float()
        eps = float(layernorm_module.eps)
        n_dim = ln_input_fp.shape[-1]

        mean = ln_input_fp.mean(dim=-1, keepdim=True)
        var = ((ln_input_fp - mean) ** 2).mean(dim=-1, keepdim=True)
        std = torch.sqrt(var + eps)
        xhat = (ln_input_fp - mean) / std

        gw = grads_fp * weight.view(1, 1, 1, -1)
        sum_gw = gw.sum(dim=-1, keepdim=True)
        sum_gw_xhat = (gw * xhat.unsqueeze(2)).sum(dim=-1, keepdim=True)
        grads_ln_in = (1.0 / n_dim) / std.unsqueeze(2) * (
            n_dim * gw - sum_gw - xhat.unsqueeze(2) * sum_gw_xhat
        )
        return grads_ln_in.to(dtype=grads_wrt_ln_out.dtype)

    def _project_qkv_gradient_to_input_heads(
        self,
        grad_output: Tensor,
        projection_module: torch.nn.Module,
        hook_name: str,
    ) -> Tensor:
        if grad_output.ndim != 3:
            raise RuntimeError(
                f"{hook_name} expected rank-3 projection output gradients [batch, seq, dim], "
                f"got {tuple(grad_output.shape)}"
            )
        if not hasattr(projection_module, "weight"):
            raise RuntimeError(f"{hook_name} projection module missing weight parameter")

        weight = projection_module.weight
        if weight.ndim != 2:
            raise RuntimeError(f"{hook_name} projection weight must be rank-2, got {weight.ndim}")

        qkv_letter = self._qkv_letter_from_hook_name(hook_name)
        qkv_index = "qkv".index(qkv_letter)
        d_model = self._config.d_model
        d_head = d_model // self._config.n_heads

        # GPT2 Conv1D-style fused qkv projection: weight shape [in, 3*d_model].
        if int(weight.shape[0]) == d_model and int(weight.shape[1]) == 3 * d_model:
            start = qkv_index * d_model
            end = (qkv_index + 1) * d_model
            grad_chunk = grad_output[:, :, start:end]
            grad_heads = grad_chunk.view(
                grad_chunk.shape[0],
                grad_chunk.shape[1],
                self._config.n_heads,
                d_head,
            )
            weight_chunk = weight[:, start:end].view(d_model, self._config.n_heads, d_head)
            projected = torch.einsum("bphd,mhd->bphm", grad_heads, weight_chunk)
        else:
            # Linear projection case: weight shape [out, in].
            out_dim = int(weight.shape[0])
            in_dim = int(weight.shape[1])
            if in_dim != d_model:
                raise RuntimeError(
                    f"{hook_name} projection input dim mismatch: weight_in={in_dim}, expected={d_model}"
                )
            if out_dim % d_head != 0:
                raise RuntimeError(
                    f"{hook_name} projection output dim {out_dim} is not divisible by d_head={d_head}"
                )
            out_heads = out_dim // d_head
            grad_heads = grad_output.view(
                grad_output.shape[0],
                grad_output.shape[1],
                out_heads,
                d_head,
            )
            weight_heads = weight.view(out_heads, d_head, d_model)
            per_head_grads = torch.einsum("bphd,hdm->bphm", grad_heads, weight_heads)
            if out_heads == self._config.n_heads:
                projected = per_head_grads
            else:
                if self._config.n_heads % out_heads != 0:
                    raise RuntimeError(
                        f"{hook_name} cannot broadcast out_heads={out_heads} to n_heads={self._config.n_heads}"
                    )
                repeat_factor = self._config.n_heads // out_heads
                projected = per_head_grads.repeat_interleave(repeat_factor, dim=2)

        layer_idx = self._layer_index_from_hook_name(hook_name)
        if layer_idx in self._ln1_residual_cache:
            ln_input = self._ln1_residual_cache[layer_idx]
            ln1_module = getattr(self._layers[layer_idx], self._layer_accessors["ln1"])
            return self._layernorm_input_grads(projected, ln_input, ln1_module)
        return projected

    def _attach_qkv_backward_hook_from_projection_output(
        self,
        projection_output: Tensor,
        projection_module: torch.nn.Module,
        hook_name: str,
        hook_ctx: SimpleNamespace,
        bwd_hook_fns: List[Callable],
    ) -> None:
        if not bwd_hook_fns:
            return
        if not torch.is_tensor(projection_output):
            raise RuntimeError(
                f"{hook_name} backward hooks require tensor projection output, got {type(projection_output)}"
            )
        if not projection_output.requires_grad:
            return

        def _bwd_chain(grad: Tensor) -> Tensor:
            projected_grad = self._project_qkv_gradient_to_input_heads(grad, projection_module, hook_name)
            updated = projected_grad
            for hook_fn in bwd_hook_fns:
                maybe_grad = hook_fn(updated, hook_ctx)
                if maybe_grad is not None:
                    updated = maybe_grad
            return grad

        projection_output.register_hook(_bwd_chain)

    def _attach_mlp_in_backward_hook_from_ln_output(
        self,
        ln_output: Tensor,
        ln_input: Tensor,
        ln_module: torch.nn.Module,
        hook_name: str,
        hook_ctx: SimpleNamespace,
        bwd_hook_fns: List[Callable],
    ) -> None:
        if not bwd_hook_fns:
            return
        if not torch.is_tensor(ln_output):
            raise RuntimeError(
                f"{hook_name} backward hooks require tensor layernorm output, got {type(ln_output)}"
            )
        if not ln_output.requires_grad:
            return

        cached_ln_input = ln_input.detach()

        def _bwd_chain(grad: Tensor) -> Tensor:
            projected_grad = self._layernorm_input_grads(
                grad.unsqueeze(2),
                cached_ln_input,
                ln_module,
            ).squeeze(2)
            updated = projected_grad
            for hook_fn in bwd_hook_fns:
                maybe_grad = hook_fn(updated, hook_ctx)
                if maybe_grad is not None:
                    updated = maybe_grad
            return grad

        ln_output.register_hook(_bwd_chain)

    def forward(
        self,
        run_inputs: BackendRunInputs,
        *,
        fwd_hooks: Optional[List[BackendHookSpec]] = None,
        bwd_hooks: Optional[List[BackendHookSpec]] = None,
    ) -> Tensor:
        merged_fwd_hooks = list(run_inputs.extra_fwd_hooks)
        if fwd_hooks:
            merged_fwd_hooks.extend(fwd_hooks)

        merged_bwd_hooks = list(bwd_hooks) if bwd_hooks else []

        if not merged_fwd_hooks and not merged_bwd_hooks:
            outputs = self.model(
                input_ids=run_inputs.input_ids,
                inputs_embeds=run_inputs.inputs_embeds,
                attention_mask=run_inputs.attention_mask,
                use_cache=False,
                return_dict=True,
                **run_inputs.model_kwargs,
            )
            if hasattr(outputs, "logits") and outputs.logits is not None:
                return outputs.logits
            if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                return outputs.last_hidden_state
            raise RuntimeError("HF model forward output does not expose logits or last_hidden_state")

        fwd_by_name: Dict[str, List[Callable]] = {}
        bwd_by_name: Dict[str, List[Callable]] = {}
        hook_order: List[str] = []

        for name, hook_fn in merged_fwd_hooks:
            if name not in fwd_by_name:
                fwd_by_name[name] = []
            fwd_by_name[name].append(hook_fn)
            if name not in hook_order:
                hook_order.append(name)

        for name, hook_fn in merged_bwd_hooks:
            if name not in bwd_by_name:
                bwd_by_name[name] = []
            bwd_by_name[name].append(hook_fn)
            if name not in hook_order:
                hook_order.append(name)

        self._ln1_residual_cache = {}
        handles = []
        try:
            captured_ln1_layers = set()
            for name in hook_order:
                if not (
                    name.endswith(".hook_q_input")
                    or name.endswith(".hook_k_input")
                    or name.endswith(".hook_v_input")
                ):
                    continue
                layer_idx = self._layer_index_from_hook_name(name)
                if layer_idx in captured_ln1_layers:
                    continue
                ln1_module = getattr(self._layers[layer_idx], self._layer_accessors["ln1"])

                def _capture_ln1_input(module, inputs, idx: int = layer_idx):
                    if len(inputs) > 0:
                        self._ln1_residual_cache[idx] = inputs[0].detach()

                handles.append(ln1_module.register_forward_pre_hook(_capture_ln1_input))
                captured_ln1_layers.add(layer_idx)

            for name in hook_order:
                target = self._hook_targets.get(name)
                if target is None:
                    raise ValueError(
                        f"Unsupported HF hook point: {name}. "
                        f"Supported hook names include: {self.supported_hook_names[:12]} ..."
                    )

                fwd_fns = fwd_by_name.get(name, [])
                bwd_fns = bwd_by_name.get(name, [])

                if target.mode == "forward":
                    handle = target.module.register_forward_hook(
                        self._make_composed_forward_hook(name, target, fwd_fns, bwd_fns)
                    )
                elif target.mode == "pre":
                    handle = target.module.register_forward_pre_hook(
                        self._make_composed_pre_hook(name, fwd_fns, bwd_fns)
                    )
                else:
                    raise RuntimeError(f"Unknown hook mode: {target.mode}")

                handles.append(handle)

            outputs = self.model(
                input_ids=run_inputs.input_ids,
                inputs_embeds=run_inputs.inputs_embeds,
                attention_mask=run_inputs.attention_mask,
                use_cache=False,
                return_dict=True,
                **run_inputs.model_kwargs,
            )

            if hasattr(outputs, "logits") and outputs.logits is not None:
                return outputs.logits
            if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                return outputs.last_hidden_state

            raise RuntimeError("HF model forward output does not expose logits or last_hidden_state")
        finally:
            for handle in reversed(handles):
                handle.remove()

    def zero_grad(self) -> None:
        self.model.zero_grad(set_to_none=True)

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return self.model.parameters()


def resolve_backend(model: Any, backend: Optional[ModelBackend]) -> ModelBackend:
    if backend is not None:
        return backend

    if _is_hooked_transformer_model(model):
        return TLensBackend(model)

    raise ValueError(
        "A backend must be provided when model is not a transformer_lens.HookedTransformer. "
        "Instantiate HFLLMBackend(model=...) or TLensBackend(model=...) and pass backend=..."
    )


def is_hf_backend(backend: ModelBackend) -> bool:
    return getattr(backend, "kind", "") == "hf"
